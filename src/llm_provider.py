"""Unified LLM provider supporting multiple free and paid backends.

Priority order:
1. Ollama (fully free, local)
2. HuggingFace transformers (fully free, local)
3. Groq (free tier: 14,400 req/day)
4. OpenAI (paid, optional)
5. Template fallback (no LLM needed)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    HF_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cached model singletons
# ---------------------------------------------------------------------------
_hf_pipeline = None
_active_provider: Optional[str] = None


def _get_hf_pipeline():
    """Lazy-load the HuggingFace text2text pipeline."""
    global _hf_pipeline
    if _hf_pipeline is None:
        from transformers import pipeline as hf_pipeline

        logger.info("Loading HuggingFace model: %s (this may take a moment)...", HF_MODEL)
        _hf_pipeline = hf_pipeline(
            "text2text-generation",
            model=HF_MODEL,
            max_new_tokens=512,
            device="cpu",
        )
        logger.info("HuggingFace model loaded successfully.")
    return _hf_pipeline


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, system: str) -> str:
    """Call Ollama REST API (free, local)."""
    import urllib.request
    import json

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": f"{system}\n\n{prompt}",
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data.get("response", "").strip()


def _call_huggingface(prompt: str, system: str) -> str:
    """Run inference with a local HuggingFace model (free, no API key)."""
    pipe = _get_hf_pipeline()
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    # flan-t5 works best with shorter prompts, truncate if needed
    if len(full_prompt) > 2048:
        full_prompt = full_prompt[:2048]
    results = pipe(full_prompt)
    return results[0]["generated_text"].strip()


def _call_groq(prompt: str, system: str) -> str:
    """Call Groq API (free tier: 14,400 requests/day)."""
    import urllib.request
    import json

    payload = json.dumps({
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": 1024,
    }).encode()

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"].strip()


def _call_openai(prompt: str, system: str) -> str:
    """Call OpenAI API (paid)."""
    import openai

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def _template_fallback(prompt: str, system: str) -> str:
    """Smart template-based answer — extracts and synthesizes info from context.

    Works without any LLM by parsing the retrieved context and building
    a structured answer using keyword matching and extraction.
    """
    import re

    lines = prompt.split("\n")
    context_lines = []
    question = ""
    capturing = False

    # Parse out context and question from the prompt
    for line in lines:
        low = line.lower().strip()
        if "context" in low and ("---" in low or ":" in low):
            capturing = True
            continue
        if "end context" in low or (low.startswith("question") and ":" in low):
            if ":" in low and "question" in low:
                question = line.split(":", 1)[-1].strip()
            capturing = False
            continue
        if capturing and line.strip():
            context_lines.append(line.strip())

    if not question:
        for line in reversed(lines):
            stripped = line.strip()
            if stripped and ("?" in stripped or stripped.lower().startswith(("what", "why", "how", "which", "when", "who", "show", "list", "summarize"))):
                question = stripped
                break

    if not question:
        question = lines[-1].strip() if lines else ""

    # If no context, return what we can
    if not context_lines:
        # Check if this is a SQL-related prompt
        if "sql" in prompt.lower() or "select" in prompt.lower():
            return prompt.split("\n")[-1].strip() if lines else "No data available."
        return f"Based on your question: {question}\n\nNo relevant context was retrieved. Please try rephrasing your question."

    # Extract source-tagged passages
    passages = []
    current_source = "unknown"
    current_text = []

    for line in context_lines:
        source_match = re.match(r'\[(\d+)\]\s*\(source:\s*([^,)]+)', line)
        if source_match:
            if current_text:
                passages.append({"source": current_source, "text": " ".join(current_text)})
            current_source = source_match.group(2).strip()
            current_text = []
        else:
            current_text.append(line)

    if current_text:
        passages.append({"source": current_source, "text": " ".join(current_text)})

    # Build a structured answer
    q_lower = question.lower()

    # Extract key terms from the question (skip stop words)
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "why", "how",
                  "which", "when", "where", "who", "do", "does", "did", "can", "could",
                  "should", "would", "will", "has", "have", "had", "be", "been", "being",
                  "in", "on", "at", "to", "for", "of", "with", "by", "from", "and", "or",
                  "not", "no", "but", "if", "then", "than", "that", "this", "it", "its",
                  "most", "show", "tell", "me", "about", "many", "much"}
    query_terms = [w for w in re.findall(r'\w+', q_lower) if w not in stop_words and len(w) > 2]

    # Score passages by relevance to the question
    scored = []
    for p in passages:
        text_lower = p["text"].lower()
        score = sum(1 for term in query_terms if term in text_lower)
        scored.append((score, p))
    scored.sort(key=lambda x: -x[0])

    # Build the answer
    parts = []
    parts.append(f"**Based on analysis of {len(passages)} retrieved document(s):**\n")

    # Take top relevant passages
    top_passages = scored[:5]
    seen_sources = set()

    for i, (score, p) in enumerate(top_passages, 1):
        text = p["text"][:300].strip()
        source = p["source"]
        if text:
            parts.append(f"**[{source}]**: {text}")
            seen_sources.add(source)

    if seen_sources:
        parts.append(f"\n*Sources: {', '.join(sorted(seen_sources))}*")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

def _detect_provider() -> str:
    """Auto-detect the best available LLM provider."""
    global _active_provider

    if _active_provider:
        return _active_provider

    # If user explicitly set a provider, try that first
    if LLM_PROVIDER and LLM_PROVIDER != "auto":
        _active_provider = LLM_PROVIDER
        logger.info("Using configured LLM provider: %s", _active_provider)
        return _active_provider

    # Auto-detect: try Ollama first (free, local)
    try:
        import urllib.request
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=2):
            _active_provider = "ollama"
            logger.info("Auto-detected Ollama running locally.")
            return _active_provider
    except Exception:
        pass

    # Try HuggingFace (free, local, always available if transformers installed)
    try:
        import transformers  # noqa: F401
        _active_provider = "huggingface"
        logger.info("Using HuggingFace transformers (local, free).")
        return _active_provider
    except ImportError:
        pass

    # Try Groq (free tier)
    if GROQ_API_KEY:
        _active_provider = "groq"
        logger.info("Using Groq API (free tier).")
        return _active_provider

    # Try OpenAI (paid)
    if OPENAI_API_KEY and OPENAI_API_KEY != "your-key-here":
        _active_provider = "openai"
        logger.info("Using OpenAI API (paid).")
        return _active_provider

    # Fallback
    _active_provider = "none"
    logger.warning("No LLM provider available. Using template fallback.")
    return _active_provider


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_llm(
    prompt: str,
    system: str = "You are a helpful enterprise assistant.",
) -> str:
    """Call the best available LLM provider.

    Tries providers in order of preference. Falls back gracefully.

    Parameters
    ----------
    prompt : str
        The user prompt / question.
    system : str
        System message for the LLM.

    Returns
    -------
    str
        The LLM response text.
    """
    provider = _detect_provider()

    providers = {
        "ollama": _call_ollama,
        "huggingface": _call_huggingface,
        "groq": _call_groq,
        "openai": _call_openai,
        "none": _template_fallback,
    }

    fn = providers.get(provider, _template_fallback)

    try:
        result = fn(prompt, system)
        return result
    except Exception as exc:
        logger.warning("Provider '%s' failed: %s. Trying fallbacks...", provider, exc)

        # Try remaining providers
        for name, fallback_fn in providers.items():
            if name == provider or name == "none":
                continue
            try:
                result = fallback_fn(prompt, system)
                global _active_provider
                _active_provider = name
                logger.info("Switched to provider: %s", name)
                return result
            except Exception:
                continue

        # Final fallback
        return _template_fallback(prompt, system)


def get_active_provider() -> str:
    """Return the name of the currently active LLM provider."""
    return _detect_provider()


if __name__ == "__main__":
    print(f"Detected provider: {get_active_provider()}")
    print()
    response = call_llm("What is 2 + 2?", "Answer concisely.")
    print(f"Response: {response}")
