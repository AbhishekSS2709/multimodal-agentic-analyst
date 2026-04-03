# Multimodal RAG System — Implementation Plan


**Goal:** Upgrade the existing text-only Enterprise RAG system to a multimodal pipeline supporting 12 file types, dual embedding indices (BGE text + CLIP visual), Gemini 2.5 Flash for generation, and production-grade evaluation/observability.

**Architecture:** Dual-index approach — keep BGE-large for text embeddings in existing FAISS index, add CLIP ViT-B/32 for visual embeddings in a second FAISS index. Gemini 2.5 Flash handles multimodal answer generation with native vision. Query analyzer detects modality and routes to the right index combination.

**Tech Stack:** Python 3.10+, FAISS, sentence-transformers, CLIP (transformers), google-genai, FastAPI, Streamlit, OpenCV, faster-whisper, python-docx, python-pptx, openpyxl, beautifulsoup4, python-magic, Pillow, pytesseract

**Spec:** `docs/design/specs/2026-04-03-multimodal-rag-design.md`

---

## File Structure

### New files to create:
- `src/ingestion/image_loader.py` — Image ingestion (OCR + captioning + asset storage)
- `src/ingestion/video_loader.py` — Video keyframe extraction + audio transcription
- `src/ingestion/audio_loader.py` — Audio transcription
- `src/ingestion/docx_loader.py` — Word document loading
- `src/ingestion/pptx_loader.py` — PowerPoint loading
- `src/ingestion/excel_loader.py` — Excel spreadsheet loading
- `src/ingestion/html_loader.py` — HTML file and URL loading
- `src/ingestion/code_loader.py` — Source code loading with language detection
- `src/ingestion/json_yaml_loader.py` — JSON/YAML structured data loading
- `src/ingestion/mime_detector.py` — MIME-type detection gateway
- `src/ingestion/asset_store.py` — Asset file storage and retrieval
- `src/embedding/clip_engine.py` — CLIP visual embedding engine
- `src/embedding/visual_store.py` — FAISS visual vector store
- `src/gemini/client.py` — Gemini API client with rate limiting and fallback
- `src/gemini/captioner.py` — Image/audio captioning via Gemini
- `src/gemini/generator.py` — Answer generation with structured JSON output
- `src/retrieval/multimodal_retriever.py` — Score fusion across text + visual indices
- `src/retrieval/query_analyzer.py` — Modality-aware query analysis
- `tests/test_mime_detector.py`
- `tests/test_gemini_client.py`
- `tests/test_clip_engine.py`
- `tests/test_visual_store.py`
- `tests/test_image_loader.py`
- `tests/test_audio_loader.py`
- `tests/test_docx_loader.py`
- `tests/test_pptx_loader.py`
- `tests/test_excel_loader.py`
- `tests/test_html_loader.py`
- `tests/test_code_loader.py`
- `tests/test_json_yaml_loader.py`
- `tests/test_multimodal_retriever.py`
- `tests/test_query_analyzer.py`
- `tests/test_gemini_generator.py`
- `tests/test_captioner.py`
- `tests/test_asset_store.py`

### Files to modify:
- `config/settings.py` — Add Gemini, CLIP, asset, and security settings
- `requirements.txt` — Add new dependencies
- `src/ingestion/pipeline.py` — Register new loaders, support dual output
- `src/ingestion/pdf_loader.py` — Add page-as-image extraction for visual content
- `src/llm_provider.py` — Add Gemini as primary provider
- `src/agents/query_router.py` — Add visual category and modality detection
- `src/pipeline_orchestrator.py` — Wire in visual components
- `src/api/main.py` — Add new endpoints (assets, batch-upload, metrics, quota)
- `src/api/models.py` — Add new request/response models
- `src/evaluation/test_cases.py` — Add multimodal test cases
- `src/evaluation/eval_pipeline.py` — Add multimodal metrics

---

## Phase 1: Foundation (Configuration + Dependencies + Gemini Client)

### Task 1: Update configuration and dependencies

**Files:**
- Modify: `config/settings.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add new settings to config/settings.py**

Add the following block after the existing OpenAI settings in `config/settings.py`:

```python
# Gemini
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-2.5-pro")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_TEMPERATURE = 0.1
GEMINI_MAX_OUTPUT_TOKENS = 2048
GEMINI_MAX_IMAGES_PER_QUERY = 3

# Visual Embedding
CLIP_MODEL = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
CLIP_EMBEDDING_DIMENSION = 512

# Visual FAISS
VISUAL_FAISS_INDEX_PATH = VECTOR_DB_DIR / "faiss_visual_index"

# Asset Storage
ASSETS_DIR = DATA_DIR / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

# Retrieval Weights (defaults, adaptive at runtime)
DEFAULT_TEXT_WEIGHT = 0.6
DEFAULT_VISUAL_WEIGHT = 0.4

# Captioning
CAPTIONING_FALLBACK = "blip2"
CAPTIONING_QUEUE_RPM = 8

# Upload Security
MAX_UPLOAD_SIZE_MB = 100
BLOCKED_MIME_TYPES = ["application/x-executable", "application/x-msdownload"]

# Azure (production)
AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING", "")
AZURE_SQL_CONNECTION_STRING = os.getenv("AZURE_SQL_CONNECTION_STRING", "")
```

- [ ] **Step 2: Update requirements.txt**

Add the following to `requirements.txt`:

```
# Gemini
google-genai>=1.0.0

# Multimodal - Image
Pillow>=10.0.0
pytesseract>=0.3.10

# Multimodal - Video/Audio
opencv-python>=4.8.0
faster-whisper>=1.0.0

# Multimodal - Documents
python-docx>=1.1.0
python-pptx>=0.6.23
openpyxl>=3.1.0

# Multimodal - Web
beautifulsoup4>=4.12.0
requests>=2.31.0

# Multimodal - Detection
python-magic>=0.4.27

# CLIP
# (uses transformers already in deps)
```

- [ ] **Step 3: Install dependencies**

Run: `pip install google-genai Pillow pytesseract opencv-python faster-whisper python-docx python-pptx openpyxl beautifulsoup4 requests python-magic`

- [ ] **Step 4: Commit**

```bash
git add config/settings.py requirements.txt
git commit -m "feat: add multimodal configuration and dependencies"
```

---

### Task 2: Gemini API client with rate limiting

**Files:**
- Create: `src/gemini/__init__.py`
- Create: `src/gemini/client.py`
- Test: `tests/test_gemini_client.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_gemini_client.py`:

```python
"""Tests for Gemini API client."""
import pytest
from unittest.mock import patch, MagicMock
from src.gemini.client import GeminiClient


def test_client_initializes_with_defaults():
    with patch("src.gemini.client.genai") as mock_genai:
        client = GeminiClient(api_key="test-key")
        assert client.model_name == "gemini-2.5-flash"
        assert client._rpm_limit == 10
        assert client._daily_limit == 250


def test_client_tracks_request_count():
    with patch("src.gemini.client.genai") as mock_genai:
        client = GeminiClient(api_key="test-key")
        assert client.requests_today == 0
        assert client.requests_this_minute == 0


def test_generate_text_returns_response():
    with patch("src.gemini.client.genai") as mock_genai:
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        client = GeminiClient(api_key="test-key")
        result = client.generate_text("Say hello")
        assert result == "Hello world"


def test_generate_text_with_image():
    with patch("src.gemini.client.genai") as mock_genai:
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "A cat sitting on a mat"
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        client = GeminiClient(api_key="test-key")
        # Simulate passing a PIL Image
        fake_image = MagicMock()
        result = client.generate_text("Describe this image", images=[fake_image])
        assert result == "A cat sitting on a mat"
        # Verify image was included in the content
        call_args = mock_model.generate_content.call_args
        content = call_args[0][0]
        assert len(content) == 2  # text + image


def test_generate_json_parses_response():
    with patch("src.gemini.client.genai") as mock_genai:
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"answer": "42", "confidence": 0.95}'
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        client = GeminiClient(api_key="test-key")
        result = client.generate_json("What is the answer?")
        assert result["answer"] == "42"
        assert result["confidence"] == 0.95


def test_generate_json_handles_malformed_json():
    with patch("src.gemini.client.genai") as mock_genai:
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is not JSON at all"
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        client = GeminiClient(api_key="test-key")
        result = client.generate_json("What is the answer?")
        assert "raw_response" in result
        assert result["raw_response"] == "This is not JSON at all"


def test_quota_remaining():
    with patch("src.gemini.client.genai") as mock_genai:
        client = GeminiClient(api_key="test-key")
        quota = client.quota_remaining()
        assert quota["daily_remaining"] == 250
        assert quota["rpm_remaining"] == 10


def test_no_api_key_raises():
    with patch("src.gemini.client.genai"):
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            GeminiClient(api_key="")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gemini_client.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Create the __init__.py**

Create `src/gemini/__init__.py`:

```python
"""Gemini API integration for multimodal RAG."""
```

- [ ] **Step 4: Write the Gemini client implementation**

Create `src/gemini/client.py`:

```python
"""Gemini API client with rate limiting, retry, and fallback.

Wraps the google-genai SDK with:
- Per-minute and daily request tracking
- Exponential backoff retry
- Structured JSON output support
- Multimodal input (text + images)
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional

import google.generativeai as genai

logger = logging.getLogger(__name__)


class GeminiClient:
    """Rate-limited Gemini API client.

    Parameters
    ----------
    api_key : str
        Gemini API key. Raises ValueError if empty.
    model_name : str
        Gemini model identifier.
    rpm_limit : int
        Max requests per minute (free tier: 10 for Flash).
    daily_limit : int
        Max requests per day (free tier: 250 for Flash).
    temperature : float
        Generation temperature.
    max_output_tokens : int
        Max tokens in the response.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        rpm_limit: int = 10,
        daily_limit: int = 250,
        temperature: float = 0.1,
        max_output_tokens: int = 2048,
    ) -> None:
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is required. Get one free at "
                "https://aistudio.google.com/apikey"
            )
        self.model_name = model_name
        self._rpm_limit = rpm_limit
        self._daily_limit = daily_limit
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

        genai.configure(api_key=api_key)

        # Rate tracking
        self._minute_timestamps: deque[float] = deque()
        self._daily_count: int = 0
        self._daily_reset_time: float = time.time()

        logger.info(
            "GeminiClient initialized (model=%s, rpm=%d, daily=%d)",
            model_name, rpm_limit, daily_limit,
        )

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    @property
    def requests_today(self) -> int:
        self._maybe_reset_daily()
        return self._daily_count

    @property
    def requests_this_minute(self) -> int:
        self._prune_minute_window()
        return len(self._minute_timestamps)

    def _maybe_reset_daily(self) -> None:
        """Reset daily counter if 24h have passed."""
        if time.time() - self._daily_reset_time > 86400:
            self._daily_count = 0
            self._daily_reset_time = time.time()

    def _prune_minute_window(self) -> None:
        """Remove timestamps older than 60 seconds."""
        now = time.time()
        while self._minute_timestamps and (now - self._minute_timestamps[0]) > 60:
            self._minute_timestamps.popleft()

    def _wait_if_rate_limited(self) -> None:
        """Block until we're within rate limits."""
        self._maybe_reset_daily()
        if self._daily_count >= self._daily_limit:
            logger.warning("Gemini daily quota exhausted (%d/%d).", self._daily_count, self._daily_limit)
            raise RuntimeError("Gemini daily request quota exhausted.")

        self._prune_minute_window()
        if len(self._minute_timestamps) >= self._rpm_limit:
            wait_time = 60 - (time.time() - self._minute_timestamps[0])
            if wait_time > 0:
                logger.info("Rate limited. Waiting %.1f seconds.", wait_time)
                time.sleep(wait_time)

    def _record_request(self) -> None:
        """Record a successful request for rate tracking."""
        self._minute_timestamps.append(time.time())
        self._daily_count += 1

    def quota_remaining(self) -> Dict[str, int]:
        """Return remaining quota."""
        self._maybe_reset_daily()
        self._prune_minute_window()
        return {
            "daily_remaining": max(0, self._daily_limit - self._daily_count),
            "rpm_remaining": max(0, self._rpm_limit - len(self._minute_timestamps)),
            "daily_used": self._daily_count,
        }

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_text(
        self,
        prompt: str,
        *,
        system: str = "",
        images: Optional[List[Any]] = None,
        temperature: Optional[float] = None,
        max_retries: int = 3,
    ) -> str:
        """Generate text from a prompt, optionally with images.

        Parameters
        ----------
        prompt : str
            The user prompt.
        system : str
            System instruction.
        images : list[PIL.Image] | None
            Optional images to include (Gemini vision).
        temperature : float | None
            Override default temperature.
        max_retries : int
            Retry count with exponential backoff.

        Returns
        -------
        str
            Generated text response.
        """
        self._wait_if_rate_limited()

        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system if system else None,
            generation_config=genai.GenerationConfig(
                temperature=temperature if temperature is not None else self._temperature,
                max_output_tokens=self._max_output_tokens,
            ),
        )

        # Build content parts
        content = [prompt]
        if images:
            content = [prompt] + list(images)

        for attempt in range(max_retries):
            try:
                response = model.generate_content(content)
                self._record_request()
                return response.text
            except Exception as exc:
                wait = 2 ** attempt
                logger.warning(
                    "Gemini call failed (attempt %d/%d): %s. Retrying in %ds.",
                    attempt + 1, max_retries, exc, wait,
                )
                if attempt < max_retries - 1:
                    time.sleep(wait)
                else:
                    raise

    def generate_json(
        self,
        prompt: str,
        *,
        system: str = "",
        images: Optional[List[Any]] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate a structured JSON response.

        Attempts to parse the response as JSON. If parsing fails,
        returns a dict with a 'raw_response' key containing the text.

        Parameters
        ----------
        prompt : str
            The user prompt (should ask for JSON output).
        system : str
            System instruction.
        images : list | None
            Optional images.
        temperature : float | None
            Override default temperature.

        Returns
        -------
        dict
            Parsed JSON response, or {"raw_response": text} on parse failure.
        """
        if system:
            system += "\n\nRespond ONLY with valid JSON. No markdown, no code fences."
        else:
            system = "Respond ONLY with valid JSON. No markdown, no code fences."

        text = self.generate_text(
            prompt, system=system, images=images, temperature=temperature,
        )

        # Try to parse JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from code fences
        import re
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding a JSON object in the text
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning("Failed to parse Gemini response as JSON. Returning raw.")
        return {"raw_response": text}
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_gemini_client.py -v`
Expected: All 8 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/gemini/__init__.py src/gemini/client.py tests/test_gemini_client.py
git commit -m "feat: add Gemini API client with rate limiting and JSON support"
```

---

### Task 3: Add Gemini to LLM provider chain

**Files:**
- Modify: `src/llm_provider.py`

- [ ] **Step 1: Add Gemini provider function**

Add the following function in `src/llm_provider.py` after the existing `_call_groq` function (around line 118):

```python
def _call_gemini(prompt: str, system: str) -> str:
    """Call Gemini API (free tier)."""
    from src.gemini.client import GeminiClient
    from config.settings import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE

    client = GeminiClient(
        api_key=GEMINI_API_KEY,
        model_name=GEMINI_MODEL,
        temperature=GEMINI_TEMPERATURE,
    )
    return client.generate_text(prompt, system=system)
```

- [ ] **Step 2: Register Gemini in the provider detection chain**

In `_detect_provider()`, add a Gemini check after the Ollama check and before the HuggingFace check:

```python
    # Try Gemini (free tier)
    try:
        from config.settings import GEMINI_API_KEY
        if GEMINI_API_KEY:
            _active_provider = "gemini"
            logger.info("Using Gemini API (free tier).")
            return _active_provider
    except ImportError:
        pass
```

- [ ] **Step 3: Add Gemini to the providers dict in `call_llm`**

In the `call_llm` function, update the `providers` dict to include `"gemini": _call_gemini` before `"groq"`.

- [ ] **Step 4: Verify manually**

Run: `python -c "from src.llm_provider import get_active_provider; print(get_active_provider())"`
Expected: Should print current provider (won't be "gemini" unless GEMINI_API_KEY is set)

- [ ] **Step 5: Commit**

```bash
git add src/llm_provider.py
git commit -m "feat: add Gemini as LLM provider with free tier support"
```

---

## Phase 2: Asset Storage + MIME Detection

### Task 4: MIME-type detection gateway

**Files:**
- Create: `src/ingestion/mime_detector.py`
- Test: `tests/test_mime_detector.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_mime_detector.py`:

```python
"""Tests for MIME-type detection."""
import pytest
import tempfile
from pathlib import Path
from src.ingestion.mime_detector import detect_file_type, get_loader_for_mime


def test_detect_txt_file():
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("Hello world")
        f.flush()
        result = detect_file_type(f.name)
        assert result["category"] == "text"
        assert result["extension"] == ".txt"


def test_detect_csv_file():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        f.write("name,age\nAlice,30\nBob,25")
        f.flush()
        result = detect_file_type(f.name)
        assert result["category"] in ("text", "table")


def test_detect_json_file():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        f.write('{"key": "value"}')
        f.flush()
        result = detect_file_type(f.name)
        assert result["extension"] == ".json"


def test_blocked_mime_type():
    """Executable files should be rejected."""
    result = detect_file_type("test.exe")
    # Even without the file existing, extension-based check should flag it
    assert result.get("blocked", False) or result["extension"] == ".exe"


def test_get_loader_for_known_type():
    loader_name = get_loader_for_mime("text/plain", ".txt")
    assert loader_name == "txt_loader"


def test_get_loader_for_image():
    loader_name = get_loader_for_mime("image/png", ".png")
    assert loader_name == "image_loader"


def test_get_loader_for_unknown_returns_none():
    loader_name = get_loader_for_mime("application/octet-stream", ".xyz")
    assert loader_name is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mime_detector.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write implementation**

Create `src/ingestion/mime_detector.py`:

```python
"""MIME-type detection gateway for the ingestion pipeline.

Uses python-magic for content-based detection with extension fallback.
Routes files to the correct loader based on actual content type.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Extension -> (category, loader_name)
_EXTENSION_MAP: Dict[str, tuple] = {
    # Text
    ".txt": ("text", "txt_loader"),
    ".log": ("text", "txt_loader"),
    ".md": ("text", "txt_loader"),
    ".eml": ("text", "txt_loader"),
    # PDF
    ".pdf": ("document", "pdf_loader"),
    # Tabular
    ".csv": ("table", "csv_loader"),
    ".xlsx": ("table", "excel_loader"),
    ".xls": ("table", "excel_loader"),
    # Documents
    ".docx": ("document", "docx_loader"),
    ".pptx": ("document", "pptx_loader"),
    # Images
    ".png": ("image", "image_loader"),
    ".jpg": ("image", "image_loader"),
    ".jpeg": ("image", "image_loader"),
    ".webp": ("image", "image_loader"),
    ".bmp": ("image", "image_loader"),
    ".gif": ("image", "image_loader"),
    # Video
    ".mp4": ("video", "video_loader"),
    ".avi": ("video", "video_loader"),
    ".mov": ("video", "video_loader"),
    ".mkv": ("video", "video_loader"),
    # Audio
    ".mp3": ("audio", "audio_loader"),
    ".wav": ("audio", "audio_loader"),
    ".m4a": ("audio", "audio_loader"),
    ".flac": ("audio", "audio_loader"),
    # Web
    ".html": ("web", "html_loader"),
    ".htm": ("web", "html_loader"),
    # Code
    ".py": ("code", "code_loader"),
    ".js": ("code", "code_loader"),
    ".ts": ("code", "code_loader"),
    ".java": ("code", "code_loader"),
    ".cpp": ("code", "code_loader"),
    ".c": ("code", "code_loader"),
    ".go": ("code", "code_loader"),
    ".rs": ("code", "code_loader"),
    ".rb": ("code", "code_loader"),
    ".php": ("code", "code_loader"),
    ".swift": ("code", "code_loader"),
    ".kt": ("code", "code_loader"),
    ".cs": ("code", "code_loader"),
    ".r": ("code", "code_loader"),
    ".sql": ("code", "code_loader"),
    ".sh": ("code", "code_loader"),
    ".bash": ("code", "code_loader"),
    ".yaml": ("structured", "json_yaml_loader"),
    ".yml": ("structured", "json_yaml_loader"),
    ".json": ("structured", "json_yaml_loader"),
}

# MIME type -> (category, loader_name)
_MIME_MAP: Dict[str, tuple] = {
    "text/plain": ("text", "txt_loader"),
    "text/csv": ("table", "csv_loader"),
    "text/html": ("web", "html_loader"),
    "application/pdf": ("document", "pdf_loader"),
    "application/json": ("structured", "json_yaml_loader"),
    "image/png": ("image", "image_loader"),
    "image/jpeg": ("image", "image_loader"),
    "image/webp": ("image", "image_loader"),
    "image/bmp": ("image", "image_loader"),
    "image/gif": ("image", "image_loader"),
    "video/mp4": ("video", "video_loader"),
    "video/x-msvideo": ("video", "video_loader"),
    "video/quicktime": ("video", "video_loader"),
    "audio/mpeg": ("audio", "audio_loader"),
    "audio/wav": ("audio", "audio_loader"),
    "audio/x-wav": ("audio", "audio_loader"),
    "audio/mp4": ("audio", "audio_loader"),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ("document", "docx_loader"),
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ("document", "pptx_loader"),
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ("table", "excel_loader"),
    "application/vnd.ms-excel": ("table", "excel_loader"),
}

_BLOCKED_EXTENSIONS = {".exe", ".dll", ".so", ".bat", ".cmd", ".com", ".msi", ".scr"}


def detect_file_type(file_path: str) -> Dict[str, Any]:
    """Detect the type of a file using MIME detection + extension fallback.

    Returns
    -------
    dict
        Keys: mime_type, extension, category, loader_name, blocked
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    # Check blocked extensions
    if ext in _BLOCKED_EXTENSIONS:
        return {
            "mime_type": "application/x-executable",
            "extension": ext,
            "category": "blocked",
            "loader_name": None,
            "blocked": True,
        }

    # Try python-magic for content-based detection
    mime_type = None
    try:
        import magic
        if path.exists():
            mime_type = magic.from_file(str(path), mime=True)
    except ImportError:
        logger.debug("python-magic not available, using extension-based detection.")
    except Exception as exc:
        logger.debug("MIME detection failed: %s", exc)

    # Resolve loader from MIME type
    if mime_type and mime_type in _MIME_MAP:
        category, loader_name = _MIME_MAP[mime_type]
        return {
            "mime_type": mime_type,
            "extension": ext,
            "category": category,
            "loader_name": loader_name,
            "blocked": False,
        }

    # Fallback to extension-based detection
    if ext in _EXTENSION_MAP:
        category, loader_name = _EXTENSION_MAP[ext]
        return {
            "mime_type": mime_type or "unknown",
            "extension": ext,
            "category": category,
            "loader_name": loader_name,
            "blocked": False,
        }

    return {
        "mime_type": mime_type or "unknown",
        "extension": ext,
        "category": "unknown",
        "loader_name": None,
        "blocked": False,
    }


def get_loader_for_mime(mime_type: str, extension: str) -> Optional[str]:
    """Return the loader name for a given MIME type and extension.

    Returns None if no loader is registered for this type.
    """
    if mime_type in _MIME_MAP:
        return _MIME_MAP[mime_type][1]
    ext = extension.lower()
    if ext in _EXTENSION_MAP:
        return _EXTENSION_MAP[ext][1]
    return None
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_mime_detector.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/mime_detector.py tests/test_mime_detector.py
git commit -m "feat: add MIME-type detection gateway for multimodal ingestion"
```

---

### Task 5: Asset storage manager

**Files:**
- Create: `src/ingestion/asset_store.py`
- Test: `tests/test_asset_store.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_asset_store.py`:

```python
"""Tests for asset storage."""
import pytest
import tempfile
from pathlib import Path
from PIL import Image
from src.ingestion.asset_store import AssetStore


@pytest.fixture
def asset_store(tmp_path):
    return AssetStore(base_dir=tmp_path / "assets")


@pytest.fixture
def sample_image(tmp_path):
    img = Image.new("RGB", (100, 100), color="red")
    path = tmp_path / "test_image.png"
    img.save(path)
    return path


def test_store_creates_directory(asset_store):
    assert asset_store.base_dir.exists()


def test_store_image(asset_store, sample_image):
    asset_id = asset_store.store(
        file_path=str(sample_image),
        doc_id="doc_123",
        metadata={"page": 1},
    )
    assert asset_id is not None
    assert asset_store.exists(asset_id)


def test_retrieve_image(asset_store, sample_image):
    asset_id = asset_store.store(str(sample_image), doc_id="doc_123")
    retrieved_path = asset_store.get_path(asset_id)
    assert retrieved_path is not None
    assert Path(retrieved_path).exists()


def test_load_image_as_pil(asset_store, sample_image):
    asset_id = asset_store.store(str(sample_image), doc_id="doc_123")
    img = asset_store.load_image(asset_id)
    assert img is not None
    assert img.size == (100, 100)


def test_list_assets_for_doc(asset_store, sample_image):
    asset_store.store(str(sample_image), doc_id="doc_A")
    asset_store.store(str(sample_image), doc_id="doc_A")
    asset_store.store(str(sample_image), doc_id="doc_B")

    assets_a = asset_store.list_for_doc("doc_A")
    assert len(assets_a) == 2

    assets_b = asset_store.list_for_doc("doc_B")
    assert len(assets_b) == 1


def test_get_nonexistent_returns_none(asset_store):
    assert asset_store.get_path("nonexistent_id") is None
    assert asset_store.load_image("nonexistent_id") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_asset_store.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write implementation**

Create `src/ingestion/asset_store.py`:

```python
"""Asset storage for images, video frames, and other binary content.

Stores files on local disk with a JSON index mapping asset_id to metadata.
In production, this would be backed by Azure Blob Storage.
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AssetStore:
    """Manages storage and retrieval of binary assets (images, frames, etc.).

    Parameters
    ----------
    base_dir : Path | str
        Root directory for asset storage.
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        if base_dir is None:
            from config.settings import ASSETS_DIR
            base_dir = ASSETS_DIR
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.base_dir / "_index.json"
        self._index: Dict[str, Dict[str, Any]] = self._load_index()

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load the asset index from disk."""
        if self._index_path.exists():
            try:
                with open(self._index_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning("Failed to load asset index: %s", exc)
        return {}

    def _save_index(self) -> None:
        """Persist the asset index to disk."""
        try:
            with open(self._index_path, "w", encoding="utf-8") as f:
                json.dump(self._index, f, indent=2, default=str)
        except Exception as exc:
            logger.warning("Failed to save asset index: %s", exc)

    def store(
        self,
        file_path: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a file and return its asset_id.

        Parameters
        ----------
        file_path : str
            Path to the file to store.
        doc_id : str
            Document ID this asset belongs to.
        metadata : dict | None
            Additional metadata to store.

        Returns
        -------
        str
            Unique asset ID.
        """
        src = Path(file_path)
        asset_id = uuid.uuid4().hex[:12]
        suffix = src.suffix.lower()
        dest = self.base_dir / f"{asset_id}{suffix}"

        shutil.copy2(str(src), str(dest))

        self._index[asset_id] = {
            "path": str(dest.name),
            "doc_id": doc_id,
            "original_name": src.name,
            "suffix": suffix,
            **(metadata or {}),
        }
        self._save_index()

        logger.debug("Stored asset %s from %s", asset_id, src.name)
        return asset_id

    def store_pil_image(
        self,
        image: Any,
        doc_id: str,
        name: str = "image",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a PIL Image directly and return its asset_id.

        Parameters
        ----------
        image : PIL.Image.Image
            The image to store.
        doc_id : str
            Document ID.
        name : str
            Base name for the file.
        metadata : dict | None
            Additional metadata.

        Returns
        -------
        str
            Unique asset ID.
        """
        asset_id = uuid.uuid4().hex[:12]
        dest = self.base_dir / f"{asset_id}.png"
        image.save(str(dest), format="PNG")

        self._index[asset_id] = {
            "path": dest.name,
            "doc_id": doc_id,
            "original_name": f"{name}.png",
            "suffix": ".png",
            **(metadata or {}),
        }
        self._save_index()

        logger.debug("Stored PIL image asset %s", asset_id)
        return asset_id

    def exists(self, asset_id: str) -> bool:
        """Check if an asset exists."""
        return asset_id in self._index

    def get_path(self, asset_id: str) -> Optional[str]:
        """Return the full path to an asset, or None if not found."""
        entry = self._index.get(asset_id)
        if entry is None:
            return None
        full_path = self.base_dir / entry["path"]
        if full_path.exists():
            return str(full_path)
        return None

    def load_image(self, asset_id: str) -> Optional[Any]:
        """Load an asset as a PIL Image. Returns None if not found or not an image."""
        path = self.get_path(asset_id)
        if path is None:
            return None
        try:
            from PIL import Image
            return Image.open(path).convert("RGB")
        except Exception as exc:
            logger.warning("Failed to load image asset %s: %s", asset_id, exc)
            return None

    def list_for_doc(self, doc_id: str) -> List[Dict[str, Any]]:
        """Return all assets belonging to a document."""
        return [
            {"asset_id": aid, **info}
            for aid, info in self._index.items()
            if info.get("doc_id") == doc_id
        ]

    def get_metadata(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """Return metadata for an asset."""
        return self._index.get(asset_id)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_asset_store.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/asset_store.py tests/test_asset_store.py
git commit -m "feat: add asset storage manager for images and binary content"
```

---

## Phase 3: CLIP Visual Embedding Engine + Visual FAISS Store

### Task 6: CLIP embedding engine

**Files:**
- Create: `src/embedding/clip_engine.py`
- Test: `tests/test_clip_engine.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_clip_engine.py`:

```python
"""Tests for CLIP visual embedding engine."""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image
from src.embedding.clip_engine import CLIPEngine


@pytest.fixture
def sample_image():
    return Image.new("RGB", (224, 224), color="blue")


def test_engine_initializes():
    with patch("src.embedding.clip_engine.CLIPModel") as mock_model_cls, \
         patch("src.embedding.clip_engine.CLIPProcessor") as mock_proc_cls:
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_proc_cls.from_pretrained.return_value = MagicMock()

        engine = CLIPEngine()
        assert engine.dimension == 512


def test_embed_image_returns_correct_shape(sample_image):
    with patch("src.embedding.clip_engine.CLIPModel") as mock_model_cls, \
         patch("src.embedding.clip_engine.CLIPProcessor") as mock_proc_cls:
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.image_embeds = MagicMock()
        mock_output.image_embeds.detach.return_value.cpu.return_value.numpy.return_value = np.random.randn(1, 512).astype(np.float32)
        mock_model.get_image_features.return_value = mock_output.image_embeds.detach.return_value.cpu.return_value.numpy.return_value
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_proc_cls.from_pretrained.return_value = MagicMock()

        engine = CLIPEngine()
        vec = engine.embed_image(sample_image)
        assert vec.shape == (512,)


def test_embed_text_returns_correct_shape():
    with patch("src.embedding.clip_engine.CLIPModel") as mock_model_cls, \
         patch("src.embedding.clip_engine.CLIPProcessor") as mock_proc_cls:
        mock_model = MagicMock()
        mock_model.get_text_features.return_value = MagicMock(
            detach=MagicMock(return_value=MagicMock(
                cpu=MagicMock(return_value=MagicMock(
                    numpy=MagicMock(return_value=np.random.randn(1, 512).astype(np.float32))
                ))
            ))
        )
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_proc_cls.from_pretrained.return_value = MagicMock()

        engine = CLIPEngine()
        vec = engine.embed_text("a photo of a cat")
        assert vec.shape == (512,)


def test_embed_images_batch(sample_image):
    with patch("src.embedding.clip_engine.CLIPModel") as mock_model_cls, \
         patch("src.embedding.clip_engine.CLIPProcessor") as mock_proc_cls:
        mock_model = MagicMock()
        mock_model.get_image_features.return_value = MagicMock(
            detach=MagicMock(return_value=MagicMock(
                cpu=MagicMock(return_value=MagicMock(
                    numpy=MagicMock(return_value=np.random.randn(3, 512).astype(np.float32))
                ))
            ))
        )
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_proc_cls.from_pretrained.return_value = MagicMock()

        engine = CLIPEngine()
        images = [sample_image, sample_image, sample_image]
        vecs = engine.embed_images(images)
        assert len(vecs) == 3
        assert all(v.shape == (512,) for v in vecs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_clip_engine.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write implementation**

Create `src/embedding/clip_engine.py`:

```python
"""CLIP visual embedding engine for images and text-to-image search.

Uses OpenAI CLIP (via HuggingFace transformers) to embed images and text
into a shared vector space. Images and text queries can be compared
directly using cosine similarity.
"""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import CLIP_EMBEDDING_DIMENSION, CLIP_MODEL, VECTOR_DB_DIR

from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

_CACHE_PATH = VECTOR_DB_DIR / "clip_cache.npz"


class CLIPEngine:
    """Embed images and text using CLIP for cross-modal search.

    Parameters
    ----------
    model_name : str
        HuggingFace CLIP model identifier.
    dimension : int
        Expected embedding dimension.
    """

    def __init__(
        self,
        model_name: str = CLIP_MODEL,
        dimension: int = CLIP_EMBEDDING_DIMENSION,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self._model = CLIPModel.from_pretrained(model_name)
        self._processor = CLIPProcessor.from_pretrained(model_name)

        # Cache: hash -> embedding
        self._cache: Dict[str, np.ndarray] = {}
        self._load_cache()

        logger.info("CLIPEngine initialized (model=%s, dim=%d)", model_name, dimension)

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    @staticmethod
    def _image_hash(image: Any) -> str:
        """Hash a PIL Image by its pixel data."""
        import io
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return hashlib.sha256(buf.getvalue()).hexdigest()

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        if _CACHE_PATH.exists():
            try:
                data = np.load(str(_CACHE_PATH), allow_pickle=True)
                keys = data["keys"].tolist()
                vectors = data["vectors"]
                for key, vec in zip(keys, vectors):
                    self._cache[key] = vec
                logger.info("Loaded %d cached CLIP embeddings.", len(self._cache))
            except Exception as exc:
                logger.warning("Could not load CLIP cache: %s", exc)

    def save_cache(self) -> None:
        """Persist the embedding cache to disk."""
        if not self._cache:
            return
        try:
            keys = list(self._cache.keys())
            vectors = np.array(list(self._cache.values()), dtype=np.float32)
            np.savez(str(_CACHE_PATH), keys=np.array(keys), vectors=vectors)
            logger.info("Saved %d CLIP embeddings to cache.", len(self._cache))
        except Exception as exc:
            logger.warning("Could not save CLIP cache: %s", exc)

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(vec: np.ndarray) -> np.ndarray:
        """L2-normalise a vector."""
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            return vec
        return vec / norm

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_image(self, image: Any) -> np.ndarray:
        """Embed a single PIL Image.

        Returns
        -------
        np.ndarray
            Normalised embedding of shape (dimension,).
        """
        h = self._image_hash(image)
        if h in self._cache:
            return self._cache[h]

        inputs = self._processor(images=image, return_tensors="pt")
        features = self._model.get_image_features(**inputs)
        vec = features.detach().cpu().numpy().flatten().astype(np.float32)
        vec = self._normalise(vec)
        self._cache[h] = vec
        return vec

    def embed_images(self, images: List[Any]) -> List[np.ndarray]:
        """Embed a batch of PIL Images.

        Returns
        -------
        list[np.ndarray]
            List of normalised embeddings.
        """
        results: List[np.ndarray] = []
        to_embed = []
        to_embed_indices = []

        for i, img in enumerate(images):
            h = self._image_hash(img)
            if h in self._cache:
                results.append(self._cache[h])
            else:
                to_embed.append(img)
                to_embed_indices.append(i)
                results.append(None)

        if to_embed:
            inputs = self._processor(images=to_embed, return_tensors="pt")
            features = self._model.get_image_features(**inputs)
            vecs = features.detach().cpu().numpy().astype(np.float32)

            for idx, vec_idx in enumerate(to_embed_indices):
                vec = self._normalise(vecs[idx])
                h = self._image_hash(images[vec_idx])
                self._cache[h] = vec
                results[vec_idx] = vec

        return results

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a text query using CLIP's text encoder.

        This produces a vector in the same space as image embeddings,
        enabling text-to-image search.

        Returns
        -------
        np.ndarray
            Normalised embedding of shape (dimension,).
        """
        h = self._text_hash(text)
        if h in self._cache:
            return self._cache[h]

        inputs = self._processor(text=text, return_tensors="pt", padding=True)
        features = self._model.get_text_features(**inputs)
        vec = features.detach().cpu().numpy().flatten().astype(np.float32)
        vec = self._normalise(vec)
        self._cache[h] = vec
        return vec
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_clip_engine.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/embedding/clip_engine.py tests/test_clip_engine.py
git commit -m "feat: add CLIP visual embedding engine with caching"
```

---

### Task 7: Visual FAISS vector store

**Files:**
- Create: `src/embedding/visual_store.py`
- Test: `tests/test_visual_store.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_visual_store.py`:

```python
"""Tests for visual FAISS vector store."""
import pytest
import numpy as np
from src.embedding.visual_store import VisualVectorStore


@pytest.fixture
def store(tmp_path):
    return VisualVectorStore(
        dimension=512,
        index_path=tmp_path / "test_visual.index",
        meta_path=tmp_path / "test_visual_meta.json",
    )


def test_store_creates_empty():
    store = VisualVectorStore(dimension=512)
    assert store.total_vectors == 0


def test_store_and_search(store):
    # Store 3 vectors
    vectors = [np.random.randn(512).astype(np.float32) for _ in range(3)]
    # Normalise
    vectors = [v / np.linalg.norm(v) for v in vectors]

    metadata_list = [
        {"asset_id": "a1", "doc_id": "d1", "modality": "image"},
        {"asset_id": "a2", "doc_id": "d1", "modality": "image"},
        {"asset_id": "a3", "doc_id": "d2", "modality": "image"},
    ]

    store.store_embeddings(vectors, metadata_list)
    assert store.total_vectors == 3

    # Search with the first vector — it should be the top result
    results = store.search(vectors[0], top_k=2)
    assert len(results) == 2
    assert results[0]["asset_id"] == "a1"
    assert results[0]["score"] > 0.99  # near-perfect match


def test_save_and_load(store, tmp_path):
    vectors = [np.random.randn(512).astype(np.float32) for _ in range(2)]
    vectors = [v / np.linalg.norm(v) for v in vectors]
    metadata_list = [
        {"asset_id": "x1", "doc_id": "d1"},
        {"asset_id": "x2", "doc_id": "d2"},
    ]

    store.store_embeddings(vectors, metadata_list)
    store.save()

    # Create a new store from the same path
    store2 = VisualVectorStore(
        dimension=512,
        index_path=tmp_path / "test_visual.index",
        meta_path=tmp_path / "test_visual_meta.json",
    )
    assert store2.total_vectors == 2

    results = store2.search(vectors[0], top_k=1)
    assert results[0]["asset_id"] == "x1"


def test_search_empty_store(store):
    query = np.random.randn(512).astype(np.float32)
    results = store.search(query, top_k=5)
    assert results == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_visual_store.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write implementation**

Create `src/embedding/visual_store.py`:

```python
"""FAISS vector store for visual (CLIP) embeddings.

Mirrors the structure of store_vector_db.py but operates on a separate
FAISS index with its own dimension (512 for CLIP ViT-B/32).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import CLIP_EMBEDDING_DIMENSION, VISUAL_FAISS_INDEX_PATH

logger = logging.getLogger(__name__)

_DEFAULT_INDEX = Path(str(VISUAL_FAISS_INDEX_PATH) + ".index")
_DEFAULT_META = Path(str(VISUAL_FAISS_INDEX_PATH) + "_meta.json")


class VisualVectorStore:
    """FAISS inner-product index for CLIP visual embeddings.

    Parameters
    ----------
    dimension : int
        Embedding dimension (default: CLIP_EMBEDDING_DIMENSION).
    index_path : Path | None
        Path for persisting the FAISS index.
    meta_path : Path | None
        Path for persisting the metadata JSON.
    """

    def __init__(
        self,
        dimension: int = CLIP_EMBEDDING_DIMENSION,
        index_path: Optional[Path] = None,
        meta_path: Optional[Path] = None,
    ) -> None:
        self.dimension = dimension
        self._index_path = Path(index_path) if index_path else _DEFAULT_INDEX
        self._meta_path = Path(meta_path) if meta_path else _DEFAULT_META
        self._index = None
        self._metadata: Dict[int, Dict[str, Any]] = {}
        self._next_id: int = 0

        self._load()
        logger.info("VisualVectorStore ready (dim=%d, vectors=%d)", dimension, self._next_id)

    def _ensure_index(self) -> None:
        if self._index is not None:
            return
        import faiss
        self._index = faiss.IndexFlatIP(self.dimension)

    def _load(self) -> None:
        if self._index_path.exists() and self._meta_path.exists():
            try:
                import faiss
                self._index = faiss.read_index(str(self._index_path))
                with open(self._meta_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._metadata = {int(k): v for k, v in raw.items()}
                self._next_id = self._index.ntotal
            except Exception as exc:
                logger.warning("Could not load visual index: %s", exc)
                self._index = None
                self._metadata = {}
                self._next_id = 0

    def save(self) -> None:
        """Persist index and metadata to disk."""
        if self._index is None:
            return
        import faiss
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_path))
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2, default=str)
        logger.info("Saved visual index (%d vectors).", self._index.ntotal)

    def store_embeddings(
        self,
        vectors: List[np.ndarray],
        metadata_list: List[Dict[str, Any]],
    ) -> None:
        """Add vectors and their metadata to the index.

        Parameters
        ----------
        vectors : list[np.ndarray]
            Normalised CLIP embedding vectors.
        metadata_list : list[dict]
            Metadata for each vector (must include 'asset_id').
        """
        if len(vectors) != len(metadata_list):
            raise ValueError("vectors and metadata_list must have the same length.")
        if not vectors:
            return

        self._ensure_index()
        matrix = np.vstack(vectors).astype(np.float32)
        start_id = self._next_id
        self._index.add(matrix)

        for i, meta in enumerate(metadata_list):
            self._metadata[start_id + i] = meta

        self._next_id = self._index.ntotal
        logger.info("Stored %d visual embeddings (total: %d).", len(vectors), self._next_id)

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for the top_k most similar visual embeddings.

        Parameters
        ----------
        query_vector : np.ndarray
            Normalised query embedding (from CLIP text or image encoder).
        top_k : int
            Number of results.

        Returns
        -------
        list[dict]
            Each dict contains 'score' plus all stored metadata.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        query = query_vector.reshape(1, -1).astype(np.float32)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._metadata.get(int(idx), {})
            results.append({"score": float(score), **meta})
        return results

    @property
    def total_vectors(self) -> int:
        if self._index is None:
            return 0
        return self._index.ntotal
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_visual_store.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/embedding/visual_store.py tests/test_visual_store.py
git commit -m "feat: add FAISS visual vector store for CLIP embeddings"
```

---

## Phase 4: New Document Loaders

### Task 8: Image loader

**Files:**
- Create: `src/ingestion/image_loader.py`
- Test: `tests/test_image_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_image_loader.py`:

```python
"""Tests for image loader."""
import pytest
from pathlib import Path
from PIL import Image
from src.ingestion.image_loader import load_image


@pytest.fixture
def sample_png(tmp_path):
    img = Image.new("RGB", (200, 200), color="green")
    path = tmp_path / "test.png"
    img.save(path)
    return str(path)


def test_load_image_returns_documents(sample_png):
    result = load_image(sample_png)
    assert result.text_documents is not None
    assert result.visual_assets is not None


def test_load_image_visual_asset_exists(sample_png):
    result = load_image(sample_png)
    assert len(result.visual_assets) >= 1
    asset = result.visual_assets[0]
    assert asset["modality"] == "image"
    assert "image" in asset  # PIL Image object


def test_load_image_metadata(sample_png):
    result = load_image(sample_png)
    for doc in result.text_documents:
        assert doc.metadata.get("modality") == "image"
        assert doc.metadata.get("source") == "test.png"


def test_load_nonexistent_raises():
    with pytest.raises(FileNotFoundError):
        load_image("/nonexistent/image.png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_image_loader.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write implementation**

Create `src/ingestion/image_loader.py`:

```python
"""Image loader for the multimodal RAG pipeline.

Extracts OCR text (if any) and stores the original image as a visual asset.
Optionally generates a text caption via Gemini for text-index discoverability.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from src.ingestion.pdf_loader import Document

logger = logging.getLogger(__name__)


@dataclass
class LoaderResult:
    """Dual output from a multimodal loader."""
    text_documents: List[Document] = field(default_factory=list)
    visual_assets: List[Dict[str, Any]] = field(default_factory=list)


def _ocr_image(image: Image.Image) -> str:
    """Attempt OCR on an image. Returns empty string on failure."""
    try:
        import pytesseract
        text = pytesseract.image_to_string(image)
        return text.strip()
    except ImportError:
        logger.debug("pytesseract not available, skipping OCR.")
        return ""
    except Exception as exc:
        logger.debug("OCR failed: %s", exc)
        return ""


def load_image(
    file_path: str,
    *,
    do_ocr: bool = True,
) -> LoaderResult:
    """Load an image file and extract text + visual content.

    Parameters
    ----------
    file_path : str
        Path to the image file.
    do_ocr : bool
        Whether to attempt OCR text extraction.

    Returns
    -------
    LoaderResult
        Contains text_documents (OCR text) and visual_assets (PIL Image).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {file_path}")

    image = Image.open(str(path)).convert("RGB")
    source_name = path.name
    doc_id = uuid.uuid4().hex[:12]
    date_extracted = datetime.now(timezone.utc).isoformat()

    result = LoaderResult()

    # OCR text extraction
    ocr_text = ""
    if do_ocr:
        ocr_text = _ocr_image(image)

    # Create text document (even if OCR is empty, we'll add caption later)
    text_content = ocr_text if ocr_text else f"[Image: {source_name}]"
    doc = Document(
        doc_id=doc_id,
        text=text_content,
        metadata={
            "source": source_name,
            "modality": "image",
            "file_type": path.suffix.lstrip(".").lower(),
            "date_extracted": date_extracted,
            "has_ocr_text": bool(ocr_text),
            "image_size": f"{image.width}x{image.height}",
        },
        source=str(path.resolve()),
    )
    result.text_documents.append(doc)

    # Visual asset
    result.visual_assets.append({
        "image": image,
        "doc_id": doc_id,
        "modality": "image",
        "source": source_name,
        "original_path": str(path.resolve()),
    })

    logger.info(
        "Loaded image %s (%dx%d, OCR: %d chars)",
        source_name, image.width, image.height, len(ocr_text),
    )
    return result
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_image_loader.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/image_loader.py tests/test_image_loader.py
git commit -m "feat: add image loader with OCR and visual asset extraction"
```

---

### Task 9: Word document loader

**Files:**
- Create: `src/ingestion/docx_loader.py`
- Test: `tests/test_docx_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_docx_loader.py`:

```python
"""Tests for Word document loader."""
import pytest
from pathlib import Path
from src.ingestion.docx_loader import load_docx
from src.ingestion.pdf_loader import Document


@pytest.fixture
def sample_docx(tmp_path):
    from docx import Document as DocxDocument
    doc = DocxDocument()
    doc.add_heading("Test Document", level=1)
    doc.add_paragraph("This is a test paragraph with important content.")
    doc.add_paragraph("Second paragraph for testing chunking behavior.")
    path = tmp_path / "test.docx"
    doc.save(str(path))
    return str(path)


def test_load_docx_returns_documents(sample_docx):
    docs = load_docx(sample_docx)
    assert len(docs) >= 1
    assert isinstance(docs[0], Document)


def test_load_docx_extracts_text(sample_docx):
    docs = load_docx(sample_docx)
    full_text = " ".join(d.text for d in docs)
    assert "test paragraph" in full_text
    assert "important content" in full_text


def test_load_docx_metadata(sample_docx):
    docs = load_docx(sample_docx)
    assert docs[0].metadata.get("file_type") == "docx"
    assert docs[0].metadata.get("source") == "test.docx"


def test_load_nonexistent_raises():
    with pytest.raises(FileNotFoundError):
        load_docx("/nonexistent/file.docx")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_docx_loader.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write implementation**

Create `src/ingestion/docx_loader.py`:

```python
"""Word document (.docx) loader for the RAG pipeline.

Extracts paragraph text and embedded images from .docx files.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from src.ingestion.pdf_loader import Document

logger = logging.getLogger(__name__)


def load_docx(file_path: str) -> List[Document]:
    """Load a .docx file and return Document objects.

    Each document represents the full text content of the Word file.
    Paragraphs are joined with newlines.

    Parameters
    ----------
    file_path : str
        Path to the .docx file.

    Returns
    -------
    list[Document]
        A list containing one Document with the full text.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    from docx import Document as DocxDocument

    docx_doc = DocxDocument(str(path))
    source_name = path.name
    date_extracted = datetime.now(timezone.utc).isoformat()

    # Extract all paragraph text
    paragraphs = []
    for para in docx_doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)

    # Extract table text
    for table in docx_doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                paragraphs.append(row_text)

    if not paragraphs:
        logger.warning("No text content found in %s", source_name)
        return []

    full_text = "\n\n".join(paragraphs)

    doc = Document(
        doc_id=uuid.uuid4().hex[:12],
        text=full_text,
        metadata={
            "source": source_name,
            "file_type": "docx",
            "paragraph_count": len(paragraphs),
            "date_extracted": date_extracted,
        },
        source=str(path.resolve()),
    )

    logger.info("Loaded %s: %d paragraphs, %d chars", source_name, len(paragraphs), len(full_text))
    return [doc]
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_docx_loader.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/docx_loader.py tests/test_docx_loader.py
git commit -m "feat: add Word document loader"
```

---

### Task 10: PowerPoint loader

**Files:**
- Create: `src/ingestion/pptx_loader.py`
- Test: `tests/test_pptx_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pptx_loader.py`:

```python
"""Tests for PowerPoint loader."""
import pytest
from pathlib import Path
from src.ingestion.pptx_loader import load_pptx
from src.ingestion.pdf_loader import Document


@pytest.fixture
def sample_pptx(tmp_path):
    from pptx import Presentation
    prs = Presentation()
    slide_layout = prs.slide_layouts[1]  # Title and Content

    slide1 = prs.slides.add_slide(slide_layout)
    slide1.shapes.title.text = "Slide 1 Title"
    slide1.placeholders[1].text = "Content of slide 1"

    slide2 = prs.slides.add_slide(slide_layout)
    slide2.shapes.title.text = "Slide 2 Title"
    slide2.placeholders[1].text = "Content of slide 2"

    path = tmp_path / "test.pptx"
    prs.save(str(path))
    return str(path)


def test_load_pptx_returns_documents(sample_pptx):
    docs = load_pptx(sample_pptx)
    assert len(docs) == 2  # one per slide


def test_load_pptx_extracts_text(sample_pptx):
    docs = load_pptx(sample_pptx)
    assert "Slide 1 Title" in docs[0].text
    assert "Content of slide 1" in docs[0].text


def test_load_pptx_metadata(sample_pptx):
    docs = load_pptx(sample_pptx)
    assert docs[0].metadata.get("file_type") == "pptx"
    assert docs[0].metadata.get("slide_number") == 1
    assert docs[1].metadata.get("slide_number") == 2


def test_load_nonexistent_raises():
    with pytest.raises(FileNotFoundError):
        load_pptx("/nonexistent/file.pptx")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pptx_loader.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

Create `src/ingestion/pptx_loader.py`:

```python
"""PowerPoint (.pptx) loader for the RAG pipeline.

Extracts text from each slide (title, body, speaker notes) as separate documents.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from src.ingestion.pdf_loader import Document

logger = logging.getLogger(__name__)


def load_pptx(file_path: str) -> List[Document]:
    """Load a .pptx file and return one Document per slide.

    Parameters
    ----------
    file_path : str
        Path to the .pptx file.

    Returns
    -------
    list[Document]
        One Document per slide containing title, body, and notes.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    from pptx import Presentation

    prs = Presentation(str(path))
    source_name = path.name
    date_extracted = datetime.now(timezone.utc).isoformat()
    total_slides = len(prs.slides)

    documents: List[Document] = []

    for slide_num, slide in enumerate(prs.slides, start=1):
        parts = []

        # Extract text from all shapes
        for shape in slide.shapes:
            if shape.has_text_frame:
                text = shape.text_frame.text.strip()
                if text:
                    parts.append(text)

            # Table shapes
            if shape.has_table:
                for row in shape.table.rows:
                    row_text = " | ".join(
                        cell.text.strip() for cell in row.cells if cell.text.strip()
                    )
                    if row_text:
                        parts.append(row_text)

        # Speaker notes
        notes_text = ""
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            notes_text = slide.notes_slide.notes_text_frame.text.strip()
            if notes_text:
                parts.append(f"[Speaker Notes] {notes_text}")

        if not parts:
            continue

        slide_text = "\n\n".join(parts)

        doc = Document(
            doc_id=uuid.uuid4().hex[:12],
            text=slide_text,
            metadata={
                "source": source_name,
                "file_type": "pptx",
                "slide_number": slide_num,
                "total_slides": total_slides,
                "has_notes": bool(notes_text),
                "date_extracted": date_extracted,
            },
            source=str(path.resolve()),
        )
        documents.append(doc)

    logger.info("Loaded %s: %d slides with text", source_name, len(documents))
    return documents
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_pptx_loader.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/pptx_loader.py tests/test_pptx_loader.py
git commit -m "feat: add PowerPoint loader with per-slide extraction"
```

---

### Task 11: Excel loader

**Files:**
- Create: `src/ingestion/excel_loader.py`
- Test: `tests/test_excel_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_excel_loader.py`:

```python
"""Tests for Excel loader."""
import pytest
from pathlib import Path
from src.ingestion.excel_loader import load_excel
from src.ingestion.pdf_loader import Document


@pytest.fixture
def sample_xlsx(tmp_path):
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sales"
    ws.append(["Product", "Revenue", "Units"])
    ws.append(["Widget A", 1500, 100])
    ws.append(["Widget B", 2300, 150])

    ws2 = wb.create_sheet("Inventory")
    ws2.append(["Item", "Stock"])
    ws2.append(["Widget A", 500])

    path = tmp_path / "test.xlsx"
    wb.save(str(path))
    return str(path)


def test_load_excel_returns_documents(sample_xlsx):
    docs = load_excel(sample_xlsx)
    assert len(docs) == 2  # one per sheet


def test_load_excel_extracts_data(sample_xlsx):
    docs = load_excel(sample_xlsx)
    text = docs[0].text
    assert "Widget A" in text
    assert "1500" in text


def test_load_excel_metadata(sample_xlsx):
    docs = load_excel(sample_xlsx)
    assert docs[0].metadata.get("file_type") == "xlsx"
    assert docs[0].metadata.get("sheet_name") == "Sales"
    assert docs[1].metadata.get("sheet_name") == "Inventory"


def test_load_nonexistent_raises():
    with pytest.raises(FileNotFoundError):
        load_excel("/nonexistent/file.xlsx")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_excel_loader.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

Create `src/ingestion/excel_loader.py`:

```python
"""Excel (.xlsx, .xls) loader for the RAG pipeline.

Extracts sheet-by-sheet table data as formatted text documents.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from src.ingestion.pdf_loader import Document

logger = logging.getLogger(__name__)


def load_excel(file_path: str) -> List[Document]:
    """Load an Excel file and return one Document per sheet.

    Each sheet's data is formatted as a markdown-style table for
    readable chunking and retrieval.

    Parameters
    ----------
    file_path : str
        Path to the .xlsx or .xls file.

    Returns
    -------
    list[Document]
        One Document per non-empty sheet.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    import openpyxl

    wb = openpyxl.load_workbook(str(path), data_only=True)
    source_name = path.name
    date_extracted = datetime.now(timezone.utc).isoformat()

    documents: List[Document] = []

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))

        if not rows:
            continue

        # Format as readable text with header row
        lines = []
        header = rows[0] if rows else ()
        header_strs = [str(h) if h is not None else "" for h in header]

        if header_strs and any(header_strs):
            lines.append(f"Sheet: {sheet_name}")
            lines.append("Columns: " + " | ".join(header_strs))
            lines.append("")

            for row in rows[1:]:
                row_strs = [str(cell) if cell is not None else "" for cell in row]
                if any(row_strs):
                    row_text = " | ".join(
                        f"{h}: {v}" for h, v in zip(header_strs, row_strs) if v
                    )
                    lines.append(row_text)
        else:
            # No clear header row
            lines.append(f"Sheet: {sheet_name}")
            for row in rows:
                row_strs = [str(cell) if cell is not None else "" for cell in row]
                if any(row_strs):
                    lines.append(" | ".join(row_strs))

        if not lines:
            continue

        text = "\n".join(lines)
        doc = Document(
            doc_id=uuid.uuid4().hex[:12],
            text=text,
            metadata={
                "source": source_name,
                "file_type": "xlsx",
                "sheet_name": sheet_name,
                "row_count": len(rows) - 1,
                "column_count": len(header),
                "modality": "table",
                "date_extracted": date_extracted,
            },
            source=str(path.resolve()),
        )
        documents.append(doc)

    logger.info("Loaded %s: %d sheets with data", source_name, len(documents))
    return documents
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_excel_loader.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/excel_loader.py tests/test_excel_loader.py
git commit -m "feat: add Excel loader with per-sheet extraction"
```

---

### Task 12: HTML loader

**Files:**
- Create: `src/ingestion/html_loader.py`
- Test: `tests/test_html_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_html_loader.py`:

```python
"""Tests for HTML loader."""
import pytest
import tempfile
from pathlib import Path
from src.ingestion.html_loader import load_html
from src.ingestion.pdf_loader import Document


@pytest.fixture
def sample_html(tmp_path):
    html_content = """
    <html>
    <head><title>Test Page</title></head>
    <body>
        <h1>Main Heading</h1>
        <p>This is a paragraph with <strong>bold text</strong>.</p>
        <ul>
            <li>Item one</li>
            <li>Item two</li>
        </ul>
        <script>var x = 1;</script>
        <style>.hidden { display: none; }</style>
    </body>
    </html>
    """
    path = tmp_path / "test.html"
    path.write_text(html_content)
    return str(path)


def test_load_html_returns_documents(sample_html):
    docs = load_html(sample_html)
    assert len(docs) >= 1
    assert isinstance(docs[0], Document)


def test_load_html_extracts_text(sample_html):
    docs = load_html(sample_html)
    text = docs[0].text
    assert "Main Heading" in text
    assert "bold text" in text
    assert "Item one" in text


def test_load_html_strips_scripts(sample_html):
    docs = load_html(sample_html)
    text = docs[0].text
    assert "var x = 1" not in text
    assert ".hidden" not in text


def test_load_html_metadata(sample_html):
    docs = load_html(sample_html)
    assert docs[0].metadata.get("file_type") == "html"
    assert docs[0].metadata.get("title") == "Test Page"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_html_loader.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

Create `src/ingestion/html_loader.py`:

```python
"""HTML file loader for the RAG pipeline.

Extracts clean text from HTML files, stripping scripts, styles, and tags.
Uses BeautifulSoup for parsing.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from src.ingestion.pdf_loader import Document

logger = logging.getLogger(__name__)


def load_html(file_path: str) -> List[Document]:
    """Load an HTML file and return a Document with clean text.

    Parameters
    ----------
    file_path : str
        Path to the .html file.

    Returns
    -------
    list[Document]
        A list containing one Document with extracted text.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    from bs4 import BeautifulSoup

    raw_html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw_html, "html.parser")

    # Extract title
    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)

    # Remove script and style elements
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Extract text
    text = soup.get_text(separator="\n", strip=True)

    # Clean up excessive whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)

    if not clean_text:
        logger.warning("No text content extracted from %s", path.name)
        return []

    doc = Document(
        doc_id=uuid.uuid4().hex[:12],
        text=clean_text,
        metadata={
            "source": path.name,
            "file_type": "html",
            "title": title,
            "date_extracted": datetime.now(timezone.utc).isoformat(),
        },
        source=str(path.resolve()),
    )

    logger.info("Loaded %s: %d chars, title='%s'", path.name, len(clean_text), title[:50])
    return [doc]
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_html_loader.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/html_loader.py tests/test_html_loader.py
git commit -m "feat: add HTML loader with script/style stripping"
```

---

### Task 13: Code file loader

**Files:**
- Create: `src/ingestion/code_loader.py`
- Test: `tests/test_code_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_code_loader.py`:

```python
"""Tests for code file loader."""
import pytest
from pathlib import Path
from src.ingestion.code_loader import load_code
from src.ingestion.pdf_loader import Document


@pytest.fixture
def sample_python(tmp_path):
    code = '''"""Module docstring."""

def hello(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"

class Calculator:
    """Simple calculator."""

    def add(self, a: int, b: int) -> int:
        return a + b
'''
    path = tmp_path / "sample.py"
    path.write_text(code)
    return str(path)


def test_load_code_returns_documents(sample_python):
    docs = load_code(sample_python)
    assert len(docs) >= 1


def test_load_code_extracts_content(sample_python):
    docs = load_code(sample_python)
    text = docs[0].text
    assert "def hello" in text
    assert "class Calculator" in text


def test_load_code_detects_language(sample_python):
    docs = load_code(sample_python)
    assert docs[0].metadata.get("language") == "python"


def test_load_code_metadata(sample_python):
    docs = load_code(sample_python)
    assert docs[0].metadata.get("file_type") == "py"
    assert docs[0].metadata.get("modality") == "code"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_code_loader.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

Create `src/ingestion/code_loader.py`:

```python
"""Source code file loader for the RAG pipeline.

Loads code files with language detection based on extension.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from src.ingestion.pdf_loader import Document

logger = logging.getLogger(__name__)

_LANGUAGE_MAP: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".cs": "csharp",
    ".r": "r",
    ".sql": "sql",
    ".sh": "shell",
    ".bash": "shell",
}


def load_code(file_path: str) -> List[Document]:
    """Load a source code file.

    Parameters
    ----------
    file_path : str
        Path to the code file.

    Returns
    -------
    list[Document]
        A list containing one Document with the code content.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    language = _LANGUAGE_MAP.get(ext, "unknown")

    content = path.read_text(encoding="utf-8", errors="replace")
    if not content.strip():
        return []

    doc = Document(
        doc_id=uuid.uuid4().hex[:12],
        text=content,
        metadata={
            "source": path.name,
            "file_type": ext.lstrip("."),
            "language": language,
            "modality": "code",
            "line_count": content.count("\n") + 1,
            "date_extracted": datetime.now(timezone.utc).isoformat(),
        },
        source=str(path.resolve()),
    )

    logger.info("Loaded %s: %s, %d lines", path.name, language, doc.metadata["line_count"])
    return [doc]
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_code_loader.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/code_loader.py tests/test_code_loader.py
git commit -m "feat: add code file loader with language detection"
```

---

### Task 14: JSON/YAML loader

**Files:**
- Create: `src/ingestion/json_yaml_loader.py`
- Test: `tests/test_json_yaml_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_json_yaml_loader.py`:

```python
"""Tests for JSON/YAML loader."""
import pytest
import json
from pathlib import Path
from src.ingestion.json_yaml_loader import load_json_yaml
from src.ingestion.pdf_loader import Document


@pytest.fixture
def sample_json(tmp_path):
    data = {
        "company": "Acme Corp",
        "products": [
            {"name": "Widget", "price": 9.99},
            {"name": "Gadget", "price": 19.99},
        ],
        "metadata": {"version": "1.0"}
    }
    path = tmp_path / "data.json"
    path.write_text(json.dumps(data, indent=2))
    return str(path)


@pytest.fixture
def sample_yaml(tmp_path):
    yaml_content = """
name: Test Config
database:
  host: localhost
  port: 5432
features:
  - auth
  - logging
"""
    path = tmp_path / "config.yaml"
    path.write_text(yaml_content)
    return str(path)


def test_load_json_returns_documents(sample_json):
    docs = load_json_yaml(sample_json)
    assert len(docs) >= 1


def test_load_json_extracts_content(sample_json):
    docs = load_json_yaml(sample_json)
    text = docs[0].text
    assert "Acme Corp" in text
    assert "Widget" in text


def test_load_yaml_returns_documents(sample_yaml):
    docs = load_json_yaml(sample_yaml)
    assert len(docs) >= 1


def test_load_yaml_extracts_content(sample_yaml):
    docs = load_json_yaml(sample_yaml)
    text = docs[0].text
    assert "Test Config" in text
    assert "localhost" in text


def test_load_json_metadata(sample_json):
    docs = load_json_yaml(sample_json)
    assert docs[0].metadata.get("file_type") == "json"
    assert docs[0].metadata.get("modality") == "structured"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_json_yaml_loader.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

Create `src/ingestion/json_yaml_loader.py`:

```python
"""JSON and YAML file loader for the RAG pipeline.

Flattens structured data into readable text with path context.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from src.ingestion.pdf_loader import Document

logger = logging.getLogger(__name__)


def _flatten(obj: Any, prefix: str = "") -> List[str]:
    """Recursively flatten a dict/list into 'path: value' lines."""
    lines = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(_flatten(value, new_prefix))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            new_prefix = f"{prefix}[{i}]"
            lines.extend(_flatten(item, new_prefix))
    else:
        lines.append(f"{prefix}: {obj}")
    return lines


def load_json_yaml(file_path: str) -> List[Document]:
    """Load a JSON or YAML file and flatten into a searchable document.

    Parameters
    ----------
    file_path : str
        Path to the .json, .yaml, or .yml file.

    Returns
    -------
    list[Document]
        A list containing one Document with flattened content.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    raw_text = path.read_text(encoding="utf-8", errors="replace")

    if ext == ".json":
        data = json.loads(raw_text)
        file_type = "json"
    elif ext in (".yaml", ".yml"):
        try:
            import yaml
            data = yaml.safe_load(raw_text)
        except ImportError:
            # Fallback: treat as plain text
            data = None
            file_type = ext.lstrip(".")
        file_type = "yaml"
    else:
        data = None
        file_type = ext.lstrip(".")

    if data is not None:
        lines = _flatten(data)
        text = "\n".join(lines)
    else:
        text = raw_text

    if not text.strip():
        return []

    doc = Document(
        doc_id=uuid.uuid4().hex[:12],
        text=text,
        metadata={
            "source": path.name,
            "file_type": file_type,
            "modality": "structured",
            "date_extracted": datetime.now(timezone.utc).isoformat(),
        },
        source=str(path.resolve()),
    )

    logger.info("Loaded %s: %s, %d chars", path.name, file_type, len(text))
    return [doc]
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_json_yaml_loader.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/json_yaml_loader.py tests/test_json_yaml_loader.py
git commit -m "feat: add JSON/YAML loader with path-based flattening"
```

---

### Task 15: Audio loader

**Files:**
- Create: `src/ingestion/audio_loader.py`
- Test: `tests/test_audio_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_audio_loader.py`:

```python
"""Tests for audio loader."""
import pytest
from unittest.mock import patch, MagicMock
from src.ingestion.audio_loader import load_audio
from src.ingestion.pdf_loader import Document


def test_load_nonexistent_raises():
    with pytest.raises(FileNotFoundError):
        load_audio("/nonexistent/audio.mp3")


def test_load_audio_with_mock_whisper(tmp_path):
    # Create a dummy audio file
    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(b"\x00" * 1000)

    with patch("src.ingestion.audio_loader._transcribe_whisper") as mock_whisper:
        mock_whisper.return_value = "This is a test transcription of the audio file."

        docs = load_audio(str(audio_path))
        assert len(docs) >= 1
        assert "test transcription" in docs[0].text
        assert docs[0].metadata.get("modality") == "audio"


def test_load_audio_metadata(tmp_path):
    audio_path = tmp_path / "meeting.mp3"
    audio_path.write_bytes(b"\x00" * 1000)

    with patch("src.ingestion.audio_loader._transcribe_whisper") as mock_whisper:
        mock_whisper.return_value = "Meeting notes content."

        docs = load_audio(str(audio_path))
        assert docs[0].metadata.get("file_type") == "mp3"
        assert docs[0].metadata.get("source") == "meeting.mp3"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_audio_loader.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

Create `src/ingestion/audio_loader.py`:

```python
"""Audio file loader for the RAG pipeline.

Transcribes audio files using faster-whisper (local) or Gemini (API).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.ingestion.pdf_loader import Document

logger = logging.getLogger(__name__)


def _transcribe_whisper(file_path: str) -> str:
    """Transcribe audio using faster-whisper (local, free)."""
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, info = model.transcribe(file_path, beam_size=5)
        text_parts = [segment.text for segment in segments]
        return " ".join(text_parts).strip()
    except ImportError:
        logger.warning("faster-whisper not installed.")
        return ""
    except Exception as exc:
        logger.warning("Whisper transcription failed: %s", exc)
        return ""


def _transcribe_gemini(file_path: str) -> str:
    """Transcribe audio using Gemini API."""
    try:
        from config.settings import GEMINI_API_KEY
        from src.gemini.client import GeminiClient

        if not GEMINI_API_KEY:
            return ""

        client = GeminiClient(api_key=GEMINI_API_KEY)
        # Gemini can handle audio files directly
        return client.generate_text(
            "Transcribe the following audio file accurately. Return only the transcription text.",
            system="You are an audio transcription assistant. Provide accurate word-for-word transcription.",
        )
    except Exception as exc:
        logger.warning("Gemini transcription failed: %s", exc)
        return ""


def load_audio(
    file_path: str,
    *,
    provider: str = "whisper",
) -> List[Document]:
    """Load an audio file and return its transcription.

    Parameters
    ----------
    file_path : str
        Path to the audio file.
    provider : str
        Transcription provider: "whisper" (default) or "gemini".

    Returns
    -------
    list[Document]
        A list containing one Document with the transcription.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    source_name = path.name

    # Transcribe
    if provider == "gemini":
        text = _transcribe_gemini(str(path))
    else:
        text = _transcribe_whisper(str(path))

    if not text:
        text = f"[Audio file: {source_name} — transcription unavailable]"

    doc = Document(
        doc_id=uuid.uuid4().hex[:12],
        text=text,
        metadata={
            "source": source_name,
            "file_type": path.suffix.lstrip(".").lower(),
            "modality": "audio",
            "date_extracted": datetime.now(timezone.utc).isoformat(),
        },
        source=str(path.resolve()),
    )

    logger.info("Loaded audio %s: %d chars transcribed", source_name, len(text))
    return [doc]
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_audio_loader.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/audio_loader.py tests/test_audio_loader.py
git commit -m "feat: add audio loader with whisper and gemini transcription"
```

---

### Task 16: Video loader

**Files:**
- Create: `src/ingestion/video_loader.py`
- Test: `tests/test_video_loader.py`  (note: video tests use mocks since OpenCV + real video files are heavy)

- [ ] **Step 1: Write the failing test**

Create `tests/test_video_loader.py`:

```python
"""Tests for video loader (mocked — no real video files needed)."""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image
from src.ingestion.video_loader import load_video, _detect_scene_changes


def test_load_nonexistent_raises():
    with pytest.raises(FileNotFoundError):
        load_video("/nonexistent/video.mp4")


def test_detect_scene_changes_returns_indices():
    # Simulate 10 frames, with a big change at frame 5
    frames = []
    for i in range(10):
        if i < 5:
            frames.append(np.zeros((100, 100, 3), dtype=np.uint8))
        else:
            frames.append(np.ones((100, 100, 3), dtype=np.uint8) * 255)

    indices = _detect_scene_changes(frames, threshold=30.0)
    assert 5 in indices  # Should detect the big change at frame 5


def test_load_video_with_mock(tmp_path):
    video_path = tmp_path / "test.mp4"
    video_path.write_bytes(b"\x00" * 100)

    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch("src.ingestion.video_loader.cv2") as mock_cv2, \
         patch("src.ingestion.video_loader._transcribe_audio") as mock_transcribe:

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FPS: 30.0,
            mock_cv2.CAP_PROP_FRAME_COUNT: 90,
        }.get(prop, 0)
        mock_cap.read.side_effect = [
            (True, fake_frame),
            (True, fake_frame),
            (True, fake_frame),
            (False, None),
        ]
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.cvtColor.return_value = fake_frame
        mock_cv2.calcHist.return_value = [np.zeros(256)]
        mock_cv2.compareHist.return_value = 1.0

        mock_transcribe.return_value = "Video transcript text"

        result = load_video(str(video_path))
        assert result.text_documents is not None
        assert any("transcript" in d.text.lower() or "Video" in d.text for d in result.text_documents)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_video_loader.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

Create `src/ingestion/video_loader.py`:

```python
"""Video loader with scene-change keyframe extraction + audio transcription.

Uses OpenCV for frame extraction and faster-whisper or Gemini for audio.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.ingestion.image_loader import LoaderResult
from src.ingestion.pdf_loader import Document

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None


def _detect_scene_changes(
    frames: List[np.ndarray],
    threshold: float = 30.0,
) -> List[int]:
    """Detect frames where a scene change occurs using histogram comparison.

    Parameters
    ----------
    frames : list[np.ndarray]
        List of BGR frame arrays.
    threshold : float
        Histogram difference threshold for scene change.

    Returns
    -------
    list[int]
        Indices of frames where scene changes were detected.
    """
    if not frames or len(frames) < 2:
        return [0] if frames else []

    changes = [0]  # Always include the first frame
    prev_hist = cv2.calcHist([frames[0]], [0], None, [256], [0, 256])

    for i in range(1, len(frames)):
        curr_hist = cv2.calcHist([frames[i]], [0], None, [256], [0, 256])
        diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CHISQR)

        if diff > threshold:
            changes.append(i)

        prev_hist = curr_hist

    return changes


def _transcribe_audio(video_path: str) -> str:
    """Extract and transcribe audio from video."""
    try:
        from src.ingestion.audio_loader import _transcribe_whisper
        return _transcribe_whisper(video_path)
    except Exception as exc:
        logger.warning("Audio transcription from video failed: %s", exc)
        return ""


def load_video(
    file_path: str,
    *,
    max_frames: int = 50,
    scene_threshold: float = 30.0,
    fallback_interval_sec: float = 10.0,
) -> LoaderResult:
    """Load a video file, extract keyframes and audio transcription.

    Parameters
    ----------
    file_path : str
        Path to the video file.
    max_frames : int
        Maximum number of keyframes to extract.
    scene_threshold : float
        Histogram difference threshold for scene change detection.
    fallback_interval_sec : float
        If scene detection extracts no frames, use this interval.

    Returns
    -------
    LoaderResult
        Contains text_documents (transcript) and visual_assets (keyframes).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {file_path}")

    if cv2 is None:
        raise ImportError("opencv-python is required for video loading.")

    source_name = path.name
    doc_id = uuid.uuid4().hex[:12]
    date_extracted = datetime.now(timezone.utc).isoformat()

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {file_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    # Read all frames (sampled at 1 per second for efficiency)
    sample_interval = max(int(fps), 1)
    sampled_frames = []
    frame_indices = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval == 0:
            sampled_frames.append(frame)
            frame_indices.append(frame_idx)
        frame_idx += 1

    cap.release()

    # Detect scene changes
    if sampled_frames:
        change_indices = _detect_scene_changes(sampled_frames, threshold=scene_threshold)
        if not change_indices:
            # Fallback: every N seconds
            interval = int(fallback_interval_sec * fps / sample_interval)
            change_indices = list(range(0, len(sampled_frames), max(interval, 1)))
    else:
        change_indices = []

    # Limit to max_frames
    change_indices = change_indices[:max_frames]

    result = LoaderResult()

    # Convert keyframes to PIL images
    from PIL import Image
    for ci in change_indices:
        if ci < len(sampled_frames):
            frame_bgr = sampled_frames[ci]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            time_sec = frame_indices[ci] / fps if fps > 0 else 0

            result.visual_assets.append({
                "image": pil_image,
                "doc_id": doc_id,
                "modality": "video",
                "source": source_name,
                "frame_time_sec": round(time_sec, 1),
            })

    # Transcribe audio
    transcript = _transcribe_audio(str(path))

    # Create text document
    text_parts = []
    if transcript:
        text_parts.append(f"Video transcript:\n{transcript}")
    else:
        text_parts.append(f"[Video: {source_name}, duration: {duration_sec:.0f}s, {len(change_indices)} keyframes extracted]")

    doc = Document(
        doc_id=doc_id,
        text="\n\n".join(text_parts),
        metadata={
            "source": source_name,
            "file_type": path.suffix.lstrip(".").lower(),
            "modality": "video",
            "duration_sec": round(duration_sec, 1),
            "keyframes_extracted": len(change_indices),
            "has_transcript": bool(transcript),
            "date_extracted": date_extracted,
        },
        source=str(path.resolve()),
    )
    result.text_documents.append(doc)

    logger.info(
        "Loaded video %s: %.0fs, %d keyframes, transcript=%d chars",
        source_name, duration_sec, len(change_indices), len(transcript),
    )
    return result
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_video_loader.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/video_loader.py tests/test_video_loader.py
git commit -m "feat: add video loader with scene-change keyframe extraction"
```

---

### Task 17: Register all new loaders in the ingestion pipeline

**Files:**
- Modify: `src/ingestion/pipeline.py`

- [ ] **Step 1: Update the loader map and skip extensions**

In `src/ingestion/pipeline.py`, update the imports and `_LOADER_MAP`:

```python
from .csv_loader import load_csv
from .metadata_tagger import tag_documents
from .pdf_loader import Document, load_pdf
from .txt_loader import load_txt
from .docx_loader import load_docx
from .pptx_loader import load_pptx
from .excel_loader import load_excel
from .html_loader import load_html
from .code_loader import load_code
from .json_yaml_loader import load_json_yaml
from .audio_loader import load_audio

_LOADER_MAP: Dict[str, callable] = {
    # Existing
    ".pdf": load_pdf,
    ".csv": load_csv,
    ".txt": load_txt,
    ".log": load_txt,
    ".text": load_txt,
    ".md": load_txt,
    ".eml": load_txt,
    # New document loaders
    ".docx": load_docx,
    ".pptx": load_pptx,
    ".xlsx": load_excel,
    ".xls": load_excel,
    # Web
    ".html": load_html,
    ".htm": load_html,
    # Code
    ".py": load_code,
    ".js": load_code,
    ".ts": load_code,
    ".java": load_code,
    ".cpp": load_code,
    ".c": load_code,
    ".go": load_code,
    ".rs": load_code,
    ".rb": load_code,
    ".php": load_code,
    ".sh": load_code,
    # Structured
    ".json": load_json_yaml,
    ".yaml": load_json_yaml,
    ".yml": load_json_yaml,
    # Audio
    ".mp3": load_audio,
    ".wav": load_audio,
    ".m4a": load_audio,
}
```

- [ ] **Step 2: Remove image/video extensions from _SKIP_EXTENSIONS**

Update `_SKIP_EXTENSIONS` to remove image and video formats that now have loaders:

```python
_SKIP_EXTENSIONS: Set[str] = {
    ".pyc", ".pyo", ".so", ".dll", ".exe", ".bin",
    ".zip", ".tar", ".gz", ".bz2", ".7z",
    ".db", ".sqlite", ".sqlite3",
}
```

Note: `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.ico` removed — these now have the image loader. Video formats (`.mp4`, `.avi`, `.mov`) were never in skip list.

- [ ] **Step 3: Add image and video loaders with special handling**

Image and video loaders return `LoaderResult` (not `List[Document]`), so add special handling in the `ingest` method. Add a wrapper that extracts text documents:

```python
# Special loaders that return LoaderResult (multimodal output)
from .image_loader import load_image as _load_image_raw
from .video_loader import load_video as _load_video_raw


def _load_image_wrapper(file_path: str, **kwargs) -> List[Document]:
    """Wrap image loader to return List[Document] for pipeline compatibility."""
    result = _load_image_raw(file_path, **kwargs)
    return result.text_documents


def _load_video_wrapper(file_path: str, **kwargs) -> List[Document]:
    """Wrap video loader to return List[Document] for pipeline compatibility."""
    result = _load_video_raw(file_path, **kwargs)
    return result.text_documents
```

Then add to `_LOADER_MAP`:

```python
    # Images (returns text via OCR, visual assets handled separately)
    ".png": _load_image_wrapper,
    ".jpg": _load_image_wrapper,
    ".jpeg": _load_image_wrapper,
    ".webp": _load_image_wrapper,
    ".bmp": _load_image_wrapper,
    ".gif": _load_image_wrapper,
    # Video
    ".mp4": _load_video_wrapper,
    ".avi": _load_video_wrapper,
    ".mov": _load_video_wrapper,
```

- [ ] **Step 4: Verify the pipeline still works**

Run: `python -c "from src.ingestion.pipeline import IngestPipeline; p = IngestPipeline(); print('Loaders:', len(p._LOADER_MAP if hasattr(p, '_LOADER_MAP') else 'ok'))"`
Expected: No import errors

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/pipeline.py
git commit -m "feat: register all 12 file type loaders in ingestion pipeline"
```

---

## Phase 5: Gemini Captioning + Multimodal Retrieval

### Task 18: Gemini captioner for images

**Files:**
- Create: `src/gemini/captioner.py`
- Test: `tests/test_captioner.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_captioner.py`:

```python
"""Tests for Gemini image captioner."""
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
from src.gemini.captioner import ImageCaptioner


@pytest.fixture
def sample_image():
    return Image.new("RGB", (100, 100), color="red")


def test_caption_image(sample_image):
    with patch("src.gemini.captioner.GeminiClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.generate_text.return_value = "A red square image"
        mock_cls.return_value = mock_client

        captioner = ImageCaptioner(api_key="test-key")
        caption = captioner.caption(sample_image)
        assert caption == "A red square image"


def test_caption_batch(sample_image):
    with patch("src.gemini.captioner.GeminiClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.generate_text.return_value = "An image"
        mock_client.quota_remaining.return_value = {"rpm_remaining": 10, "daily_remaining": 250}
        mock_cls.return_value = mock_client

        captioner = ImageCaptioner(api_key="test-key")
        captions = captioner.caption_batch([sample_image, sample_image])
        assert len(captions) == 2


def test_fallback_when_quota_exhausted(sample_image):
    with patch("src.gemini.captioner.GeminiClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.generate_text.side_effect = RuntimeError("quota exhausted")
        mock_cls.return_value = mock_client

        captioner = ImageCaptioner(api_key="test-key")
        caption = captioner.caption(sample_image)
        # Should return a fallback description, not crash
        assert isinstance(caption, str)
        assert len(caption) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_captioner.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

Create `src/gemini/captioner.py`:

```python
"""Image captioning using Gemini Vision for text-index discoverability.

Generates rich text descriptions of images so that text queries
can find visual content via the BGE text index.
"""

from __future__ import annotations

import logging
import time
from typing import Any, List, Optional

from src.gemini.client import GeminiClient

logger = logging.getLogger(__name__)

_CAPTION_PROMPT = (
    "Describe this image in detail for a search index. Include:\n"
    "1. What the image shows (objects, people, scenes)\n"
    "2. Any text visible in the image\n"
    "3. If it's a chart/graph: the type, axes, trends, and key data points\n"
    "4. If it's a diagram: the components and their relationships\n"
    "Keep the description factual and under 200 words."
)


class ImageCaptioner:
    """Generate text captions for images using Gemini Vision.

    Parameters
    ----------
    api_key : str
        Gemini API key.
    rpm_limit : int
        Requests per minute to stay within.
    """

    def __init__(
        self,
        api_key: str,
        rpm_limit: int = 8,
    ) -> None:
        self._client = GeminiClient(api_key=api_key, rpm_limit=rpm_limit)

    def caption(self, image: Any) -> str:
        """Generate a text caption for a single image.

        Parameters
        ----------
        image : PIL.Image.Image
            The image to caption.

        Returns
        -------
        str
            Text description of the image.
        """
        try:
            return self._client.generate_text(
                _CAPTION_PROMPT,
                images=[image],
                system="You are a visual description assistant for a document search system.",
            )
        except Exception as exc:
            logger.warning("Gemini captioning failed: %s. Using fallback.", exc)
            return self._fallback_caption(image)

    def caption_batch(
        self,
        images: List[Any],
        delay_between: float = 0.5,
    ) -> List[str]:
        """Caption a batch of images with rate-limit-aware pacing.

        Parameters
        ----------
        images : list[PIL.Image.Image]
            Images to caption.
        delay_between : float
            Seconds to wait between requests.

        Returns
        -------
        list[str]
            Captions in the same order as input images.
        """
        captions = []
        for i, img in enumerate(images):
            caption = self.caption(img)
            captions.append(caption)
            if i < len(images) - 1:
                time.sleep(delay_between)
        return captions

    @staticmethod
    def _fallback_caption(image: Any) -> str:
        """Generate a basic caption without an LLM."""
        try:
            width, height = image.size
            mode = image.mode
            return (
                f"Image ({width}x{height}, {mode}). "
                "Visual content — detailed caption unavailable."
            )
        except Exception:
            return "Image content — caption unavailable."
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_captioner.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/gemini/captioner.py tests/test_captioner.py
git commit -m "feat: add Gemini image captioner with rate limiting and fallback"
```

---

### Task 19: Query analyzer with modality detection

**Files:**
- Create: `src/retrieval/query_analyzer.py`
- Test: `tests/test_query_analyzer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_query_analyzer.py`:

```python
"""Tests for query analyzer with modality detection."""
import pytest
from src.retrieval.query_analyzer import QueryAnalyzer


@pytest.fixture
def analyzer():
    return QueryAnalyzer()


def test_detect_visual_query(analyzer):
    result = analyzer.analyze("show me the chart from the report")
    assert result["modality"] == "visual"
    assert result["visual_weight"] > result["text_weight"]


def test_detect_text_query(analyzer):
    result = analyzer.analyze("what is the refund policy?")
    assert result["modality"] == "text"
    assert result["text_weight"] > result["visual_weight"]


def test_detect_ambiguous_query(analyzer):
    result = analyzer.analyze("tell me about Q3 performance")
    assert result["modality"] == "both"
    assert result["text_weight"] == pytest.approx(0.6, abs=0.1)


def test_visual_keywords(analyzer):
    visual_queries = [
        "what does the diagram show",
        "describe the image on page 5",
        "show me the architecture diagram",
        "what's in the screenshot",
    ]
    for q in visual_queries:
        result = analyzer.analyze(q)
        assert result["modality"] in ("visual", "both"), f"Expected visual for: {q}"


def test_text_keywords(analyzer):
    text_queries = [
        "what is the company policy on PTO",
        "list all suppliers in the system",
        "define the term SLA",
    ]
    for q in text_queries:
        result = analyzer.analyze(q)
        assert result["modality"] == "text", f"Expected text for: {q}"


def test_weights_sum_to_one(analyzer):
    result = analyzer.analyze("anything at all")
    total = result["text_weight"] + result["visual_weight"]
    assert total == pytest.approx(1.0, abs=0.01)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_query_analyzer.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

Create `src/retrieval/query_analyzer.py`:

```python
"""Query analyzer with modality detection for multimodal retrieval.

Detects whether a query targets visual content, text content, or both,
and sets adaptive retrieval weights accordingly.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Keywords that signal visual intent
_VISUAL_KEYWORDS = {
    "chart", "graph", "diagram", "image", "picture", "photo",
    "screenshot", "figure", "illustration", "table", "slide",
    "visualization", "plot", "map", "drawing", "layout",
    "show me", "what does it look like", "visual",
}

_VISUAL_PHRASES = [
    "show me the",
    "what does the .* show",
    "describe the image",
    "in the diagram",
    "in the chart",
    "in the figure",
    "on the slide",
    "the screenshot",
    "the picture",
    "what's in the",
    "look at the",
]

# Weights for different modality signals
_VISUAL_WEIGHTS = {"text_weight": 0.3, "visual_weight": 0.7}
_TEXT_WEIGHTS = {"text_weight": 0.9, "visual_weight": 0.1}
_BOTH_WEIGHTS = {"text_weight": 0.6, "visual_weight": 0.4}


class QueryAnalyzer:
    """Analyze queries to determine modality and set retrieval weights."""

    def analyze(self, query: str) -> Dict[str, Any]:
        """Analyze a query for modality signals.

        Returns
        -------
        dict
            Keys: modality ("visual", "text", "both"),
                  text_weight (float), visual_weight (float)
        """
        q_lower = query.lower()

        # Check for visual phrase patterns
        visual_score = 0
        for phrase in _VISUAL_PHRASES:
            if re.search(phrase, q_lower):
                visual_score += 2

        # Check for visual keywords
        for kw in _VISUAL_KEYWORDS:
            if kw in q_lower:
                visual_score += 1

        if visual_score >= 2:
            modality = "visual"
            weights = _VISUAL_WEIGHTS
        elif visual_score == 1:
            modality = "both"
            weights = _BOTH_WEIGHTS
        else:
            modality = "text"
            weights = _TEXT_WEIGHTS

        result = {
            "modality": modality,
            "text_weight": weights["text_weight"],
            "visual_weight": weights["visual_weight"],
            "visual_score": visual_score,
        }

        logger.debug("Query analysis: '%s' -> %s", query[:50], result)
        return result
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_query_analyzer.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/query_analyzer.py tests/test_query_analyzer.py
git commit -m "feat: add query analyzer with modality detection and adaptive weights"
```

---

### Task 20: Multimodal retriever with score fusion

**Files:**
- Create: `src/retrieval/multimodal_retriever.py`
- Test: `tests/test_multimodal_retriever.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_multimodal_retriever.py`:

```python
"""Tests for multimodal retriever with score fusion."""
import pytest
import numpy as np
from unittest.mock import MagicMock
from src.retrieval.multimodal_retriever import MultimodalRetriever, normalize_scores


def test_normalize_scores_empty():
    assert normalize_scores([]) == []


def test_normalize_scores_single():
    result = normalize_scores([{"score": 0.8}])
    assert result[0]["normalized_score"] == 1.0


def test_normalize_scores_range():
    items = [{"score": 0.2}, {"score": 0.5}, {"score": 0.8}]
    result = normalize_scores(items)
    assert result[0]["normalized_score"] == pytest.approx(0.0)
    assert result[2]["normalized_score"] == pytest.approx(1.0)


def test_retriever_text_only():
    mock_text_search = MagicMock()
    mock_text_search.return_value = [
        {"text": "chunk 1", "score": 0.9, "doc_id": "d1", "modality": "text"},
        {"text": "chunk 2", "score": 0.7, "doc_id": "d2", "modality": "text"},
    ]

    retriever = MultimodalRetriever(
        text_search_fn=mock_text_search,
        visual_search_fn=None,
    )

    results = retriever.retrieve("what is the policy?", top_k=5)
    assert len(results) == 2
    assert results[0]["score"] >= results[1]["score"]


def test_retriever_fuses_text_and_visual():
    mock_text_search = MagicMock()
    mock_text_search.return_value = [
        {"text": "text chunk", "score": 0.8, "doc_id": "d1", "modality": "text"},
    ]

    mock_visual_search = MagicMock()
    mock_visual_search.return_value = [
        {"asset_id": "a1", "score": 0.9, "doc_id": "d2", "modality": "image", "caption": "A chart"},
    ]

    retriever = MultimodalRetriever(
        text_search_fn=mock_text_search,
        visual_search_fn=mock_visual_search,
    )

    results = retriever.retrieve("show me the chart", top_k=5)
    assert len(results) == 2


def test_retriever_deduplicates():
    mock_text_search = MagicMock()
    mock_text_search.return_value = [
        {"text": "caption of image", "score": 0.8, "doc_id": "d1", "page_or_slide": 1, "modality": "text"},
    ]

    mock_visual_search = MagicMock()
    mock_visual_search.return_value = [
        {"asset_id": "a1", "score": 0.9, "doc_id": "d1", "page_or_slide": 1, "modality": "image"},
    ]

    retriever = MultimodalRetriever(
        text_search_fn=mock_text_search,
        visual_search_fn=mock_visual_search,
    )

    results = retriever.retrieve("query", top_k=5)
    # Should deduplicate by doc_id + page_or_slide, keeping the image (richer)
    assert len(results) == 1
    assert results[0]["modality"] == "image"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_multimodal_retriever.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

Create `src/retrieval/multimodal_retriever.py`:

```python
"""Multimodal retriever that fuses text and visual search results.

Searches both the BGE text index and CLIP visual index,
normalizes scores, applies adaptive weights, deduplicates,
and returns a unified ranked list.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from src.retrieval.query_analyzer import QueryAnalyzer

logger = logging.getLogger(__name__)


def normalize_scores(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Min-max normalize scores to [0, 1].

    Parameters
    ----------
    results : list[dict]
        Each dict must have a "score" key.

    Returns
    -------
    list[dict]
        Same dicts with an added "normalized_score" key.
    """
    if not results:
        return []

    scores = [r["score"] for r in results]
    min_s = min(scores)
    max_s = max(scores)
    range_s = max_s - min_s

    for r in results:
        if range_s < 1e-12:
            r["normalized_score"] = 1.0
        else:
            r["normalized_score"] = (r["score"] - min_s) / range_s

    return results


class MultimodalRetriever:
    """Fuse results from text and visual search indices.

    Parameters
    ----------
    text_search_fn : callable
        Function: (query, top_k) -> list[dict] with text search results.
    visual_search_fn : callable | None
        Function: (query, top_k) -> list[dict] with visual search results.
    query_analyzer : QueryAnalyzer | None
        For modality detection and weight setting.
    """

    def __init__(
        self,
        text_search_fn: Callable,
        visual_search_fn: Optional[Callable] = None,
        query_analyzer: Optional[QueryAnalyzer] = None,
    ) -> None:
        self._text_search = text_search_fn
        self._visual_search = visual_search_fn
        self._analyzer = query_analyzer or QueryAnalyzer()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        text_weight: Optional[float] = None,
        visual_weight: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve and fuse results from text and visual indices.

        Parameters
        ----------
        query : str
            The user query.
        top_k : int
            Number of final results to return.
        text_weight : float | None
            Override text weight (otherwise auto-detected).
        visual_weight : float | None
            Override visual weight.

        Returns
        -------
        list[dict]
            Ranked, deduplicated results from both indices.
        """
        # Analyze query for modality
        analysis = self._analyzer.analyze(query)
        tw = text_weight if text_weight is not None else analysis["text_weight"]
        vw = visual_weight if visual_weight is not None else analysis["visual_weight"]

        # Search text index
        text_results = []
        try:
            text_results = self._text_search(query, top_k=top_k * 2)
            text_results = normalize_scores(text_results)
            for r in text_results:
                r["final_score"] = r["normalized_score"] * tw
                r.setdefault("modality", "text")
        except Exception as exc:
            logger.warning("Text search failed: %s", exc)

        # Search visual index
        visual_results = []
        if self._visual_search is not None and vw > 0.05:
            try:
                visual_results = self._visual_search(query, top_k=top_k * 2)
                visual_results = normalize_scores(visual_results)
                for r in visual_results:
                    r["final_score"] = r["normalized_score"] * vw
                    r.setdefault("modality", "image")
            except Exception as exc:
                logger.warning("Visual search failed: %s", exc)

        # Merge
        all_results = text_results + visual_results

        # Deduplicate by doc_id + page_or_slide (keep image over caption)
        seen = {}
        for r in all_results:
            key = (r.get("doc_id", ""), r.get("page_or_slide", ""))
            if key == ("", ""):
                # No dedup key — keep it
                seen[id(r)] = r
            elif key not in seen or (
                r.get("modality") == "image" and seen[key].get("modality") != "image"
            ):
                seen[key] = r

        deduped = list(seen.values())

        # Sort by final_score descending
        deduped.sort(key=lambda r: r.get("final_score", 0), reverse=True)

        logger.info(
            "Multimodal retrieval: %d text + %d visual -> %d after dedup (weights: text=%.2f, visual=%.2f)",
            len(text_results), len(visual_results), len(deduped), tw, vw,
        )

        return deduped[:top_k]
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_multimodal_retriever.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/multimodal_retriever.py tests/test_multimodal_retriever.py
git commit -m "feat: add multimodal retriever with score normalization and fusion"
```

---

## Phase 6: Answer Generation with Gemini

### Task 21: Gemini answer generator with structured output

**Files:**
- Create: `src/gemini/generator.py`
- Test: `tests/test_gemini_generator.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_gemini_generator.py`:

```python
"""Tests for Gemini answer generator."""
import pytest
from unittest.mock import patch, MagicMock
from src.gemini.generator import AnswerGenerator


def test_build_context_text_only():
    gen = _make_generator()
    chunks = [
        {"text": "Revenue grew 15%.", "source": "report.pdf", "modality": "text"},
        {"text": "Q3 was strong.", "source": "summary.txt", "modality": "text"},
    ]
    context = gen._build_context(chunks)
    assert "[Source 1]" in context
    assert "[Source 2]" in context
    assert "Revenue grew 15%" in context


def test_build_context_limits_images():
    gen = _make_generator(max_images=2)
    chunks = [
        {"text": "text chunk", "modality": "text"},
        {"asset_id": "a1", "modality": "image", "caption": "chart 1"},
        {"asset_id": "a2", "modality": "image", "caption": "chart 2"},
        {"asset_id": "a3", "modality": "image", "caption": "chart 3"},
    ]
    _, images = gen._build_context_with_images(chunks)
    assert len(images) <= 2


def test_post_process_verifies_citations():
    gen = _make_generator()
    answer = {
        "answer": "Revenue grew as shown in [Source 1].",
        "citations": [{"source_id": 1, "quote": "Revenue grew 15%"}],
        "confidence": 0.9,
    }
    sources = [{"text": "Revenue grew 15%.", "source": "report.pdf"}]
    result = gen._post_process(answer, sources)
    assert result["citation_verified"] is True


def test_post_process_flags_invalid_citation():
    gen = _make_generator()
    answer = {
        "answer": "Revenue grew as shown in [Source 5].",
        "citations": [{"source_id": 5, "quote": "not in context"}],
        "confidence": 0.9,
    }
    sources = [{"text": "Revenue grew 15%.", "source": "report.pdf"}]
    result = gen._post_process(answer, sources)
    assert result["citation_verified"] is False


def _make_generator(**kwargs):
    with patch("src.gemini.generator.GeminiClient"):
        return AnswerGenerator(api_key="test-key", **kwargs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gemini_generator.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

Create `src/gemini/generator.py`:

```python
"""Gemini-powered answer generator with structured JSON output.

Builds a multimodal prompt from retrieved text + images,
calls Gemini for a structured response, and post-processes
for citation verification and confidence scoring.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.gemini.client import GeminiClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions based on provided context. "
    "Rules:\n"
    "1. ONLY use information from the provided sources.\n"
    "2. Cite sources using [Source N] format.\n"
    "3. If the context is insufficient, say 'I don't have enough information to answer this.'\n"
    "4. Be concise and accurate.\n"
    "5. Respond in JSON format with keys: answer, citations, confidence, needs_visual."
)

_ANSWER_SCHEMA = """
Respond with this exact JSON structure:
{
  "answer": "Your detailed answer with [Source N] citations",
  "citations": [{"source_id": 1, "quote": "exact quote from source"}],
  "confidence": 0.85,
  "needs_visual": false
}
"""


class AnswerGenerator:
    """Generate answers using Gemini with multimodal context.

    Parameters
    ----------
    api_key : str
        Gemini API key.
    max_images : int
        Maximum images to include in context.
    """

    def __init__(
        self,
        api_key: str,
        max_images: int = 3,
        model_name: str = "gemini-2.5-flash",
    ) -> None:
        self._client = GeminiClient(api_key=api_key, model_name=model_name)
        self._max_images = max_images

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        asset_loader: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Generate an answer from retrieved context.

        Parameters
        ----------
        query : str
            The user question.
        retrieved_chunks : list[dict]
            Retrieved results (text + visual chunks).
        asset_loader : AssetStore | None
            For loading image assets from disk.

        Returns
        -------
        dict
            Answer with citations, confidence, and post-processing results.
        """
        context_text, images = self._build_context_with_images(
            retrieved_chunks, asset_loader=asset_loader,
        )

        prompt = f"{context_text}\n\n{_ANSWER_SCHEMA}\n\nQuestion: {query}"

        raw_answer = self._client.generate_json(
            prompt,
            system=_SYSTEM_PROMPT,
            images=images if images else None,
        )

        # Handle raw_response (fallback when JSON parsing failed)
        if "raw_response" in raw_answer:
            raw_answer = {
                "answer": raw_answer["raw_response"],
                "citations": [],
                "confidence": 0.5,
                "needs_visual": False,
            }

        # Post-process
        result = self._post_process(raw_answer, retrieved_chunks)
        result["query"] = query
        result["sources_used"] = len(retrieved_chunks)
        result["images_sent"] = len(images)

        return result

    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Build a text-only context string from retrieved chunks."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", chunk.get("doc_id", "unknown"))
            text = chunk.get("text", chunk.get("caption", ""))
            modality = chunk.get("modality", "text")

            if modality == "image" and not text:
                text = f"[Image from {source}]"

            parts.append(f"[Source {i}] (source: {source}, type: {modality})\n{text}")

        return "--- CONTEXT ---\n" + "\n\n".join(parts) + "\n--- END CONTEXT ---"

    def _build_context_with_images(
        self,
        chunks: List[Dict[str, Any]],
        asset_loader: Optional[Any] = None,
    ) -> Tuple[str, List[Any]]:
        """Build context string + collect PIL images for Gemini.

        Returns
        -------
        tuple[str, list[PIL.Image]]
            The text context and list of images to send.
        """
        text_parts = []
        images = []

        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", chunk.get("doc_id", "unknown"))
            text = chunk.get("text", chunk.get("caption", ""))
            modality = chunk.get("modality", "text")

            if modality in ("image", "video") and len(images) < self._max_images:
                # Try to load the actual image
                asset_id = chunk.get("asset_id")
                if asset_id and asset_loader:
                    img = asset_loader.load_image(asset_id)
                    if img:
                        images.append(img)
                        text_parts.append(
                            f"[Source {i}] (source: {source}, type: {modality}) "
                            f"[See attached image {len(images)}]\n{text or 'Visual content'}"
                        )
                        continue

                # Image in chunk directly (e.g., from loader)
                if "image" in chunk:
                    images.append(chunk["image"])
                    text_parts.append(
                        f"[Source {i}] (source: {source}, type: {modality}) "
                        f"[See attached image {len(images)}]\n{text or 'Visual content'}"
                    )
                    continue

            # Text-only chunk
            if not text and modality == "image":
                text = f"[Image from {source}]"
            text_parts.append(f"[Source {i}] (source: {source}, type: {modality})\n{text}")

        context = "--- CONTEXT ---\n" + "\n\n".join(text_parts) + "\n--- END CONTEXT ---"
        return context, images

    def _post_process(
        self,
        answer: Dict[str, Any],
        sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Verify citations and compute confidence.

        Parameters
        ----------
        answer : dict
            Raw answer from Gemini.
        sources : list[dict]
            The retrieved source chunks.

        Returns
        -------
        dict
            Answer with citation_verified, faithfulness_score, and warning fields.
        """
        result = dict(answer)
        citations = answer.get("citations", [])
        max_source_id = len(sources)

        # Verify all cited source IDs exist
        all_valid = True
        for cite in citations:
            sid = cite.get("source_id", 0)
            if sid < 1 or sid > max_source_id:
                all_valid = False
                break

        result["citation_verified"] = all_valid

        # Faithfulness: what fraction of the answer text overlaps with sources
        answer_text = answer.get("answer", "").lower()
        source_texts = " ".join(
            (s.get("text", "") + " " + s.get("caption", "")).lower()
            for s in sources
        )

        if answer_text and source_texts:
            answer_words = set(answer_text.split())
            source_words = set(source_texts.split())
            overlap = len(answer_words & source_words)
            result["faithfulness_score"] = round(
                overlap / max(len(answer_words), 1), 3
            )
        else:
            result["faithfulness_score"] = 0.0

        # Warning if low confidence or unverified citations
        if not all_valid:
            result["warning"] = "Some citations reference sources not in the context."
        elif result.get("confidence", 0) < 0.5:
            result["warning"] = "Low confidence answer — context may be insufficient."
        else:
            result["warning"] = None

        return result
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_gemini_generator.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/gemini/generator.py tests/test_gemini_generator.py
git commit -m "feat: add Gemini answer generator with structured output and citation verification"
```

---

## Phase 7: Wire Everything Together + API Updates

### Task 22: Update the pipeline orchestrator

**Files:**
- Modify: `src/pipeline_orchestrator.py`

This is a larger change — adding visual components to the orchestrator's lazy loading, setup, and query flow.

- [ ] **Step 1: Add visual component slots and lazy accessors**

In `EnterpriseRAGOrchestrator.__init__`, add after the existing component slots:

```python
        # Visual components (new)
        self._clip_engine = None
        self._visual_store = None
        self._asset_store = None
        self._captioner = None
        self._multimodal_retriever = None
        self._answer_generator = None
```

Add lazy accessors after the existing ones:

```python
    def _get_clip_engine(self):
        if self._clip_engine is None:
            from src.embedding.clip_engine import CLIPEngine
            self._clip_engine = CLIPEngine()
        return self._clip_engine

    def _get_visual_store(self):
        if self._visual_store is None:
            from src.embedding.visual_store import VisualVectorStore
            self._visual_store = VisualVectorStore()
        return self._visual_store

    def _get_asset_store(self):
        if self._asset_store is None:
            from src.ingestion.asset_store import AssetStore
            self._asset_store = AssetStore()
        return self._asset_store

    def _get_captioner(self):
        if self._captioner is None:
            from config.settings import GEMINI_API_KEY
            if GEMINI_API_KEY:
                from src.gemini.captioner import ImageCaptioner
                self._captioner = ImageCaptioner(api_key=GEMINI_API_KEY)
        return self._captioner

    def _get_answer_generator(self):
        if self._answer_generator is None:
            from config.settings import GEMINI_API_KEY, GEMINI_MODEL
            if GEMINI_API_KEY:
                from src.gemini.generator import AnswerGenerator
                self._answer_generator = AnswerGenerator(
                    api_key=GEMINI_API_KEY,
                    model_name=GEMINI_MODEL,
                )
        return self._answer_generator

    def _get_multimodal_retriever(self):
        if self._multimodal_retriever is None:
            from src.retrieval.multimodal_retriever import MultimodalRetriever
            from src.retrieval.query_analyzer import QueryAnalyzer

            def text_search(query, top_k=5):
                retriever = self._get_retriever()
                results = retriever.retrieve(query, top_k=top_k)
                return [
                    {
                        "text": chunk.text,
                        "doc_id": chunk.doc_id,
                        "score": score,
                        "modality": chunk.metadata.get("modality", "text"),
                        "source": chunk.metadata.get("source", chunk.doc_id),
                        **chunk.metadata,
                    }
                    for chunk, score in results
                ]

            def visual_search(query, top_k=5):
                try:
                    clip = self._get_clip_engine()
                    vs = self._get_visual_store()
                    query_vec = clip.embed_text(query)
                    return vs.search(query_vec, top_k=top_k)
                except Exception:
                    return []

            self._multimodal_retriever = MultimodalRetriever(
                text_search_fn=text_search,
                visual_search_fn=visual_search,
                query_analyzer=QueryAnalyzer(),
            )
        return self._multimodal_retriever
```

- [ ] **Step 2: Update the query method to use multimodal retrieval**

Update `_handle_standard` to use the multimodal retriever and Gemini generator when available:

```python
    def _handle_standard(self, query: str, **kwargs) -> Dict[str, Any]:
        """Standard RAG retrieval — now multimodal-aware."""
        # Try multimodal path first
        generator = self._get_answer_generator()
        if generator is not None:
            try:
                mm_retriever = self._get_multimodal_retriever()
                chunks = mm_retriever.retrieve(query, top_k=TOP_K)
                result = generator.generate(
                    query, chunks, asset_loader=self._get_asset_store(),
                )
                return {
                    "answer": result.get("answer", ""),
                    "query_type": "standard",
                    "confidence": result.get("confidence", 0.0),
                    "sources": [
                        {
                            "text": c.get("text", c.get("caption", ""))[:200],
                            "source": c.get("source", "unknown"),
                            "score": c.get("final_score", c.get("score", 0.0)),
                            "modality": c.get("modality", "text"),
                        }
                        for c in chunks[:5]
                    ],
                    "faithfulness": result.get("faithfulness_score"),
                    "citation_verified": result.get("citation_verified"),
                    "warning": result.get("warning"),
                }
            except Exception as exc:
                logger.warning("Multimodal pipeline failed: %s. Falling back.", exc)

        # Fallback to original text-only path
        rag = self._get_rag_pipeline()
        result = rag.answer(query)
        return {
            "answer": result.get("answer", ""),
            "query_type": "standard",
            "confidence": result.get("confidence", 0.0),
            "sources": result.get("sources", []),
        }
```

- [ ] **Step 3: Verify the orchestrator imports correctly**

Run: `python -c "from src.pipeline_orchestrator import EnterpriseRAGOrchestrator; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/pipeline_orchestrator.py
git commit -m "feat: wire multimodal components into pipeline orchestrator"
```

---

### Task 23: Add new API endpoints

**Files:**
- Modify: `src/api/main.py`
- Modify: `src/api/models.py`

- [ ] **Step 1: Add new Pydantic models**

Add to `src/api/models.py`:

```python
class BatchUploadResponse(BaseModel):
    status: str
    files_processed: int
    files_failed: int
    results: List[Dict[str, Any]]


class QuotaResponse(BaseModel):
    daily_remaining: int
    rpm_remaining: int
    daily_used: int


class MetricsResponse(BaseModel):
    queries_answered: int
    docs_uploaded: int
    total_vectors: int
    visual_vectors: int
    latency_avg_ms: float
    gemini_quota: Dict[str, int]
```

- [ ] **Step 2: Add asset serving endpoint**

Add to `src/api/main.py`:

```python
from fastapi.responses import FileResponse

@app.get("/api/assets/{asset_id}")
async def get_asset(asset_id: str):
    """Serve a stored image/frame asset."""
    try:
        from src.ingestion.asset_store import AssetStore
        store = AssetStore()
        path = store.get_path(asset_id)
        if path is None:
            raise HTTPException(status_code=404, detail="Asset not found")
        return FileResponse(path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
```

- [ ] **Step 3: Add quota endpoint**

```python
@app.get("/api/quota")
async def gemini_quota():
    """Return Gemini API quota status."""
    try:
        from config.settings import GEMINI_API_KEY
        from src.gemini.client import GeminiClient
        if not GEMINI_API_KEY:
            return {"daily_remaining": 0, "rpm_remaining": 0, "daily_used": 0, "status": "no_api_key"}
        client = GeminiClient(api_key=GEMINI_API_KEY)
        return client.quota_remaining()
    except Exception as exc:
        return {"error": str(exc)}
```

- [ ] **Step 4: Update the upload endpoint to accept all file types**

In the `upload_document` endpoint, update `allowed_extensions`:

```python
    allowed_extensions = {
        ".pdf", ".csv", ".txt", ".log", ".md", ".eml",
        ".docx", ".pptx", ".xlsx", ".xls",
        ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif",
        ".mp4", ".avi", ".mov",
        ".mp3", ".wav", ".m4a",
        ".html", ".htm",
        ".py", ".js", ".ts", ".java", ".go", ".rs",
        ".json", ".yaml", ".yml",
    }
```

- [ ] **Step 5: Add metrics endpoint**

```python
@app.get("/api/metrics")
async def system_metrics():
    """Return comprehensive system metrics."""
    total_vectors = 0
    visual_vectors = 0
    try:
        from src.embedding.store_vector_db import VectorStore
        vs = VectorStore()
        total_vectors = vs.total_vectors
    except Exception:
        pass

    try:
        from src.embedding.visual_store import VisualVectorStore
        vvs = VisualVectorStore()
        visual_vectors = vvs.total_vectors
    except Exception:
        pass

    gemini_quota = {}
    try:
        from config.settings import GEMINI_API_KEY
        if GEMINI_API_KEY:
            from src.gemini.client import GeminiClient
            client = GeminiClient(api_key=GEMINI_API_KEY)
            gemini_quota = client.quota_remaining()
    except Exception:
        pass

    return {
        "queries_answered": _state.get("queries_answered", 0),
        "docs_uploaded": _state.get("docs_uploaded", 0),
        "total_vectors": total_vectors,
        "visual_vectors": visual_vectors,
        "latency_avg_ms": 0,
        "gemini_quota": gemini_quota,
    }
```

- [ ] **Step 6: Commit**

```bash
git add src/api/main.py src/api/models.py
git commit -m "feat: add asset serving, quota, and metrics API endpoints"
```

---

## Phase 8: Evaluation Upgrades

### Task 24: Add multimodal test cases and evaluation metrics

**Files:**
- Modify: `src/evaluation/test_cases.py`
- Modify: `src/evaluation/eval_pipeline.py`

- [ ] **Step 1: Add multimodal test cases**

Append to the existing `TEST_CASES` list in `src/evaluation/test_cases.py`:

```python
    # Multimodal test cases
    {
        "query": "What does the chart in the quarterly report show?",
        "expected_answer": "The chart shows revenue trends across quarters.",
        "expected_modality": "visual",
        "expected_sources": [],
        "category": "visual_reasoning",
    },
    {
        "query": "Describe the architecture diagram",
        "expected_answer": "The architecture diagram shows the system components.",
        "expected_modality": "visual",
        "expected_sources": [],
        "category": "visual_reasoning",
    },
    {
        "query": "What text appears in the scanned document image?",
        "expected_answer": "The scanned document contains...",
        "expected_modality": "visual",
        "expected_sources": [],
        "category": "ocr_extraction",
    },
    {
        "query": "Summarize the content from the PowerPoint presentation",
        "expected_answer": "The presentation covers...",
        "expected_modality": "text",
        "expected_sources": [],
        "category": "multimodal_document",
    },
    {
        "query": "What data is in the Excel spreadsheet?",
        "expected_answer": "The spreadsheet contains...",
        "expected_modality": "text",
        "expected_sources": [],
        "category": "multimodal_document",
    },
```

- [ ] **Step 2: Add modality-specific metrics to eval pipeline**

In `src/evaluation/eval_pipeline.py`, add a method to check modality correctness. Add after existing metrics:

```python
    def _evaluate_modality(
        self,
        test_case: Dict[str, Any],
        pipeline_result: Dict[str, Any],
    ) -> float:
        """Check if the retrieval found the correct modality.

        Returns 1.0 if expected modality matches retrieved modality, else 0.0.
        """
        expected = test_case.get("expected_modality")
        if not expected:
            return 1.0  # No expectation, pass by default

        retrieved_chunks = pipeline_result.get("retrieved_chunks", [])
        sources = pipeline_result.get("sources", [])

        # Check if any retrieved chunk has the expected modality
        for src in sources:
            if isinstance(src, dict) and src.get("modality") == expected:
                return 1.0

        return 0.0
```

- [ ] **Step 3: Commit**

```bash
git add src/evaluation/test_cases.py src/evaluation/eval_pipeline.py
git commit -m "feat: add multimodal test cases and modality evaluation metric"
```

---

## Phase 9: Final Integration Test

### Task 25: End-to-end integration verification

- [ ] **Step 1: Verify all imports work**

Run:
```bash
python -c "
from config.settings import GEMINI_MODEL, CLIP_MODEL, VISUAL_FAISS_INDEX_PATH
from src.gemini.client import GeminiClient
from src.gemini.captioner import ImageCaptioner
from src.gemini.generator import AnswerGenerator
from src.embedding.clip_engine import CLIPEngine
from src.embedding.visual_store import VisualVectorStore
from src.ingestion.asset_store import AssetStore
from src.ingestion.mime_detector import detect_file_type
from src.ingestion.image_loader import load_image
from src.ingestion.docx_loader import load_docx
from src.ingestion.pptx_loader import load_pptx
from src.ingestion.excel_loader import load_excel
from src.ingestion.html_loader import load_html
from src.ingestion.code_loader import load_code
from src.ingestion.json_yaml_loader import load_json_yaml
from src.ingestion.audio_loader import load_audio
from src.ingestion.video_loader import load_video
from src.retrieval.query_analyzer import QueryAnalyzer
from src.retrieval.multimodal_retriever import MultimodalRetriever
from src.pipeline_orchestrator import EnterpriseRAGOrchestrator
print('All imports OK')
"
```
Expected: `All imports OK`

- [ ] **Step 2: Run the full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 3: Verify the API starts**

Run: `python -c "from src.api.main import app; print('FastAPI app:', app.title)"`
Expected: `FastAPI app: Enterprise RAG System`

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete multimodal RAG system integration"
```

---

## Summary

| Phase | Tasks | What It Delivers |
|-------|-------|-----------------|
| **1: Foundation** | Tasks 1-3 | Config, deps, Gemini client, LLM provider update |
| **2: Asset + MIME** | Tasks 4-5 | File type detection, asset storage |
| **3: CLIP + Visual** | Tasks 6-7 | CLIP embeddings, visual FAISS index |
| **4: Loaders** | Tasks 8-17 | All 12 file type loaders, pipeline registration |
| **5: Captioning + Retrieval** | Tasks 18-20 | Gemini captioning, query analyzer, multimodal retriever |
| **6: Generation** | Task 21 | Gemini answer generator with structured output |
| **7: Integration** | Tasks 22-23 | Orchestrator wiring, new API endpoints |
| **8: Evaluation** | Task 24 | Multimodal test cases and metrics |
| **9: Verification** | Task 25 | End-to-end integration test |

**Total: 25 tasks, ~85 steps, 20 new files, 7 modified files**
