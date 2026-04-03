# Multimodal Production-Grade RAG System — Design Spec

**Date:** 2026-04-03
**Approach:** Gemini-Native Multimodal (Approach 1) — Incremental Rebuild
**Status:** Approved

---

## 1. Overview

Upgrade the existing Enterprise RAG system from text-only to a full multimodal pipeline supporting all major content types (PDF, images, video, audio, Excel, Word, PowerPoint, HTML, code, JSON/YAML). The system uses Gemini 2.5 Flash (free tier) as the primary LLM with native vision capabilities, dual embedding indices (BGE for text, CLIP for visual), and a production-grade evaluation/observability stack. Deployed on Azure, serving both a polished UI and a public API.

### Key Decisions

- **LLM Provider:** Gemini 2.5 Flash (free tier, 10 RPM / 250 req/day). Fallback to Flash-Lite, then local HuggingFace.
- **Architecture:** Dual-index (text + visual) with Gemini-native multimodal answer generation.
- **Rebuild Strategy:** Incremental — keep existing codebase, upgrade piece by piece.
- **Deployment:** Azure (App Service + Blob Storage + Azure SQL + Application Insights).
- **Target Users:** Non-technical business users (UI) + API consumers (integrations).

---

## 2. Ingestion Layer

### 2.1 Supported File Types & Loaders

| Loader | File Types | Extraction Strategy | Library |
|--------|-----------|---------------------|---------|
| `pdf_loader.py` | .pdf | Text extraction + page-as-image for visual content | `pypdf` (existing) + `Pillow` |
| `image_loader.py` | .png, .jpg, .webp, .bmp | OCR text + Gemini captioning + store original | `Pillow` + `pytesseract` + Gemini Vision |
| `video_loader.py` | .mp4, .avi, .mov | Scene-change keyframe extraction + audio transcription | `opencv-python` + Gemini / `faster-whisper` |
| `audio_loader.py` | .mp3, .wav, .m4a | Speech-to-text transcription | Gemini audio API / `faster-whisper` (fallback) |
| `docx_loader.py` | .docx | Text + embedded images extraction | `python-docx` |
| `pptx_loader.py` | .pptx | Slide text + slide-as-image + speaker notes | `python-pptx` + `Pillow` |
| `excel_loader.py` | .xlsx, .xls | Sheet-by-sheet table extraction + chart images | `openpyxl` |
| `html_loader.py` | .html, URLs | Clean text extraction (Playwright optional for JS) | `beautifulsoup4` + `requests` |
| `code_loader.py` | .py, .js, .ts, etc. | Code with syntax-aware chunking | Built-in with language detection |
| `json_yaml_loader.py` | .json, .yaml, .yml | Structured data flattening with path context | Built-in |
| `csv_loader.py` | .csv | Existing loader | `pandas` (existing) |
| `txt_loader.py` | .txt | Existing loader | Built-in (existing) |

### 2.2 Dual Output Per Document

Every loader produces two content types:

- **TextContent**: extracted text chunks -> BGE text index
- **VisualContent**: images/frames/diagrams -> CLIP visual index + stored as asset files

Both carry a shared `doc_id` and linked metadata.

### 2.3 Gemini-Assisted Captioning

At ingestion time, images/charts/diagrams are sent to Gemini 2.5 Flash for rich text captioning. These captions are embedded in the text index so text queries can discover visual content.

**Rate limit handling:** Captioning queue with rate limiting. When Gemini quota is exhausted, fall back to local BLIP-2 captioning.

### 2.4 File Type Detection

Use `python-magic` for MIME-type detection at the ingestion gateway. Route to the correct loader based on actual content type, not just file extension.

### 2.5 Modality Metadata

Every chunk is tagged with:

```python
metadata = {
    "modality": "image" | "text" | "audio" | "video" | "table",
    "source_file": "report.pptx",
    "page_or_slide": 12,
    "original_asset_path": "assets/slide_12.png",
}
```

### 2.6 Asset Storage

Original images and video frames are stored in `data/assets/` (local) or Azure Blob Storage (production). Metadata references the asset path. Assets are loaded at answer-generation time when Gemini needs to "see" them.

### 2.7 Video Frame Extraction

Use OpenCV's scene-change detection (`cv2.absdiff` with histogram comparison, threshold-based) to extract keyframes from video. This keeps the index lean — only frames where visual content actually changes are stored. Fallback: if scene detection fails, extract 1 frame every 10 seconds.

---

## 3. Embedding & Storage Layer

### 3.1 Dual-Index Architecture

```
Text chunks + captions ─── BGE-large-en-v1.5 (1024-dim) ─── FAISS Text Index
Images/frames ──────────── CLIP ViT-B/32 (512-dim) ───────── FAISS Visual Index
```

- **Text Index** (existing, upgraded): Stores text chunks + Gemini-generated captions of visual content. Searched with BGE query embedding.
- **Visual Index** (new): Stores CLIP embeddings of images/frames/diagrams. Searched with CLIP text encoder (shared embedding space).

### 3.2 Model Selection

| Model | Dimension | Size | Purpose |
|-------|-----------|------|---------|
| `BAAI/bge-large-en-v1.5` | 1024 | ~1.3 GB | Text embedding (existing) |
| `openai/clip-vit-base-patch32` | 512 | ~340 MB | Visual embedding (dev) |
| `openai/clip-vit-large-patch14` | 768 | ~900 MB | Visual embedding (production, GPU) |

CLIP model is configurable in `settings.py`. Start with ViT-B/32 for development, upgrade to ViT-L/14 on GPU-backed Azure tier for production.

### 3.3 Embedding Cache

Both text and image embeddings are cached on disk using hash-based keys. Extends the existing `EmbeddingEngine.save_cache()` pattern to cover CLIP embeddings.

### 3.4 BM25 Index

Existing BM25 index stays. Indexes text chunks + Gemini captions (not raw images). Remains valuable for keyword-heavy queries.

### 3.5 Metadata Store

SQLite sidecar (existing) mapping `chunk_id` to file path, modality, page number, asset path. Migrates to Azure SQL for production.

### 3.6 Azure Storage

- FAISS indices + metadata: Azure Blob Storage (persistent)
- Asset files (images, frames): Azure Blob Storage with `assets/` prefix
- SQLite: local for dev, Azure SQL for production

---

## 4. Retrieval & Reranking

### 4.1 Query Flow

```
User Query
    |
    v
Query Analyzer -- detects: modality hints, intent, complexity
    |
    +----------+----------+
    |                     |
    v                     v
Text Search           Visual Search
(BGE + BM25           (CLIP text-to-image)
 hybrid 60/40)
    |                     |
    v                     v
Score Fusion & Dedup -- normalize + weighted combine
    |
    v
Cross-Encoder Reranker -- modality-aware reranking
    |
    v
Top-K Results -- mixed text + visual chunks
```

### 4.2 Query Analyzer

Upgrade of existing `QueryRouter`. Now also detects:

- Does the query reference visual content? ("show me the chart", "what does the diagram say")
- Should we search visual index, text index, or both?
- Multi-hop needed? SQL needed? (existing logic stays)

### 4.3 Adaptive Weight Tuning

Instead of fixed weights, the query analyzer sets retrieval weights based on modality signal:

| Query Type | Text Weight | Visual Weight |
|-----------|-------------|---------------|
| Clearly visual ("show me the diagram") | 0.3 | 0.7 |
| Clearly textual ("what's the refund policy") | 0.9 | 0.1 |
| Ambiguous (default) | 0.6 | 0.4 |

### 4.4 Score Normalization

BGE and CLIP scores are on different scales. Before fusion:

- Normalize each source's scores to [0, 1] using min-max normalization within each result set
- Then apply weighted combination

### 4.5 Deduplication

If a caption and its source image both rank high, keep the image (richer context for Gemini). Deduplicate by `doc_id` + `page_or_slide`.

### 4.6 Modality-Aware Reranking

- **Text chunks**: Cross-encoder reranker (existing `CrossEncoderReranker`)
- **Visual chunks with captions**: Cross-encoder on caption text
- **Visual chunks without captions**: Score by CLIP similarity directly, bypass cross-encoder

### 4.7 Retrieval Fallback Chain

If primary retrieval returns low-confidence results (below threshold), automatically try the other modality:

1. Text search returns low scores -> try visual search
2. Visual search returns low scores -> try text search
3. Both low -> return best available with confidence warning

### 4.8 Multi-hop Retrieval

Existing `MultiHopRetriever` stays for complex reasoning queries across multiple documents. Extended to include visual chunks in the hop chain.

---

## 5. Answer Generation

### 5.1 Pipeline

```
Retrieved Results (text chunks + images)
    |
    v
Context Builder -- assembles prompt with text + images (max 3 images)
    |
    v
Gemini 2.5 Flash -- generates structured JSON answer with citations
    |
    v
Post-Processor -- citation verification, confidence scoring, hallucination guard
```

### 5.2 Context Builder

- Text chunks: inserted as numbered passages
- Visual chunks: attached as inline images (Gemini handles natively)
- Each source gets a reference tag `[Source 1]`, `[Source 2]`, etc.
- **Max 3 images per query** (configurable) — prioritized by retrieval score
- System prompt enforces: cite sources, say "I don't know" when unsure

### 5.3 Gemini Configuration

- **Model:** `gemini-2.5-flash` (free tier)
- **Temperature:** `0.1` (low for factual accuracy)
- **Max output:** `2048` tokens
- **Response format:** `response_mime_type="application/json"` for structured output
- **Rate limiting:** built-in retry with exponential backoff
- **SDK:** `google-genai`

### 5.4 Structured Output

Gemini returns:

```json
{
  "answer": "...",
  "citations": [{"source_id": 1, "quote": "..."}],
  "confidence": 0.85,
  "needs_visual": true
}
```

Fallback parser handles cases where Gemini breaks JSON format.

### 5.5 Post-Processor

- **Citation verification**: checks that cited sources actually exist in the context
- **Confidence scoring**: based on retrieval scores, citation coverage, answer length vs context length
- **Hallucination guard**: if answer contains claims not traceable to any source, flag with warning
- **Faithfulness score**: existing `SourceAttributor` logic, upgraded

### 5.6 Fallback Chain

If Gemini is unavailable:

1. **Fallback 1:** Gemini 2.5 Flash-Lite (higher rate limit: 15 RPM, 1000 req/day)
2. **Fallback 2:** Local HuggingFace model (`flan-t5`, text-only, no vision)
3. **Fallback 3:** Return retrieved context directly without generation

---

## 6. Evaluation & Observability

### 6.1 Automated Evaluation Pipeline

| Metric | What It Measures | Method |
|--------|-----------------|--------|
| Retrieval Precision@K | Are retrieved chunks relevant? | Gemini Pro as judge |
| Retrieval Recall | Did we find all relevant chunks? | Against labeled ground-truth test sets |
| MRR | Is the best chunk ranked first? | Position of first relevant result |
| Answer Faithfulness | Is the answer grounded in sources? | Cross-check claims against context |
| Answer Completeness | Does it address the full question? | Gemini Pro as judge |
| Citation Accuracy | Are citations correct? | Verify cited quotes exist in sources |
| Multimodal Retrieval | Can it find the right image/chart? | Test cases with visual ground truth |

### 6.2 Gemini-as-Judge

- **Generator:** Gemini 2.5 Flash (temperature 0.1)
- **Judge:** Gemini 2.5 Pro (temperature 0.0) — different model for independent evaluation
- Judge uses 5 RPM / 100 req/day from free tier (sufficient for eval batches)

### 6.3 Test Case Format

```python
{
    "query": "What does the Q3 revenue chart show?",
    "expected_answer": "Q3 revenue increased 15%...",
    "expected_modality": "visual",
    "expected_sources": ["quarterly_report.pdf"],
    "category": "visual_reasoning"
}
```

### 6.4 User Feedback Loop

- **Thumbs up/down** on answers (existing, kept)
- **Citation corrections**: user can flag "this source is wrong"
- **Answer corrections**: user provides the right answer -> stored as ground truth
- **Feedback-driven reranking**: chunks that get consistently downvoted are penalized in future retrievals (existing `learning_loop.py`, upgraded)

### 6.5 System Monitoring

| Signal | Tool | Tracks |
|--------|------|--------|
| Latency | Built-in timing | Per-component: ingestion, retrieval, reranking, generation |
| Gemini usage | Request counter | RPM, daily quota remaining, token consumption |
| Error rates | Logging + alerts | Failed retrievals, Gemini timeouts, ingestion errors |
| Index health | Periodic checks | FAISS index size, BM25 corpus size, stale documents |

---

## 7. API & Frontend

### 7.1 API Endpoints (FastAPI)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `POST /query` | existing | Upgraded: returns multimodal results, supports `?stream=true` |
| `POST /upload` | existing | Extended: handles all 12 file types |
| `POST /batch-upload` | new | Upload multiple files with progress tracking |
| `GET /assets/{asset_id}` | new | Serve stored images/frames |
| `POST /feedback` | existing | Extended: citation + answer corrections |
| `GET /metrics` | new | System health, latency percentiles, quota usage |
| `GET /eval/run` | existing | Trigger evaluation pipeline |
| `GET /eval/results` | existing | Fetch latest evaluation report |
| `GET /quota` | new | Gemini rate limit status |

### 7.2 API Improvements

- **Async endpoints**: `async def` for all Gemini calls
- **Dual response mode**: Streaming SSE (`?stream=true`) for UI, structured JSON for API clients
- **Auth middleware**: API key-based authentication
- **Rate limiting**: per-client limits to protect Gemini quota
- **CORS**: configured for frontend origin
- **Upload security**: 100MB limit, MIME-type validation via `python-magic`, reject executables

### 7.3 Frontend (Streamlit)

| Tab | Status | Features |
|-----|--------|----------|
| **Ask** | upgraded | Mixed text + image results inline, clickable citations, confidence badge, faithfulness score |
| **Upload** | upgraded | Drag-and-drop all file types, batch upload with progress, content preview |
| **Analytics Dashboard** | new | Query volume, latency trends, error rates, Gemini quota, retrieval accuracy over time |
| **Evaluation** | upgraded | Visual report with multimodal test case results |
| **Knowledge Graph** | existing | Kept as-is |
| **Settings** | new | Retrieval weights, Gemini model selection, chunk size, top-K, reranking toggle |

### 7.4 Azure Deployment

```
Azure Resource Group
  - App Service: FastAPI + Streamlit
  - Blob Storage: FAISS indices, assets (images/frames), documents
  - Azure SQL: feedback, metadata (replaces SQLite in production)
  - Application Insights: monitoring, alerting
  - External: Gemini API (google-genai SDK)
```

---

## 8. New Dependencies

| Package | Purpose | Required |
|---------|---------|----------|
| `google-genai` | Gemini API SDK | Yes |
| `Pillow` | Image processing | Yes |
| `pytesseract` | OCR fallback | Yes |
| `opencv-python` | Video frame extraction | Yes |
| `faster-whisper` | Audio transcription fallback | Yes |
| `python-docx` | Word document loading | Yes |
| `python-pptx` | PowerPoint loading | Yes |
| `openpyxl` | Excel loading | Yes |
| `beautifulsoup4` | HTML parsing | Yes |
| `python-magic` | MIME-type detection | Yes |
| `transformers` (CLIP) | Visual embedding | Yes |
| `playwright` | JS-rendered HTML scraping | Optional |
| `azure-storage-blob` | Azure Blob Storage SDK | Production |
| `azure-identity` | Azure auth | Production |
| `applicationinsights` | Azure monitoring | Production |

---

## 9. Configuration Additions (settings.py)

```python
# Gemini
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_JUDGE_MODEL = "gemini-2.5-pro"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_TEMPERATURE = 0.1
GEMINI_MAX_OUTPUT_TOKENS = 2048
GEMINI_MAX_IMAGES_PER_QUERY = 3

# Visual Embedding
CLIP_MODEL = "openai/clip-vit-base-patch32"  # dev
# CLIP_MODEL = "openai/clip-vit-large-patch14"  # production
CLIP_EMBEDDING_DIMENSION = 512  # 768 for ViT-L/14

# Visual FAISS
VISUAL_FAISS_INDEX_PATH = VECTOR_DB_DIR / "faiss_visual_index"

# Asset Storage
ASSETS_DIR = DATA_DIR / "assets"

# Retrieval Weights (defaults, adaptive at runtime)
DEFAULT_TEXT_WEIGHT = 0.6
DEFAULT_VISUAL_WEIGHT = 0.4

# Captioning
CAPTIONING_FALLBACK = "blip2"  # when Gemini quota exhausted
CAPTIONING_QUEUE_RPM = 8  # stay under Gemini's 10 RPM limit

# Upload Security
MAX_UPLOAD_SIZE_MB = 100
BLOCKED_MIME_TYPES = ["application/x-executable", "application/x-msdownload"]

# Azure (production)
AZURE_BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING", "")
AZURE_SQL_CONNECTION_STRING = os.getenv("AZURE_SQL_CONNECTION_STRING", "")
```

---

## 10. What Stays Unchanged

These existing components are kept as-is:

- `src/chunking/chunker.py` — SemanticChunker (minor extension: code-aware chunking added)
- `src/retrieval/bm25_search.py` — BM25Search (unchanged)
- `src/retrieval/hybrid_retriever.py` — HybridRetriever (minor change: weights become dynamic)
- `src/retrieval/multi_hop.py` — MultiHopRetriever
- `src/retrieval/reranker.py` — CrossEncoderReranker + SimpleReranker
- `src/retrieval/source_attribution.py` — SourceAttributor
- `src/knowledge_graph/` — graph builder + retriever
- `src/sql_tool/` — SQL analytics pipeline
- `src/feedback/` — feedback store + learning loop (extended)
- `src/evaluation/` — eval pipeline + report generator (extended)
- `src/pipeline_orchestrator.py` — orchestrator pattern (extended with visual components)
