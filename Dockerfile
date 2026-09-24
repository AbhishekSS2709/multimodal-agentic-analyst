# Multimodal Agentic Analyst: FastAPI backend + Streamlit UI in one container.
# Built for Hugging Face Spaces (Docker SDK, port 7860); runs anywhere Docker does.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache

# tesseract: OCR for scanned images; libmagic: MIME detection;
# libgl1/libglib2.0-0: OpenCV runtime; ffmpeg: audio/video ingestion.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      tesseract-ocr libmagic1 libgl1 libglib2.0-0 ffmpeg curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first: the default wheel pulls ~2 GB of CUDA libraries that
# a CPU container can never use.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt

COPY . .

# Download the embedding models and build the FAISS/BM25/SQLite/graph indices
# at build time, so the container starts in seconds and a broken index fails
# the build instead of shipping. Set BUILD_INDEX=0 for a quick dependency check.
ARG BUILD_INDEX=1
RUN if [ "$BUILD_INDEX" = "1" ]; then python scripts/build_index.py; fi

# Spaces runs the container as uid 1000; only the directories the app writes
# to at runtime are handed over, not the multi-GB model cache.
RUN useradd -m -u 1000 user \
 && mkdir -p vector_store static/charts evaluation_results \
 && chown -R user:user data vector_store static evaluation_results

USER user

ENV DEMO_MODE=1 \
    DEMO_DAILY_QUERY_LIMIT=200 \
    API_BASE=http://127.0.0.1:8000 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PORT=7860

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl -sf "http://127.0.0.1:${PORT}/_stcore/health" || exit 1

CMD ["bash", "docker/start.sh"]
