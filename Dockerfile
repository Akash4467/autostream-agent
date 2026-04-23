# ── Stage 1: dependency builder ─────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

RUN pip install --upgrade pip

# ✅ STEP 1: CPU-only PyTorch FIRST (~250 MB) — prevents pip from pulling
#    the default CUDA wheel (torch 530 MB + cuDNN 366 MB) that
#    sentence-transformers would otherwise trigger automatically.
RUN pip install --prefix=/install \
        torch==2.2.2+cpu \
        torchvision==0.17.2+cpu \
        --extra-index-url https://download.pytorch.org/whl/cpu

# ✅ STEP 2: Everything else — torch is already satisfied above,
#    pip skips the CUDA index entirely.
RUN pip install --prefix=/install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt


# ── Stage 2: ingest ──────────────────────────────────────────
# Runs ingest.py at build time so chroma_db is baked into the image.
# The container starts instantly — no manual ingest step needed.
FROM python:3.11-slim AS ingest

WORKDIR /app

COPY --from=builder /install /usr/local

COPY knowledge_base/ ./knowledge_base/

RUN python knowledge_base/ingest.py

# ── Stage 3: runtime image ───────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="AutoStream AI Agent"
LABEL description="LangGraph + Streamlit conversational lead-capture agent"

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed dependencies
COPY --from=builder /install /usr/local

# Copy app code
COPY --chown=appuser:appuser . .

# ✅ Copy pre-built Chroma DB from ingest stage (overwrites any local chroma_db)
COPY --from=ingest --chown=appuser:appuser /app/chroma_db ./chroma_db

# ✅ Fix permissions for DB + data
RUN mkdir -p /data /app/chroma_db && \
    chown -R appuser:appuser /data /app/chroma_db && \
    chmod -R 755 /data /app/chroma_db

# ✅ Streamlit config
RUN mkdir -p /home/appuser/.streamlit && \
    printf "[server]\nport=8501\naddress=\"0.0.0.0\"\nheadless=true\nenableCORS=false\nenableXsrfProtection=false\n\n[browser]\ngatherUsageStats=false\n\n[theme]\nbase=\"dark\"\n" > /home/appuser/.streamlit/config.toml && \
    chown -R appuser:appuser /home/appuser/.streamlit

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8501

# Healthcheck — increased start_period to 40s to allow ingest model warm-up
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Start app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]