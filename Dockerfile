# syntax=docker/dockerfile:1.7

# ── Stage 1: builder ───────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Use BuildKit cache for pip downloads and install in one layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --prefer-binary -r requirements.txt


# ── Stage 2: runtime ───────────────────────────────────────
FROM python:3.11-slim AS runtime
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# copy installed packages
COPY --from=builder /usr/local /usr/local

# copy only runtime project files (keeps layer smaller and cache-friendly)
COPY --chown=appuser:appuser app.py main.py ./
COPY --chown=appuser:appuser agent ./agent
COPY --chown=appuser:appuser tools ./tools
COPY --chown=appuser:appuser knowledge_base ./knowledge_base

USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]