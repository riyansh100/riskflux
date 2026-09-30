# syntax=docker/dockerfile:1
# RiskFlux inference image: FastAPI + the @champion model baked in (D10).
#
# Build (after `python -m riskflux.serving.fetch_model` has filled build/model):
#   docker build -t riskflux:v1 .
# Run:
#   docker run --rm -p 8080:8080 riskflux:v1

# ---- builder: resolve dependencies from uv.lock into a venv ----------------------
FROM python:3.11-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencies first: this layer is cached until uv.lock changes, so code edits rebuild fast.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Then the riskflux package itself, installed as a normal (non-editable) package.
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# ---- runtime: slim image with only the venv and the model --------------------------
FROM python:3.11-slim-bookworm AS runtime
# LightGBM needs the OpenMP runtime (the Linux counterpart of `brew install libomp`).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 app

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY build/model /app/model

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    RISKFLUX_MODEL_DIR=/app/model \
    PORT=8080
USER app
EXPOSE 8080

# Local Docker only; Cloud Run uses its own probes (configured at deploy time).
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/ready', timeout=2)"]

# One uvicorn worker per container: Cloud Run scales by adding containers.
# exec -> uvicorn is PID 1 and receives SIGTERM directly for graceful shutdown.
# Uvicorn's own plain-text logs are limited to warnings; app logs are JSON (serving/logs.py).
CMD ["sh", "-c", "exec uvicorn riskflux.serving.app:app --host 0.0.0.0 --port ${PORT} --no-access-log --log-level warning"]
