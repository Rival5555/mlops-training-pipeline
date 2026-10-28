# Multi-stage production build for MLOps training pipeline

# Stage 1: Build & Dependency Resolution
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast, reliable package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project specification and sources required for build
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install dependencies into virtual environment
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN uv pip install --no-cache -e .

# Stage 2: Slim Production Runtime
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime dependencies for OpenCV/Pillow & Git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app

# Create unprivileged user for security compliance
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data /app/models /app/mlruns && \
    chown -R appuser:appuser /app

USER appuser

# Copy application source code and configs
COPY --chown=appuser:appuser configs/ /app/configs/
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser scripts/ /app/scripts/
COPY --chown=appuser:appuser Makefile /app/Makefile
COPY --chown=appuser:appuser dvc.yaml /app/dvc.yaml
COPY --chown=appuser:appuser README.md /app/README.md

# Health check verifies Python environment & imports
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import torch, mlflow, prefect; print('Healthy')" || exit 1

ENTRYPOINT ["python", "-m", "src.pipeline.flow"]
CMD ["--config", "configs/config.yaml"]
