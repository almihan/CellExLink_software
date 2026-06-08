# syntax=docker/dockerfile:1

# Minimal CPU image for CellExLink.
# Build with:
#   docker build -t cellexlink:0.1.0 .
#
# For development/testing inside Docker:
#   docker build --build-arg INSTALL_EXTRAS=dev -t cellexlink:dev .

FROM python:3.12-slim-bookworm

ARG INSTALL_EXTRAS=""
ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/cache/huggingface

# build-essential is needed when Python packages with native extensions are
# built from source, including abbreviation-related dependencies on some
# platforms. libgomp1 is commonly needed by scientific Python wheels.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash --uid 1000 cellexlink && \
    mkdir -p /opt/cellexlink /workspace /models /data /outputs /cache/huggingface && \
    chown -R cellexlink:cellexlink /opt/cellexlink /workspace /models /data /outputs /cache

WORKDIR /opt/cellexlink

# Copy package metadata first so Docker can reuse dependency layers when only
# source files change.
COPY --chown=cellexlink:cellexlink pyproject.toml README.md LICENSE.txt CITATION.cff MANIFEST.in ./
COPY --chown=cellexlink:cellexlink src ./src
COPY --chown=cellexlink:cellexlink examples ./examples
COPY --chown=cellexlink:cellexlink benchmarks ./benchmarks
COPY --chown=cellexlink:cellexlink tests ./tests

RUN python -m pip install --upgrade pip setuptools wheel && \
    if [ -n "$INSTALL_EXTRAS" ]; then \
        python -m pip install ".[${INSTALL_EXTRAS}]"; \
    else \
        python -m pip install "."; \
    fi

USER cellexlink
WORKDIR /workspace

# Model checkpoints are intentionally not copied into the image. Mount them at
# runtime, for example: -v "$PWD/models:/models:ro".
CMD ["cellexlink", "--help"]
