# Multi-stage build for RapidFire AI end-user image

# Build arguments
ARG VERSION=0.10.2

# Stage 1: Build frontend assets
FROM node:22-slim AS frontend-builder

WORKDIR /build

# Copy only package files first for better caching
COPY rapidfireai/frontend/package.json rapidfireai/frontend/yarn.lock ./
COPY rapidfireai/frontend/yarn ./yarn

# Install frontend dependencies
RUN node ./yarn/releases/yarn-4.9.1.cjs install --frozen-lockfile

# Copy frontend source and build
COPY rapidfireai/frontend .
RUN node ./yarn/releases/yarn-4.9.1.cjs build

# Stage 2: Runtime image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Re-declare build argument (needed after FROM)
ARG VERSION=0.10.2

# Metadata
LABEL maintainer="RapidFire AI"
LABEL description="RapidFire AI - Fast LLM fine-tuning with GPU support"
LABEL version="${VERSION}"

# Avoid interactive prompts during build
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install Python 3.12 and system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    git \
    ca-certificates \
    lsof \
    netcat-openbsd \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Create non-root user for running the application
RUN useradd -m -u 1000 -s /bin/bash rfuser

# Set working directory
WORKDIR /app

# Create virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install RapidFire AI from PyPI
# This includes all dependencies: torch, transformers, peft, trl, mlflow, etc.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir rapidfireai

# Run rapidfireai init to copy tutorial notebooks
# Note: GPU-specific packages are already included in the rapidfireai package
RUN rapidfireai init || echo "Init completed"

# Copy built frontend from Stage 1
COPY --from=frontend-builder /build/build /opt/venv/lib/python3.12/site-packages/rapidfireai/frontend/build

# Create necessary directories for data persistence
RUN mkdir -p /app/mlruns \
    /app/rapidfire_experiments \
    /app/tutorial_notebooks \
    /app/logs \
    /app/data \
    && chown -R rfuser:rfuser /app /opt/venv

# Switch to non-root user
USER rfuser

# Expose ports
# 3000: Frontend dashboard
# 5002: MLflow tracking server  
# 8081: Dispatcher API
EXPOSE 3000 5002 8081

# Environment variables with defaults (can be overridden)
ENV RF_MLFLOW_PORT=5002 \
    RF_MLFLOW_HOST=0.0.0.0 \
    RF_FRONTEND_PORT=3000 \
    RF_FRONTEND_HOST=0.0.0.0 \
    RF_API_PORT=8081 \
    RF_API_HOST=0.0.0.0 \
    RF_EXPERIMENT_PATH=/app/rapidfire_experiments \
    RF_TUTORIAL_PATH=/app/tutorial_notebooks \
    MLFLOW_URL=http://localhost:5002

# Health check to ensure services are running
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5002/ && \
        curl -f http://localhost:8081/dispatcher/health-check && \
        curl -f http://localhost:3000/ || exit 1

# Start RapidFire AI services
ENTRYPOINT ["rapidfireai"]
CMD ["start"]

