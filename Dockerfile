# Use multi-stage build for optimization
FROM python:3.13.8-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /uvx /bin/

WORKDIR /app

ENV PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean


COPY pyproject.toml uv.lock ./


RUN uv sync --locked --no-dev --no-install-project --no-cache

# ------------------------------------------------------------------------------------
FROM python:3.13.8-slim

# Labels for better maintainability
LABEL maintainer="Valentin LAURENT" \
      version="1.0" \
      description="Document Q&A RAG Application"


ENV PYTHONPATH=/app/src
ENV PATH="/app/.venv/bin:$PATH"

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder stage
COPY --from=builder /app/.venv /app/.venv

WORKDIR /app

# Expose port (Gradio default)
EXPOSE 7860
