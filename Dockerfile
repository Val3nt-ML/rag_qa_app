# Use multi-stage build for optimization
FROM python:3.13.8-slim as builder

ENV PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean


COPY requirements.txt ./


RUN pip install --upgrade pip uv \
    && uv pip install --system --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------------------------
FROM python:3.13.8-slim

# Labels for better maintainability
LABEL maintainer="Valentin LAURENT" \
      version="1.0" \
      description="Document Q&A RAG Application"


ENV PYTHONPATH=/app/src

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python packages from builder stage
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Expose port (Gradio default)
EXPOSE 7860
