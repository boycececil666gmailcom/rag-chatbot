# ==============================================================================
# Lightweight Dockerfile for Fintech RAG API Gateway
# ==============================================================================
FROM python:3.11-slim

WORKDIR /app

# Install only minimal packages required for gateway execution
# This avoids compiling native C modules like chromadb/flashrank on the gateway container
RUN pip install --no-cache-dir fastapi uvicorn httpx pydantic

# Create a non-root user for production security compliance
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -m -s /bin/bash appuser

# Copy application files (specifically including core config & models)
COPY --chown=appuser:appgroup src/ /app/src/

ENV PYTHONUNBUFFERED=1
ENV PATH=/home/appuser/.local/bin:$PATH

# Expose API Gateway default port
EXPOSE 8080

# Switch to non-root user
USER appuser

# Run API Gateway using uvicorn
CMD ["uvicorn", "src.api_gateway.main:app", "--host", "0.0.0.0", "--port", "8080"]
