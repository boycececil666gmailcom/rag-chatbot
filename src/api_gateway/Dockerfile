# ==============================================================================
# Lightweight Dockerfile for Fintech RAG API Gateway
# ==============================================================================
FROM python:3.12-slim

WORKDIR /app

# Install only minimal packages required for gateway execution
# This avoids compiling native C modules like flashrank on the gateway container
COPY src/api_gateway/requirements.txt /app/src/api_gateway/requirements.txt
RUN pip install --no-cache-dir -r /app/src/api_gateway/requirements.txt

# Create a non-root user for production security compliance
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -m -s /bin/bash appuser

# Copy application files (specifically including core config & models)
COPY --chown=appuser:appgroup src/api_gateway/ /app/src/api_gateway/

ENV PYTHONUNBUFFERED=1
ENV PATH=/home/appuser/.local/bin:$PATH

# Expose API Gateway default port
EXPOSE 8080

# Switch to non-root user
USER appuser

# Run API Gateway using uvicorn
CMD ["uvicorn", "src.api_gateway.main:app", "--host", "0.0.0.0", "--port", "8080"]
