# ==============================================================================
# Multi-stage Dockerfile for Theme-Based RAG Workflow Backend
# ==============================================================================

# --- Stage 1: Build dependencies ---
FROM python:3.12-slim AS builder

WORKDIR /app

# Install compilation tools required for flashrank
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY src/theme_based_rag_backend/requirements.txt /app/src/theme_based_rag_backend/requirements.txt

# Install dependencies into a local directory to copy to the runner stage
RUN pip install --no-cache-dir --user -r /app/src/theme_based_rag_backend/requirements.txt


# --- Stage 2: Production runner ---
FROM python:3.12-slim AS runner

WORKDIR /app

# Create a non-root user for security compliance in production Kubernetes
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -m -s /bin/bash appuser

# Copy installed Python packages from the builder stage
COPY --from=builder /root/.local /home/appuser/.local
COPY --chown=appuser:appgroup src/theme_based_rag_backend/ /app/src/theme_based_rag_backend/

# Update PATH variable to include user's local bin
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Expose FastAPI default port
EXPOSE 8000

# Switch to non-root user
USER appuser

# Run Chatbot Backend using uvicorn
CMD ["uvicorn", "src.theme_based_rag_backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
