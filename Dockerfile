# Multi-stage Dockerfile for production deployment
# Optimized for size and security

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM python:3.11-slim

# Create non-root user
RUN useradd -m -u 1000 router && \
    mkdir -p /app && \
    chown -R router:router /app

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder --chown=router:router /root/.local /home/router/.local

# Copy application code
COPY --chown=router:router src/ ./src/
COPY --chown=router:router training/ ./training/

# Set Python path
ENV PATH=/home/router/.local/bin:$PATH
ENV PYTHONPATH=/app

# Switch to non-root user
USER router

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/v1/health')"

# Run application
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
