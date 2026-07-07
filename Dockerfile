# ── Stage 1: Build & Dependencies ───────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies needed to compile certain wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install to a local directory
COPY project/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# ── Stage 2: Final Runtime Image ───────────────────────────
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime dependencies (like tesseract for local OCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy backend source files
COPY project/ ./project/

# Expose port
EXPOSE 8000

# Set environment defaults
ENV APP_ENV=production
ENV LOG_LEVEL=INFO
ENV HOST=0.0.0.0
ENV PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Start FastAPI server
CMD ["sh", "-c", "uvicorn project.api:app --host $HOST --port $PORT"]
