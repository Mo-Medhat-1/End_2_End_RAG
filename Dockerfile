FROM python:3.12-slim

# Prevent Python from writing .pyc files to disk and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (Tesseract OCR for scanned PDF support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements first to leverage Docker build cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-privileged user to run the application for security hardening
RUN useradd -u 10001 -U -m appuser \
    && mkdir -p /app/data/raw /app/vectorstore/faiss_index \
    && chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Copy application source code (excluding ignored files in .dockerignore)
COPY --chown=appuser:appuser . .

# Expose the default Streamlit port
EXPOSE 8501

# Lightweight health check using Python's native urllib (avoids installing curl in slim image)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Run Streamlit with headless settings
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
