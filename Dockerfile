# CineMatch — Dockerfile
#
# Build:
#   docker build -t cinematch .
#
# Run Streamlit:
#   docker run -p 8501:8501 -e TMDB_API_KEY=your_key cinematch
#
# Run FastAPI:
#   docker run -p 8000:8000 cinematch uvicorn api.main:app --host 0.0.0.0 --port 8000
#
# Required environment variables:
#   TMDB_API_KEY   — TMDB v3 API key for poster/trailer lookups (optional;
#                    the recommendation engine works without it).
#
# Secrets must NEVER be baked into the image.
# Pass them at runtime with:  -e TMDB_API_KEY=...  or  --env-file .env
#
# The model artifacts (models/) must exist before building the image.
# Generate them locally first:
#   python scripts/build_model.py
# ---------------------------------------------------------------------------

FROM python:3.12-slim

# Keeps Python from generating .pyc files on import
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
# Streamlit: disable browser auto-open and usage statistics
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/       ./src/
COPY api/       ./api/
COPY app_pages/ ./app_pages/
COPY app.py     ./app.py
COPY assets/    ./assets/

# Copy data directory (CSV files excluded via .dockerignore; only kept if present)
COPY data/      ./data/

# Copy pre-built model artifacts
# These must be generated before the Docker build:
#   python scripts/build_model.py
COPY models/    ./models/

# Expose both the Streamlit and FastAPI ports
EXPOSE 8501 8000

# Default command: run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
