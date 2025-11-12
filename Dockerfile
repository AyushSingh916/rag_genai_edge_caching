# Dockerfile for Hugging Face Spaces
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files from src directory
COPY src/ ./src/

# Expose port (Hugging Face Spaces uses port 7860)
EXPOSE 7860

# Health check (optional, Hugging Face handles this)
# HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Run Streamlit (app.py is in src directory)
CMD ["streamlit", "run", "src/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
