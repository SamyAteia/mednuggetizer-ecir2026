# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies for PDF processing and ML libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /tmp/uploads

# Expose port 4000 (as per app.py)
EXPOSE 4000

# Set environment variables
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:4000/api/health || exit 1

# Run the application with a single Gunicorn worker so SSE progress state stays shared
# gthread with multiple threads provides concurrency without splitting processes
CMD ["gunicorn", "--bind", "0.0.0.0:4000", "--workers", "1", "--worker-class", "gthread", "--threads", "8", "--timeout", "300", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
