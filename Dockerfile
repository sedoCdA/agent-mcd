# Use an official lightweight Python image
FROM python:3.10-slim

# Install system dependencies (needed for audio processing/Whisper file formatting)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for Python performance and paths
ENV PYTHONUNBUFFERED=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Create a non-root user for Hugging Face security compliance
RUN useradd -m -u 1000 user
WORKDIR $HOME/app

# Copy requirements first to optimize Docker build caching
COPY --chown=user:user requirements.txt .

# Install dependencies as the non-root user
USER user
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY --chown=user:user . .

# Expose port 7860 (Hugging Face expects port 7860 by default)
EXPOSE 7860

# Run FastAPI backend using Uvicorn on port 7860
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]