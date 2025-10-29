FROM python:3.12-slim

# Avoid buffering so logs appear in real time
ENV PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1

# Install system dependencies required by many packages (OpenCV, ffmpeg, build tools)
RUN apt-get update && \
	apt-get install -y --no-install-recommends \
		build-essential \
		gcc \
		git \
		curl \
		ffmpeg \
		libgl1 \
		libglib2.0-0 \
		libsm6 \
		libxext6 \
		pkg-config \
		libgomp1 && \
	rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and filter out GPU-only NVIDIA/triton packages that will fail in a CPU-only VM.
# We do this at build time so the original requirements.txt in the repo is untouched.
COPY requirements.txt ./

# Upgrade pip/tools and install Python dependencies from a filtered requirements file.
RUN python -m pip install --upgrade pip setuptools wheel && \
	# remove lines that begin with nvidia-, nvidia_ or the triton package which are GPU-specific
	grep -v -E '^(nvidia-|nvidia_|triton)' requirements.txt > requirements.docker.txt || true && \
	pip install --no-cache-dir -r requirements.docker.txt

# Copy application code
COPY . .

# Use a non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# App listens on 8000 by default (uvicorn)
EXPOSE 8000

# Default command - run the FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
