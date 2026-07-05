# Dockerfile for SEM-Swarm API
# This file is built to be deployed on Easypanel.

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install uv for fast dependency resolution and some basic tools
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

# Copy requirements first to leverage Docker cache
COPY api/requirements.txt ./

# Install dependencies using uv pip
RUN uv pip install --system --no-cache -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run the API via uvicorn (--app-dir puts api/ on sys.path so the flat
# imports in api/main.py — `from config import settings` — resolve)
CMD ["uvicorn", "main:app", "--app-dir", "api", "--host", "0.0.0.0", "--port", "8000"]
