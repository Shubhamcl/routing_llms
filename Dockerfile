# Multi-stage Dockerfile for Router Quality Classifier API
# Supports both CPU and GPU deployment

# Base image with Python
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY router_api.py .

# Create directories for models and data
RUN mkdir -p /app/runs /app/models /app/data

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "router_api.py"]

# ---

# # GPU variant (if needed)
# FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as gpu-base

# # Set environment variables
# ENV PYTHONUNBUFFERED=1 \
#     PYTHONDONTWRITEBYTECODE=1 \
#     PIP_NO_CACHE_DIR=1 \
#     PIP_DISABLE_PIP_VERSION_CHECK=1 \
#     DEBIAN_FRONTEND=noninteractive

# # Install Python and system dependencies
# RUN apt-get update && apt-get install -y \
#     python3.10 \
#     python3.10-dev \
#     python3-pip \
#     build-essential \
#     curl \
#     && rm -rf /var/lib/apt/lists/* \
#     && ln -s /usr/bin/python3.10 /usr/bin/python

# # Create app directory
# WORKDIR /app

# # Copy requirements and install Python dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy application code
# COPY router_api.py .

# # Create directories
# RUN mkdir -p /app/runs /app/models /app/data

# # Expose port
# EXPOSE 8000

# # Health check
# HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
#     CMD curl -f http://localhost:8000/health || exit 1

# # Default command
# CMD ["python", "router_api.py"]
