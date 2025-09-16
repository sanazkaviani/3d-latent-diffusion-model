# Multi-stage Docker build for 3D Latent Diffusion Model
# Optimized for production deployment with minimal image size

# Base stage with CUDA and PyTorch
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install Python packages in build stage
FROM base as builder

# Install pip and build tools
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt /tmp/requirements.txt
COPY mlops/azure/conda_env.yml /tmp/conda_env.yml

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir \
    fastapi==0.103.0 \
    uvicorn[standard]==0.23.0 \
    gunicorn==21.2.0 \
    prometheus-client==0.17.0 \
    azure-storage-blob==12.17.0 \
    azure-identity==1.14.0

# Production stage
FROM base as production

# Create non-root user for security
RUN groupadd -r modeluser && useradd -r -g modeluser modeluser

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.8/dist-packages /usr/local/lib/python3.8/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY train_autoencoder.py train_diffusion.py inference.py utils.py visualize_image.py ./
COPY config/ ./config/
COPY mlops/deployment/score.py ./
COPY mlops/docker/api_server.py ./
COPY mlops/docker/health_check.py ./

# Create directories for models and outputs
RUN mkdir -p /app/models /app/outputs /app/logs && \
    chown -R modeluser:modeluser /app

# Copy startup script
COPY mlops/docker/entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Switch to non-root user
USER modeluser

# Expose ports
EXPOSE 8000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python health_check.py

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"]

# Default command
CMD ["api"]

# Development stage for local development
FROM production as development

USER root

# Install development tools
RUN pip install --no-cache-dir \
    jupyter==1.0.0 \
    notebook==6.5.4 \
    ipykernel==6.25.0 \
    pytest==7.4.0 \
    black==23.7.0 \
    flake8==6.0.0

# Install VS Code server for remote development
RUN curl -fsSL https://code-server.dev/install.sh | sh

USER modeluser

# Expose additional ports for development
EXPOSE 8888 8443

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]