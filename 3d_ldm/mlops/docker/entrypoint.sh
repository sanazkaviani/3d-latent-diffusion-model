#!/bin/bash

# Entrypoint script for 3D LDM Docker container
# Supports multiple modes: api, train, inference, jupyter

set -e

# Default values
MODE="${1:-api}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"

# Create necessary directories
mkdir -p /app/logs /app/outputs /app/models

# Function to wait for GPU
wait_for_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo "🔍 Waiting for GPU to be available..."
        while ! nvidia-smi &> /dev/null; do
            echo "GPU not ready, waiting..."
            sleep 5
        done
        echo "✅ GPU is available"
        nvidia-smi
    else
        echo "⚠️  No NVIDIA GPU detected, running on CPU"
    fi
}

# Function to check model files
check_model_files() {
    echo "🔍 Checking for model files..."
    if [ -f "/app/models/3d_ldm_model.pth" ]; then
        echo "✅ Model file found"
    else
        echo "⚠️  No model file found, will use dummy model for testing"
    fi
}

# Function to run health check
health_check() {
    echo "🏥 Running health check..."
    python health_check.py
}

echo "🚀 Starting 3D Latent Diffusion Model container in '$MODE' mode"
echo "🐳 Container: $(hostname)"
echo "📅 Time: $(date)"
echo "👤 User: $(whoami)"
echo "📍 Working Directory: $(pwd)"

# Wait for GPU if available
wait_for_gpu

# Check model files
check_model_files

case "$MODE" in
    "api")
        echo "🌐 Starting API server on port $PORT with $WORKERS workers"
        if [ "$WORKERS" -eq 1 ]; then
            # Single worker mode for development
            exec uvicorn api_server:app \
                --host 0.0.0.0 \
                --port $PORT \
                --log-level info \
                --access-log \
                --reload false
        else
            # Multi-worker mode for production
            exec gunicorn api_server:app \
                --bind 0.0.0.0:$PORT \
                --workers $WORKERS \
                --worker-class uvicorn.workers.UvicornWorker \
                --timeout 300 \
                --keep-alive 5 \
                --max-requests 1000 \
                --max-requests-jitter 100 \
                --log-level info \
                --access-logfile - \
                --error-logfile -
        fi
        ;;
        
    "train")
        echo "🎓 Starting training mode"
        CONFIG_FILE="${CONFIG_FILE:-/app/config/config_train_32g.json}"
        ENV_FILE="${ENV_FILE:-/app/config/environment.json}"
        
        if [ "$TRAIN_TYPE" = "autoencoder" ]; then
            echo "🔧 Training autoencoder..."
            exec python train_autoencoder.py \
                -c "$CONFIG_FILE" \
                -e "$ENV_FILE" \
                -g 1
        elif [ "$TRAIN_TYPE" = "diffusion" ]; then
            echo "🌊 Training diffusion model..."
            exec python train_diffusion.py \
                -c "$CONFIG_FILE" \
                -e "$ENV_FILE" \
                -g 1
        else
            echo "❌ Invalid TRAIN_TYPE. Use 'autoencoder' or 'diffusion'"
            exit 1
        fi
        ;;
        
    "inference")
        echo "🔮 Starting inference mode"
        CONFIG_FILE="${CONFIG_FILE:-/app/config/config_train_32g.json}"
        ENV_FILE="${ENV_FILE:-/app/config/environment.json}"
        NUM_SAMPLES="${NUM_SAMPLES:-5}"
        
        exec python inference.py \
            -c "$CONFIG_FILE" \
            -e "$ENV_FILE" \
            --num "$NUM_SAMPLES"
        ;;
        
    "jupyter")
        echo "📓 Starting Jupyter notebook server"
        exec jupyter notebook \
            --ip=0.0.0.0 \
            --port=8888 \
            --no-browser \
            --allow-root \
            --NotebookApp.token='' \
            --NotebookApp.password='' \
            --NotebookApp.allow_origin='*'
        ;;
        
    "bash")
        echo "🐚 Starting bash shell"
        exec /bin/bash
        ;;
        
    "health")
        health_check
        ;;
        
    *)
        echo "❌ Unknown mode: $MODE"
        echo "Available modes: api, train, inference, jupyter, bash, health"
        exit 1
        ;;
esac