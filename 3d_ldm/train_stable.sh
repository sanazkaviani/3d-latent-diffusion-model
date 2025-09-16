#!/bin/bash

# Stable Multi-GPU Training Script for 3D Autoencoder
# This script includes stability fixes for NaN losses

set -e  # Exit on any error

# Set default values with more conservative settings
NUM_GPUS=${1:-3}
CONFIG_FILE=${2:-"./config/config_train_stable.json"}
ENV_FILE="./config/environment.json"
USE_AMP=${3:-true}
USE_COMPILE=${4:-false}

echo "=== Stable Multi-GPU Training ==="
echo "Number of GPUs: $NUM_GPUS"
echo "Config file: $CONFIG_FILE"
echo "AMP enabled: $USE_AMP"
echo "Compile enabled: $USE_COMPILE"

# Activate conda environment
conda activate py39

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Make sure CUDA is installed."
    exit 1
fi

# Check available GPUs
AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)
echo "Available GPUs: $AVAILABLE_GPUS"

if [ $NUM_GPUS -gt $AVAILABLE_GPUS ]; then
    echo "Error: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available."
    exit 1
fi

# Set conservative environment variables for stability
export MASTER_ADDR="localhost"
export MASTER_PORT="12355"
export WORLD_SIZE=$NUM_GPUS
export OMP_NUM_THREADS=8
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Additional stability settings
export CUDA_LAUNCH_BLOCKING=1  # For debugging
export TORCH_USE_CUDA_DSA=1    # For better error reporting

echo "Environment variables set for stability"

# Build the command using torchrun
CMD="torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train_autoencoder.py"
CMD="$CMD --environment-file $ENV_FILE"
CMD="$CMD --config-file $CONFIG_FILE"
CMD="$CMD --gpus $NUM_GPUS"
CMD="$CMD --no-images"  # Disable image logging for stability

if [ "$USE_AMP" = "true" ]; then
    CMD="$CMD --amp"
fi

if [ "$USE_COMPILE" = "true" ]; then
    CMD="$CMD --compile"
fi

echo "Executing: $CMD"
echo "============================================"

# Run the training with error handling
if eval $CMD; then
    echo "Training completed successfully!"
else
    echo "Training failed. Check the error logs above."
    echo "Try using the stable config: config/config_train_stable.json"
    echo "Or reduce learning rate and increase warmup epochs."
    exit 1
fi
