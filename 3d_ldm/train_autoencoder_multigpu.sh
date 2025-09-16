#!/bin/bash
#SBATCH --job-name=DDPM_finetune
#SBATCH --partition=batch
#SBATCH --nodelist=ohi-hpc2-mesana02
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=/q/RUBY-AI/Denoising/codes/my_codes/3d_ldm/slm_output/simclr_%j.out
#SBATCH --error=/q/RUBY-AI/Denoising/codes/my_codes/3d_ldm/slm_output/simclr_%j.err
#SBATCH --exclusive

# Multi-GPU Training Script for 3D Autoencoder
# Usage: ./train_autoencoder_multigpu.sh [num_gpus] [config_file] [amp] [compile]

# Set default values
NUM_GPUS=${1:-2}
CONFIG_FILE=${2:-"./config/config_train_32g.json"}
ENV_FILE="./config/environment.json"
USE_AMP=${3:-false}
USE_COMPILE=${4:-false}

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

# Set distributed training environment variables
export MASTER_ADDR="localhost"
export MASTER_PORT="12355"
export WORLD_SIZE=$NUM_GPUS

echo "Starting multi-GPU training with $NUM_GPUS GPUs"
echo "Config file: $CONFIG_FILE"
echo "Environment file: $ENV_FILE"
echo "AMP enabled: $USE_AMP"
echo "Compile enabled: $USE_COMPILE"
conda activate py39
# Build the command using torchrun (recommended over torch.distributed.launch)
CMD="torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train_autoencoder.py"
CMD="$CMD --environment-file $ENV_FILE"
CMD="$CMD --config-file $CONFIG_FILE"
CMD="$CMD --gpus $NUM_GPUS"

if [ "$USE_AMP" = "true" ]; then
    CMD="$CMD --amp"
fi

if [ "$USE_COMPILE" = "true" ]; then
    CMD="$CMD --compile"
fi

echo "Executing: $CMD"
echo "============================================"

# Run the training
eval $CMD

echo "Training completed!"
