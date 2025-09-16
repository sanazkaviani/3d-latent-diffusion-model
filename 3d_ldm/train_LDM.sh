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


#ohi-hpc2-keon01  mesana02
export PATH=$HOME/miniconda/bin:$PATH
eval "$(conda shell.bash hook)"
conda activate py39 


# Print SLURM and environment info
echo "SLURM_JOB_ID      : $SLURM_JOB_ID"
echo "Node list         : $SLURM_NODELIST"
echo "GPUs allocated    : $SLURM_GPUS"
echo "Working directory : $SLURM_SUBMIT_DIR"

# Kill any previous torchrun/train.py processes (cleanup)
pkill -u $USER -f torchrun
pkill -u $USER -f train_LDM.py

export MASTER_ADDR=$(hostname)
#export MASTER_PORT=$(shuf -i 20000-40000 -n 1)
export MASTER_PORT=29500
echo "Using MASTER_PORT: $MASTER_PORT"
lsof -i :29500 || echo "Port 29500 is free"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK # 32


#export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Set your data directory and result folder
DATA_DIR="/q/RUBY-AI/Denoising/Images/Dataset/npz_dataset_50/train"
VAL_DIR="/q/RUBY-AI/Denoising/Images/Dataset/npz_dataset_50/val"
#DATA_DIR="/q/RUBY-AI/Denoising/
RESULT_FOLDER="./checkpoints"
# Set training parameters

SAMPLE_FLAGS="--batch_size 1"
# MODEL_FLAGS="--attention_resolutions 1000 --large_size 64 --small_size 64\
#         --num_channels 128 --use_fp16 True --num_head_channels 64 --learn_sigma True\
#         --resblock_updown True --use_scale_shift_norm True --class_cond True"
# DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear \
#                 --rescale_learned_sigmas True --rescale_timesteps True"

# SCHEDULE_SAMPLER="uniform"
LR=1e-4
WEIGHT_DECAY=0.001
LR_ANNEAL_STEPS=0

LOG_INTERVAL=1
SAVE_INTERVAL=1
# RESUME_CHECKPOINT="./checkpoint_finetune_cond/model040000.pt"
SCRIPT="/q/RUBY-AI/Denoising/codes/my_codes/3d_ldm/train_autoencoder.py"
#SCRIPT="/q/RUBY-AI/Denoising/codes/my_codes/3d_ldm/train_diffusion.py"
ENV="/q/RUBY-AI/Denoising/codes/my_codes/3d_ldm/config/environment.json"
export PYTHONPATH=/q/RUBY-AI/Denoising/codes/my_codes/3d_ldm:$PYTHONPATH

srun --ntasks=1 --gpus-per-task=3 \
    torchrun --nproc_per_node=3 --nnodes=1 --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    $SCRIPT \
    -g 3 \
    -e "$ENV" \
    --amp

