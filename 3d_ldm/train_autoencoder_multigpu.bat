@echo off
REM Multi-GPU Training Script for 3D Autoencoder (Windows)
REM Usage: train_autoencoder_multigpu.bat [num_gpus] [config_file] [amp] [compile]

REM Set default values
if "%1"=="" (set NUM_GPUS=2) else (set NUM_GPUS=%1)
if "%2"=="" (set CONFIG_FILE=.\config\config_train_32g.json) else (set CONFIG_FILE=%2)
set ENV_FILE=.\config\environment.json
if "%3"=="" (set USE_AMP=false) else (set USE_AMP=%3)
if "%4"=="" (set USE_COMPILE=false) else (set USE_COMPILE=%4)

REM Check if CUDA is available
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo Error: nvidia-smi not found. Make sure CUDA is installed.
    exit /b 1
)

REM Set distributed training environment variables
set MASTER_ADDR=localhost
set MASTER_PORT=12355
set WORLD_SIZE=%NUM_GPUS%

echo Starting multi-GPU training with %NUM_GPUS% GPUs
echo Config file: %CONFIG_FILE%
echo Environment file: %ENV_FILE%
echo AMP enabled: %USE_AMP%
echo Compile enabled: %USE_COMPILE%

REM Build the command using torchrun (recommended over torch.distributed.launch)
set CMD=torchrun --nproc_per_node=%NUM_GPUS% --master_port=%MASTER_PORT% train_autoencoder.py
set CMD=%CMD% --environment-file %ENV_FILE%
set CMD=%CMD% --config-file %CONFIG_FILE%
set CMD=%CMD% --gpus %NUM_GPUS%
set CMD=%CMD% --no-images

if "%USE_AMP%"=="true" (
    set CMD=%CMD% --amp
)

if "%USE_COMPILE%"=="true" (
    set CMD=%CMD% --compile
)

echo Executing: %CMD%
echo ============================================

REM Run the training
%CMD%

echo Training completed!
pause
