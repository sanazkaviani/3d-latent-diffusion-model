# Multi-GPU Training Troubleshooting Guide

## Common Errors and Solutions

### Error 1: `unrecognized arguments: --local_rank=0`

#### Problem Description
When launching multi-GPU training with `torch.distributed.launch`, you may encounter:
```
train_autoencoder.py: error: unrecognized arguments: --local_rank=0
```

#### Root Cause
The `torch.distributed.launch` utility automatically passes `--local_rank` arguments to your script, but the script's argument parser doesn't recognize this argument.

#### Solutions (Choose One)

**Solution 1: Use torchrun (Recommended)**
Replace `torch.distributed.launch` with `torchrun`:

**Before (deprecated):**
```bash
python -m torch.distributed.launch --nproc_per_node=2 train_autoencoder.py --gpus 2
```

**After (recommended):**
```bash
torchrun --nproc_per_node=2 train_autoencoder.py --gpus 2
```

### Error 2: `TypeError: __init__() got an unexpected keyword argument 'fused'`

#### Problem Description
```
TypeError: __init__() got an unexpected keyword argument 'fused'
```

#### Root Cause
The `fused=True` parameter for optimizers was introduced in PyTorch 2.0+. If you're using an older version, this parameter is not supported.

#### Solution
The script now automatically detects PyTorch version and only uses `fused=True` if supported. Update to the latest train_autoencoder.py script.

### Error 3: `Losses become NaN` - Training Instability

#### Problem Description
```
Epoch 0/1000 (242.85s) - Recon: nan, KL: 299287707648.0000, Perceptual: nan
```

#### Root Cause
Training instability causing loss explosion, typically due to:
- Learning rate too high for multi-GPU setup
- Gradient explosion
- Numerical instability in loss calculations
- Data preprocessing issues

#### Solutions

**Immediate Fix:**
Use the stable configuration file:
```bash
# Use the more conservative configuration
./train_autoencoder_multigpu.sh 3 config/config_train_stable.json true false
```

**Manual Configuration Adjustments:**
1. **Reduce Learning Rate:** From `1e-3` to `1e-4` or `5e-4`
2. **Lower KL Weight:** From `1e-7` to `1e-8` or `1e-9` 
3. **Reduce Perceptual Weight:** From `0.001` to `0.0001` or `0.00001`
4. **Use L1 Loss:** Change from `"l2"` to `"l1"` for more stability
5. **Smaller Model:** Reduce channels and use fewer attention layers

**Updated Configuration Example:**
```json
{
    "autoencoder_train": {
        "batch_size": 1,
        "lr": 1e-4,
        "perceptual_weight": 0.00001,
        "kl_weight": 1e-9,
        "recon_loss": "l1"
    }
}
```

### Error 6: `TypeError: Cannot handle this data type` - TensorBoard Image Logging

#### Problem Description
```
TypeError: Cannot handle this data type: (1, 1, 96), |u1
```

#### Root Cause
TensorBoard/PIL cannot handle the specific tensor format being passed for image logging. This occurs when the image tensor has an incompatible shape or data type.

#### Solutions

**Quick Fix - Disable Image Logging:**
```bash
# Add --no-images flag to disable image logging
torchrun --nproc_per_node=3 train_autoencoder.py --config-file config/config_train_stable.json --gpus 3 --amp --no-images
```

**Permanent Fix:**
The script now includes:
- ✅ Robust error handling for image logging
- ✅ Proper tensor format conversion (C, H, W) for TensorBoard
- ✅ Safe fallback when image logging fails
- ✅ Option to disable image logging completely

**Updated Image Function:**
The `visualize_one_slice_in_3d_image` function now properly converts tensors to the correct format for TensorBoard.

### Error 5: `RuntimeError: grad can be implicitly created only for scalar outputs`

#### Problem Description
```
RuntimeError: grad can be implicitly created only for scalar outputs
```

#### Root Cause
This error occurs when trying to call `.backward()` on a tensor that is not a scalar. PyTorch requires loss tensors to be scalars (single values) for backpropagation.

#### Solution
The script now ensures all losses are properly reduced to scalars:
```python
# Ensure all losses are scalars
if recons_loss.numel() > 1:
    recons_loss = recons_loss.mean()
if loss_g.numel() > 1:
    loss_g = loss_g.mean()
```

#### Prevention
Always ensure loss functions return scalar values before calling `.backward()`.

### Error 4: `Boolean value of Tensor with more than one value is ambiguous`

#### Problem Description
```
RuntimeError: Boolean value of Tensor with more than one value is ambiguous
```

#### Root Cause
This occurs when trying to use `torch.isnan(tensor)` directly in an `if` statement when the tensor has multiple values. PyTorch doesn't know whether you want to check if ANY or ALL values are NaN.

#### Solution
The script now uses safe NaN checking:
```python
# Fixed: Use .any().item() for multi-element tensors
recons_nan = torch.isnan(recons_loss).any().item() if recons_loss.numel() > 1 else torch.isnan(recons_loss).item()
```

#### Prevention
Always use `.any().item()` or `.all().item()` when checking conditions on tensors that might have multiple values.

### Fixed Launch Commands

#### Linux/Unix:
```bash
# Using torchrun (recommended)
torchrun --nproc_per_node=2 --master_port=12355 train_autoencoder.py \
    --config-file config/config_train_multigpu.json \
    --gpus 2 --amp

# Using the provided script
./train_autoencoder_multigpu.sh 2 config/config_train_multigpu.json true false
```

#### Windows:
```cmd
REM Using torchrun (recommended)
torchrun --nproc_per_node=2 --master_port=12355 train_autoencoder.py ^
    --config-file config/config_train_multigpu.json ^
    --gpus 2 --amp

REM Using the provided batch file
train_autoencoder_multigpu.bat 2 config/config_train_multigpu.json true false
```

### Environment Variables
Make sure these are set for distributed training:
```bash
export MASTER_ADDR="localhost"
export MASTER_PORT="12355"
export WORLD_SIZE=2  # Number of GPUs
```

### Additional Troubleshooting

#### Check PyTorch Version
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

#### Verify CUDA Setup
```bash
nvidia-smi
nvcc --version
```

#### Test Single GPU First
Before attempting multi-GPU training, ensure single GPU training works:
```bash
python train_autoencoder.py --config-file config/config_train_32g.json --gpus 1
```

### Updated Script Features
The optimized script now includes:
- ✅ `--local_rank` argument support for backward compatibility
- ✅ Updated launch scripts using `torchrun`
- ✅ Enhanced error handling and warnings
- ✅ Automatic mixed precision support with `--amp`
- ✅ PyTorch 2.0+ optimizations with `--compile`

### Performance Monitoring
Monitor your training with:
```bash
# GPU utilization
watch -n 1 nvidia-smi

# TensorBoard
tensorboard --logdir runs/autoencoder

# System resources
htop
```

This should resolve the `--local_rank` argument error and provide better multi-GPU training performance.
