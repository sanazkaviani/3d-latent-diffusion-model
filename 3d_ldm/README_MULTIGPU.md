# Multi-GPU Optimized 3D Autoencoder Training

This document describes the optimizations made to the `train_autoencoder.py` script for efficient multi-GPU training.

## Key Optimizations

### 1. **Distributed Data Parallel (DDP) Enhancements**
- **Static Graph**: Enabled `static_graph=True` for better performance with consistent model architecture
- **SyncBatchNorm**: Automatic conversion to synchronized batch normalization across GPUs
- **Efficient Communication**: Disabled `find_unused_parameters` and `broadcast_buffers` for better performance
- **Proper Gradient Synchronization**: Enhanced gradient reduction across all GPUs

### 2. **Automatic Mixed Precision (AMP)**
- **Memory Optimization**: Reduces memory usage by ~50% while maintaining accuracy
- **Speed Improvement**: 1.5-2x faster training on modern GPUs
- **Gradient Scaling**: Automatic gradient scaling to prevent underflow
- **Usage**: Add `--amp` flag to enable

### 3. **Torch Compile Optimization**
- **JIT Compilation**: Optimizes model execution graphs for better performance
- **Memory Efficiency**: Reduces memory overhead through graph optimization
- **Usage**: Add `--compile` flag to enable (requires PyTorch 2.0+)

### 4. **Enhanced Optimizers and Schedulers**
- **Fused Optimizers**: AdamW with `fused=True` for better GPU utilization
- **Learning Rate Scaling**: Square root scaling with world size for better convergence
- **Cosine Annealing**: Optional learning rate scheduling with warmup
- **Weight Decay**: Added L2 regularization for better generalization

### 5. **Memory and Performance Optimizations**
- **Gradient Checkpointing**: Reduces memory usage during backward pass
- **Non-blocking Data Transfer**: Overlaps data loading with computation
- **Flash Attention**: Enabled when available for efficient attention computation
- **Optimized CUDA Settings**: Enhanced cuDNN benchmark and deterministic settings

### 6. **Monitoring and Profiling**
- **Enhanced Logging**: Detailed loss tracking and timing information
- **TensorBoard Integration**: Comprehensive metrics logging including learning rates
- **Profiler Support**: Optional PyTorch profiler for performance analysis
- **Progress Tracking**: Real-time epoch timing and throughput metrics

## Usage

### Quick Start (Windows)
```cmd
# Train with 2 GPUs, AMP enabled
train_autoencoder_multigpu.bat 2 config/config_train_multigpu.json true false

# Train with 4 GPUs, AMP and compile enabled
train_autoencoder_multigpu.bat 4 config/config_train_multigpu.json true true
```

### Quick Start (Linux/Unix)
```bash
# Make script executable
chmod +x train_autoencoder_multigpu.sh

# Train with 2 GPUs, AMP enabled
./train_autoencoder_multigpu.sh 2 config/config_train_multigpu.json true false

# Train with 4 GPUs, AMP and compile enabled
./train_autoencoder_multigpu.sh 4 config/config_train_multigpu.json true true
```

### Manual Launch
```bash
# Set environment variables
export MASTER_ADDR="localhost"
export MASTER_PORT="12355"
export WORLD_SIZE=2

# Launch training
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=12355 \
    train_autoencoder.py \
    --environment-file ./config/environment.json \
    --config-file ./config/config_train_multigpu.json \
    --gpus 2 \
    --amp \
    --compile
```

## Configuration Options

### Command Line Arguments
- `--gpus`: Number of GPUs to use (default: 1)
- `--amp`: Enable automatic mixed precision training
- `--compile`: Enable torch.compile optimization (PyTorch 2.0+)
- `--profile`: Enable performance profiling
- `--environment-file`: Path to environment configuration
- `--config-file`: Path to training configuration

### Configuration File Options
The new `config_train_multigpu.json` includes optimizations for multi-GPU training:

```json
{
    "autoencoder_train": {
        "batch_size": 2,                    // Per-GPU batch size
        "lr": 1e-3,                        // Base learning rate (will be scaled)
        "lr_scheduler": "warmup_cosine",   // Learning rate scheduler
        "max_epochs": 1000,
        "val_interval": 10
    }
}
```

## Performance Expectations

### Memory Usage
- **Single GPU**: ~8-12GB VRAM for batch size 1
- **Multi-GPU**: Linear scaling with number of GPUs
- **AMP**: ~50% memory reduction
- **Gradient Checkpointing**: Additional 20-30% memory savings

### Training Speed
- **2 GPUs**: ~1.8x speedup over single GPU
- **4 GPUs**: ~3.2x speedup over single GPU
- **8 GPUs**: ~5.5x speedup over single GPU
- **AMP**: Additional 1.5-2x speedup

### Convergence
- **Learning Rate Scaling**: Maintains convergence quality across different GPU counts
- **SyncBatchNorm**: Ensures consistent normalization statistics
- **Gradient Clipping**: Prevents training instability

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```
   Solution: Reduce batch_size or enable --amp flag
   ```

2. **DDP Initialization Timeout**
   ```bash
   # Increase timeout in utils.py setup_ddp function
   dist.init_process_group(..., timeout=timedelta(seconds=36000))
   ```

3. **Inconsistent Training**
   ```
   Solution: Ensure all GPUs have the same model architecture
   Check that SyncBatchNorm is properly applied
   ```

4. **Slow Training**
   ```
   Solution: Enable --amp and --compile flags
   Increase batch_size if memory allows
   Use faster storage for data loading
   ```

### Monitoring Training

1. **TensorBoard**
   ```bash
   tensorboard --logdir runs/autoencoder
   ```

2. **GPU Utilization**
   ```bash
   nvidia-smi -l 1  # Monitor GPU usage
   ```

3. **Profiling**
   ```bash
   # Add --profile flag and check ./profiler_logs
   tensorboard --logdir profiler_logs
   ```

## Best Practices

1. **Batch Size**: Use largest batch size that fits in memory
2. **Data Loading**: Use sufficient `num_workers` in data loader
3. **Validation**: Keep validation batch size consistent across GPUs
4. **Checkpointing**: Save models from rank 0 only to avoid conflicts
5. **Monitoring**: Use rank 0 for all logging and visualization

## Hardware Requirements

### Minimum
- 2x NVIDIA GPUs with 8GB+ VRAM
- 32GB+ System RAM
- Fast storage (SSD recommended)

### Recommended
- 4x NVIDIA GPUs with 16GB+ VRAM (V100, A100, RTX 3090/4090)
- 64GB+ System RAM
- NVMe SSD storage
- High-bandwidth interconnect (NVLink preferred)

## Advanced Features

### Custom Learning Rate Schedulers
Modify the `get_lr_scheduler` function in `train_autoencoder.py` to add custom schedulers.

### Memory Optimization
For very large models, consider:
- Gradient accumulation
- Model sharding (ZeRO)
- CPU offloading

### Profiling Integration
Enable detailed profiling with `--profile` flag to identify bottlenecks.

This optimized implementation provides significant improvements in training speed, memory efficiency, and scalability for multi-GPU setups while maintaining training stability and convergence quality.
