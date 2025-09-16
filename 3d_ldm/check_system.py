#!/usr/bin/env python3
"""
System Configuration Checker for Multi-GPU 3D Autoencoder Training
This script analyzes your system and provides recommendations for optimal training settings.
"""

import os
import sys
import subprocess
import psutil
import torch
import json
from pathlib import Path

def check_cuda_availability():
    """Check CUDA installation and GPU information"""
    print("=== CUDA and GPU Information ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Multi-GPU training not possible.")
        return False, []
    
    num_gpus = torch.cuda.device_count()
    print(f"✅ CUDA available with {num_gpus} GPU(s)")
    
    gpu_info = []
    total_memory = 0
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        total_memory += memory_gb
        
        gpu_info.append({
            'id': i,
            'name': props.name,
            'memory_gb': memory_gb,
            'compute_capability': f"{props.major}.{props.minor}"
        })
        
        print(f"  GPU {i}: {props.name}")
        print(f"    Memory: {memory_gb:.1f} GB")
        print(f"    Compute Capability: {props.major}.{props.minor}")
        
        # Check if GPU supports mixed precision
        if props.major >= 7:  # Tensor cores available on V100, RTX series, A100, etc.
            print(f"    ✅ Supports mixed precision (AMP)")
        else:
            print(f"    ⚠️  Limited mixed precision support")
    
    print(f"Total GPU Memory: {total_memory:.1f} GB")
    return True, gpu_info

def check_system_memory():
    """Check system RAM"""
    print("\n=== System Memory ===")
    
    memory = psutil.virtual_memory()
    memory_gb = memory.total / 1024**3
    
    print(f"Total RAM: {memory_gb:.1f} GB")
    print(f"Available RAM: {memory.available / 1024**3:.1f} GB")
    
    if memory_gb < 16:
        print("⚠️  Warning: Less than 16GB RAM may limit data loading performance")
    elif memory_gb >= 32:
        print("✅ Sufficient RAM for efficient data loading")
    else:
        print("✅ Adequate RAM for training")
    
    return memory_gb

def check_storage():
    """Check storage type and available space"""
    print("\n=== Storage Information ===")
    
    current_dir = Path.cwd()
    disk_usage = psutil.disk_usage(current_dir)
    
    free_space_gb = disk_usage.free / 1024**3
    print(f"Free space: {free_space_gb:.1f} GB")
    
    # Try to detect SSD vs HDD (Linux/Windows methods)
    storage_type = "Unknown"
    try:
        if sys.platform.startswith('linux'):
            # Linux method to check if SSD
            result = subprocess.run(['lsblk', '-d', '-o', 'name,rota'], 
                                  capture_output=True, text=True)
            if result.returncode == 0 and '0' in result.stdout:
                storage_type = "SSD"
            else:
                storage_type = "HDD"
        elif sys.platform == 'win32':
            # Windows method (basic check)
            # This is a simplified check
            storage_type = "SSD/Unknown"
    except:
        pass
    
    print(f"Storage type: {storage_type}")
    
    if free_space_gb < 50:
        print("⚠️  Warning: Less than 50GB free space may be insufficient for checkpoints and logs")
    else:
        print("✅ Sufficient storage space")
    
    return free_space_gb, storage_type

def check_pytorch_version():
    """Check PyTorch version and features"""
    print("\n=== PyTorch Information ===")
    
    torch_version = torch.__version__
    print(f"PyTorch version: {torch_version}")
    
    # Check for PyTorch 2.0+ features
    major_version = int(torch_version.split('.')[0])
    minor_version = int(torch_version.split('.')[1])
    
    compile_available = hasattr(torch, 'compile') and major_version >= 2
    print(f"torch.compile available: {'✅ Yes' if compile_available else '❌ No (requires PyTorch 2.0+)'}")
    
    # Check for CUDA version compatibility
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        print(f"CUDA version: {cuda_version}")
    
    return torch_version, compile_available

def recommend_settings(gpu_info, memory_gb, compile_available):
    """Generate training recommendations based on system specs"""
    print("\n=== Training Recommendations ===")
    
    num_gpus = len(gpu_info)
    min_gpu_memory = min(gpu['memory_gb'] for gpu in gpu_info) if gpu_info else 0
    
    recommendations = {
        'num_gpus': num_gpus,
        'use_amp': False,
        'use_compile': False,
        'batch_size_per_gpu': 1,
        'num_workers': 4,
        'recommended_patch_size': [64, 64, 64]
    }
    
    # GPU recommendations
    if num_gpus == 0:
        print("❌ No GPUs available - CPU training only (very slow)")
        return recommendations
    elif num_gpus == 1:
        print(f"📱 Single GPU training on {gpu_info[0]['name']}")
        recommendations['num_gpus'] = 1
    else:
        print(f"🚀 Multi-GPU training recommended with {num_gpus} GPUs")
        recommendations['num_gpus'] = min(num_gpus, 8)  # Cap at 8 GPUs for diminishing returns
    
    # Memory-based recommendations
    if min_gpu_memory >= 24:  # RTX 3090/4090, A100, etc.
        recommendations['batch_size_per_gpu'] = 2
        recommendations['use_amp'] = True
        print(f"✅ High memory GPUs ({min_gpu_memory:.0f}GB) - can use larger batch sizes")
    elif min_gpu_memory >= 16:  # RTX 4080, etc.
        recommendations['batch_size_per_gpu'] = 2
        recommendations['use_amp'] = True
        print(f"✅ Good memory GPUs ({min_gpu_memory:.0f}GB) - recommend AMP")
    elif min_gpu_memory >= 8:  # RTX 3070, etc.
        recommendations['batch_size_per_gpu'] = 1
        recommendations['use_amp'] = True
        print(f"⚠️  Moderate memory GPUs ({min_gpu_memory:.0f}GB) - AMP required")
    else:
        recommendations['batch_size_per_gpu'] = 1
        recommendations['use_amp'] = True
        recommendations['recommended_patch_size'] = [48, 48, 48]  # Smaller patches
        print(f"⚠️  Low memory GPUs ({min_gpu_memory:.0f}GB) - use smaller patches and AMP")
    
    # Compile recommendations
    if compile_available:
        recommendations['use_compile'] = True
        print("✅ torch.compile available - recommended for 10-20% speedup")
    else:
        print("❌ torch.compile not available - upgrade to PyTorch 2.0+ for better performance")
    
    # Worker recommendations based on CPU and memory
    cpu_count = os.cpu_count() or 4
    if num_gpus > 1:
        recommendations['num_workers'] = min(8, cpu_count // num_gpus)
    else:
        recommendations['num_workers'] = min(4, cpu_count // 2)
    
    print(f"💻 Recommended data loading workers: {recommendations['num_workers']}")
    
    return recommendations

def generate_config_file(recommendations):
    """Generate optimized configuration file"""
    print("\n=== Generated Configuration ===")
    
    config = {
        "channel": 1,
        "spacing": [5, 3.2, 3.2],
        "spatial_dims": 3,
        "image_channels": 1,
        "latent_channels": 16,
        "autoencoder_def": {
            "_target_": "monai.networks.nets.AutoencoderKL",
            "spatial_dims": "@spatial_dims",
            "in_channels": "$@image_channels",
            "out_channels": "@image_channels",
            "latent_channels": "@latent_channels",
            "channels": [128, 256, 256],
            "num_res_blocks": 2,
            "norm_num_groups": 32,
            "norm_eps": 1e-06,
            "attention_levels": [False, True, True],
            "with_encoder_nonlocal_attn": False,
            "with_decoder_nonlocal_attn": True
        },
        "autoencoder_train": {
            "batch_size": recommendations['batch_size_per_gpu'],
            "patch_size": recommendations['recommended_patch_size'],
            "lr": 1e-3,
            "lr_scheduler": "warmup_cosine",
            "perceptual_weight": 0.001,
            "kl_weight": 1e-7,
            "recon_loss": "l2",
            "max_epochs": 1000,
            "val_interval": 10
        }
    }
    
    config_path = "config/config_optimized.json"
    os.makedirs("config", exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"✅ Optimized config saved to: {config_path}")
    return config_path

def generate_launch_command(recommendations, config_path):
    """Generate the optimal launch command"""
    print("\n=== Recommended Launch Command ===")
    
    if recommendations['num_gpus'] > 1:
        # Multi-GPU command using torchrun (recommended)
        cmd_parts = [
            f"torchrun",
            f"--nproc_per_node={recommendations['num_gpus']}",
            f"--master_port=12355",
            f"train_autoencoder.py",
            f"--config-file {config_path}",
            f"--gpus {recommendations['num_gpus']}"
        ]
        
        if recommendations['use_amp']:
            cmd_parts.append("--amp")
        
        if recommendations['use_compile']:
            cmd_parts.append("--compile")
        
        cmd = " \\\n    ".join(cmd_parts)
        print("Multi-GPU launch (using torchrun):")
        print(f"```bash\n{cmd}\n```")
        
        # Also show legacy torch.distributed.launch version
        legacy_cmd_parts = [
            f"python -m torch.distributed.launch",
            f"--nproc_per_node={recommendations['num_gpus']}",
            f"--master_port=12355",
            f"train_autoencoder.py",
            f"--config-file {config_path}",
            f"--gpus {recommendations['num_gpus']}"
        ]
        
        if recommendations['use_amp']:
            legacy_cmd_parts.append("--amp")
        
        if recommendations['use_compile']:
            legacy_cmd_parts.append("--compile")
        
        legacy_cmd = " \\\n    ".join(legacy_cmd_parts)
        print(f"\nLegacy launch (deprecated):")
        print(f"```bash\n{legacy_cmd}\n```")
        
        # Also show batch file version for Windows
        batch_cmd = f"train_autoencoder_multigpu.bat {recommendations['num_gpus']} {config_path}"
        if recommendations['use_amp']:
            batch_cmd += " true"
        else:
            batch_cmd += " false"
        
        if recommendations['use_compile']:
            batch_cmd += " true"
        else:
            batch_cmd += " false"
        
        print(f"\nWindows batch file:")
        print(f"```cmd\n{batch_cmd}\n```")
    
    else:
        # Single GPU command
        cmd_parts = [
            f"python train_autoencoder.py",
            f"--config-file {config_path}",
            f"--gpus 1"
        ]
        
        if recommendations['use_amp']:
            cmd_parts.append("--amp")
        
        if recommendations['use_compile']:
            cmd_parts.append("--compile")
        
        cmd = " ".join(cmd_parts)
        print("Single-GPU launch:")
        print(f"```bash\n{cmd}\n```")

def main():
    """Main function to run all checks and generate recommendations"""
    print("🔍 3D Autoencoder Training System Analysis")
    print("=" * 50)
    
    # System checks
    cuda_available, gpu_info = check_cuda_availability()
    memory_gb = check_system_memory()
    free_space, storage_type = check_storage()
    torch_version, compile_available = check_pytorch_version()
    
    if not cuda_available:
        print("\n❌ Cannot proceed with GPU training. Please install CUDA and PyTorch with GPU support.")
        return
    
    # Generate recommendations
    recommendations = recommend_settings(gpu_info, memory_gb, compile_available)
    config_path = generate_config_file(recommendations)
    generate_launch_command(recommendations, config_path)
    
    # Performance estimates
    print("\n=== Performance Estimates ===")
    estimated_speedup = min(recommendations['num_gpus'] * 0.85, recommendations['num_gpus'])
    print(f"Expected speedup vs single GPU: {estimated_speedup:.1f}x")
    
    if recommendations['use_amp']:
        print("Expected memory savings with AMP: ~40-50%")
        print("Expected additional speedup with AMP: ~1.5-2x")
    
    if recommendations['use_compile']:
        print("Expected additional speedup with torch.compile: ~10-20%")
    
    print("\n✅ Analysis complete! Use the generated configuration and commands above.")

if __name__ == "__main__":
    main()
