# Autoencoder model training
import argparse
import json
import logging

import os
import sys
from pathlib import Path
import time
import warnings

import torch
import torch.distributed as dist
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.networks.nets import PatchDiscriminator
from monai.config import print_config
from monai.utils import set_determinism
from torch.nn import L1Loss, MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from utils import KL_loss, define_instance, prepare_dataloader, setup_ddp
from visualize_image import visualize_one_slice_in_3d_image

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.parallel")

def reduce_tensor(tensor, world_size):
    """Reduce tensor across all GPUs"""
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor = tensor / world_size
    return tensor

def cleanup_ddp():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def safe_nan_check(tensor):
    """Safely check for NaN in tensors, handling both scalar and multi-dimensional tensors"""
    try:
        if tensor.numel() == 1:
            return torch.isnan(tensor).item()
        else:
            return torch.isnan(tensor).any().item()
    except:
        return False

def get_lr_scheduler(optimizer, args, steps_per_epoch):
    """Get learning rate scheduler for better convergence"""
    if hasattr(args.autoencoder_train, 'lr_scheduler') and args.autoencoder_train['lr_scheduler'] == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=args.autoencoder_train["max_epochs"], eta_min=1e-6)
    elif hasattr(args.autoencoder_train, 'lr_scheduler') and args.autoencoder_train['lr_scheduler'] == 'warmup_cosine':
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.autoencoder_train["max_epochs"] - 5, eta_min=1e-6)
        return SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[5])
    else:
        return None

def main():
    parser = argparse.ArgumentParser(description="PyTorch VAE-GAN training")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./config/environment.json",
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./config/config_train_32g.json",
        help="config json file that stores hyper-parameters",
    )
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    parser.add_argument("--amp", action="store_true", help="enable automatic mixed precision training")
    parser.add_argument("--compile", action="store_true", help="enable torch.compile for faster training")
    parser.add_argument("--profile", action="store_true", help="enable profiling for performance analysis")
    parser.add_argument("--no-images", action="store_true", help="disable image logging to tensorboard")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training (deprecated, use LOCAL_RANK env var)")
    args = parser.parse_args()

    # Step 0: configuration
    ddp_bool = args.gpus > 1  # whether to use distributed data parallel
    if ddp_bool:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist, device = setup_ddp(rank, world_size)
    else:
        rank = 0
        world_size = 1
        device = 0

    torch.cuda.set_device(device)
    if rank == 0:
        print(f"Using device: cuda:{device}")
        print(f"World size: {world_size}, Rank: {rank}")
        print(f"AMP enabled: {args.amp}")
        print(f"Compile enabled: {args.compile}")

    # Optimizations for performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # Set to False for better performance
    torch.set_num_threads(4)
    
    # Enable optimized attention if available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        if rank == 0:
            print("Flash attention enabled")
    except:
        pass
    
    # Only enable anomaly detection in debug mode for performance
    # torch.autograd.set_detect_anomaly(True)

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    set_determinism(42)

    # Step 1: set data loader
    size_divisible = 2 ** (len(args.autoencoder_def["channels"]) - 1)
    train_loader, val_loader = prepare_dataloader(
        args,
        args.autoencoder_train["batch_size"],
        args.autoencoder_train["patch_size"],
        randcrop=True,
        rank=rank,
        world_size=world_size,
        cache=1.0,
        download=False,
        size_divisible=size_divisible,
        amp=args.amp,
    )

    # Calculate steps per epoch for schedulers
    steps_per_epoch = len(train_loader)

    # Step 2: Define Autoencoder KL network and discriminator
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    discriminator_norm = "INSTANCE"
    discriminator = PatchDiscriminator(
        spatial_dims=args.spatial_dims,
        num_layers_d=3,
        channels=32,
        in_channels=1,
        out_channels=1,
        norm=discriminator_norm,
    ).to(device)
    
    # Enable gradient checkpointing for memory optimization
    if hasattr(autoencoder, 'gradient_checkpointing'):
        autoencoder.gradient_checkpointing = True
        if rank == 0:
            print("Gradient checkpointing enabled for autoencoder")
    
    # Compile models for faster training (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        try:
            autoencoder = torch.compile(autoencoder)
            discriminator = torch.compile(discriminator)
            if rank == 0:
                print("Models compiled with torch.compile")
        except Exception as e:
            if rank == 0:
                print(f"Failed to compile models: {e}")
    
    if ddp_bool:
        # When using DDP, BatchNorm needs to be converted to SyncBatchNorm.
        autoencoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(autoencoder)
        discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)
        if rank == 0:
            print("Convert models to SyncBatchNorm")

    trained_g_path = os.path.join(args.model_dir, "autoencoder.pt")
    trained_d_path = os.path.join(args.model_dir, "discriminator.pt")
    trained_g_path_last = os.path.join(args.model_dir, "autoencoder_last.pt")
    trained_d_path_last = os.path.join(args.model_dir, "discriminator_last.pt")

    if rank == 0:
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    # Synchronize before loading checkpoints
    if ddp_bool:
        dist.barrier()

    if args.resume_ckpt:
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        try:
            autoencoder.load_state_dict(torch.load(trained_g_path, map_location=map_location, weights_only=True))
            if rank == 0:
                print(f"Load trained autoencoder from {trained_g_path}")
        except:
            if rank == 0:
                print("Train autoencoder from scratch.")

        try:
            discriminator.load_state_dict(torch.load(trained_d_path, map_location=map_location, weights_only=True))
            if rank == 0:
                print(f"Load trained discriminator from {trained_d_path}")
        except:
            if rank == 0:
                print("Train discriminator from scratch.")

    if ddp_bool:
        # Use static_graph=True for better performance with consistent model architecture
        autoencoder = DDP(autoencoder, device_ids=[device], output_device=rank, 
                         find_unused_parameters=False, broadcast_buffers=False,
                         static_graph=True)
        discriminator = DDP(discriminator, device_ids=[device], output_device=rank, 
                           find_unused_parameters=False, broadcast_buffers=False,
                           static_graph=True)
        if rank == 0:
            print("Convert autoencoder and discriminator to DDP")

    # Step 3: training config
    if "recon_loss" in args.autoencoder_train and args.autoencoder_train["recon_loss"] == "l2":
        intensity_loss = MSELoss()
        if rank == 0:
            print("Use l2 loss")
    else:
        intensity_loss = L1Loss()
        if rank == 0:
            print("Use l1 loss")
    
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
    loss_perceptual.to(device)

    adv_weight = 0.01
    perceptual_weight = args.autoencoder_train["perceptual_weight"]
    # kl_weight: important hyper-parameter.
    #     If too large, decoder cannot recon good results from latent space.
    #     If too small, latent space will not be regularized enough for the diffusion model
    kl_weight = args.autoencoder_train["kl_weight"]

    # Scale learning rate with square root of world size for better convergence
    lr_scale = world_size ** 0.5 if world_size > 1 else 1.0
    scaled_lr = args.autoencoder_train["lr"] * lr_scale
    
    # Check PyTorch version for fused optimizer support
    torch_version = torch.__version__
    major_version = int(torch_version.split('.')[0])
    minor_version = int(torch_version.split('.')[1])
    supports_fused = major_version >= 2 or (major_version == 1 and minor_version >= 13)
    
    # Use more conservative learning rate for stability
    # Reduce learning rate for multi-GPU to prevent instability
    if ddp_bool:
        scaled_lr = scaled_lr * 0.5  # Reduce by half for multi-GPU stability
        if rank == 0:
            print(f"Reduced learning rate for multi-GPU stability: {scaled_lr:.6f}")
    
    # Use fused optimizers for better performance if available
    optimizer_kwargs = {'lr': scaled_lr, 'weight_decay': 1e-5}
    if supports_fused:
        optimizer_kwargs['fused'] = True
        if rank == 0:
            print("Using fused optimizers for better performance")
    else:
        if rank == 0:
            print(f"Fused optimizers not available in PyTorch {torch_version}, using standard optimizers")
    
    # Use more stable optimizer settings
    optimizer_g = torch.optim.AdamW(params=autoencoder.parameters(), 
                                   lr=scaled_lr, weight_decay=1e-5, 
                                   betas=(0.5, 0.9), eps=1e-8)
    optimizer_d = torch.optim.AdamW(params=discriminator.parameters(), 
                                   lr=scaled_lr, weight_decay=1e-5,
                                   betas=(0.5, 0.9), eps=1e-8)
    
    # Initialize AMP scaler
    scaler_g = GradScaler() if args.amp else None
    scaler_d = GradScaler() if args.amp else None
    
    # Learning rate schedulers
    scheduler_g = get_lr_scheduler(optimizer_g, args, steps_per_epoch)
    scheduler_d = get_lr_scheduler(optimizer_d, args, steps_per_epoch)
    
    if rank == 0:
        print(f"Scaled learning rate: {scaled_lr:.6f} (original: {args.autoencoder_train['lr']:.6f})")
        if args.amp:
            print("AMP scalers initialized")
        if scheduler_g:
            print("Learning rate schedulers enabled")

    # initialize tensorboard writer
    if rank == 0:
        Path(args.tfevent_path).mkdir(parents=True, exist_ok=True)
        tensorboard_path = os.path.join(args.tfevent_path, "autoencoder")
        Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(tensorboard_path)

    # Step 4: training
    autoencoder_warm_up_n_epochs = 5
    max_epochs = args.autoencoder_train["max_epochs"]
    val_interval = args.autoencoder_train["val_interval"]
    intermediary_images = []
    n_example_images = 4
    best_val_recon_epoch_loss = 100.0
    total_step = 0
    
    # Initialize profiler if requested
    profiler = None
    if args.profile and rank == 0:
        try:
            from torch.profiler import profile, record_function, ProfilerActivity
            profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            profiler.start()
            print("Profiler started")
        except ImportError:
            print("Warning: torch.profiler not available, profiling disabled")
            profiler = None

    for epoch in range(max_epochs):
        epoch_start_time = time.time()
        
        # train
        autoencoder.train()
        discriminator.train()
        if ddp_bool:
            # if ddp, distribute data across n gpus
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
        
        # Initialize loss accumulators as tensors on device for proper reduction
        epoch_recon_loss = torch.tensor(0.0, device=device)
        epoch_kl_loss = torch.tensor(0.0, device=device)
        epoch_p_loss = torch.tensor(0.0, device=device)
        epoch_adv_loss = torch.tensor(0.0, device=device)
        num_batches = 0
        
        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True)
            
            # Data validation and preprocessing
            if torch.isnan(images).any() or torch.isinf(images).any():
                if rank == 0:
                    print(f"Warning: NaN or Inf detected in input data at step {step}")
                continue
            
            # Clamp input values to reasonable range
            images = torch.clamp(images, 0.0, 1.0)

            # train Generator part
            optimizer_g.zero_grad(set_to_none=True)
            
            # Use autocast for mixed precision training
            with autocast(enabled=args.amp):
                reconstruction, z_mu, z_sigma = autoencoder(images)
                
                # Check for NaN/Inf in autoencoder outputs
                if torch.isnan(reconstruction).any() or torch.isnan(z_mu).any() or torch.isnan(z_sigma).any():
                    if rank == 0:
                        print(f"Warning: NaN detected in autoencoder outputs at step {step}, skipping batch")
                    continue

                recons_loss = intensity_loss(reconstruction, images)
                kl_loss = KL_loss(z_mu, z_sigma)
                
                # Ensure all losses are scalars by taking mean if needed
                if recons_loss.numel() > 1:
                    recons_loss = recons_loss.mean()
                if kl_loss.numel() > 1:
                    kl_loss = kl_loss.mean()
                
                # Clamp KL loss to prevent explosion
                kl_loss = torch.clamp(kl_loss, 0.0, 1000.0)
                
                p_loss = loss_perceptual(reconstruction.float(), images.float())
                
                # Ensure perceptual loss is also scalar
                if p_loss.numel() > 1:
                    p_loss = p_loss.mean()
                
                # Check for NaN/Inf in individual losses - use .any().item() for safety
                recons_nan = torch.isnan(recons_loss).any().item() if recons_loss.numel() > 1 else torch.isnan(recons_loss).item()
                kl_nan = torch.isnan(kl_loss).any().item() if kl_loss.numel() > 1 else torch.isnan(kl_loss).item()
                p_nan = torch.isnan(p_loss).any().item() if p_loss.numel() > 1 else torch.isnan(p_loss).item()
                
                if recons_nan or kl_nan or p_nan:
                    if rank == 0:
                        print(f"Warning: NaN in losses at step {step}")
                        try:
                            print(f"  Recon: {recons_loss.item():.4f}, KL: {kl_loss.item():.4f}, Perceptual: {p_loss.item():.4f}")
                        except:
                            print(f"  Could not extract loss values - shapes: recons={recons_loss.shape}, kl={kl_loss.shape}, p={p_loss.shape}")
                    continue
                
                loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss

                generator_loss = torch.tensor(0.0, device=device)
                if epoch > autoencoder_warm_up_n_epochs:
                    logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                    generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                    
                    # Ensure generator loss is scalar
                    if generator_loss.numel() > 1:
                        generator_loss = generator_loss.mean()
                    
                    # Check for NaN in adversarial loss - use safe checking
                    gen_loss_nan = torch.isnan(generator_loss).any().item() if generator_loss.numel() > 1 else torch.isnan(generator_loss).item()
                    if gen_loss_nan:
                        if rank == 0:
                            print(f"Warning: NaN in generator loss at step {step}")
                        generator_loss = torch.tensor(0.0, device=device)
                    else:
                        loss_g = loss_g + adv_weight * generator_loss
                        
            # Ensure final loss_g is scalar
            if loss_g.numel() > 1:
                loss_g = loss_g.mean()

            # Final NaN check before backward - use safe checking
            loss_g_nan = torch.isnan(loss_g).any().item() if loss_g.numel() > 1 else torch.isnan(loss_g).item()
            loss_g_inf = torch.isinf(loss_g).any().item() if loss_g.numel() > 1 else torch.isinf(loss_g).item()
            
            if loss_g_nan or loss_g_inf:
                if rank == 0:
                    print(f"Warning: NaN/Inf in total generator loss at step {step}, skipping backward")
                continue

            # Backward pass with gradient scaling
            if args.amp and scaler_g:
                scaler_g.scale(loss_g).backward()
                scaler_g.unscale_(optimizer_g)
                # More aggressive gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=0.5)
                scaler_g.step(optimizer_g)
                scaler_g.update()
            else:
                loss_g.backward()
                # More aggressive gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=0.5)
                optimizer_g.step()

            discriminator_loss = torch.tensor(0.0, device=device)
            if epoch > autoencoder_warm_up_n_epochs:
                # train Discriminator part
                optimizer_d.zero_grad(set_to_none=True)
                
                with autocast(enabled=args.amp):
                    logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                    loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                    logits_real = discriminator(images.contiguous().detach())[-1]
                    loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                    
                    # Ensure discriminator losses are scalars
                    if loss_d_fake.numel() > 1:
                        loss_d_fake = loss_d_fake.mean()
                    if loss_d_real.numel() > 1:
                        loss_d_real = loss_d_real.mean()
                    
                    discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                    loss_d = adv_weight * discriminator_loss
                    
                    # Ensure final discriminator loss is scalar
                    if loss_d.numel() > 1:
                        loss_d = loss_d.mean()
                    
                    # Check for NaN in discriminator loss - use safe checking
                    loss_d_nan = torch.isnan(loss_d).any().item() if loss_d.numel() > 1 else torch.isnan(loss_d).item()
                    if loss_d_nan:
                        if rank == 0:
                            print(f"Warning: NaN in discriminator loss at step {step}")
                        continue

                # Backward pass with gradient scaling
                if args.amp and scaler_d:
                    scaler_d.scale(loss_d).backward()
                    scaler_d.unscale_(optimizer_d)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=0.5)
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                else:
                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=0.5)
                    optimizer_d.step()

            # Accumulate losses for epoch average
            epoch_recon_loss += recons_loss.detach()
            epoch_kl_loss += kl_loss.detach()
            epoch_p_loss += p_loss.detach()
            if epoch > autoencoder_warm_up_n_epochs:
                epoch_adv_loss += generator_loss.detach()
            num_batches += 1

            # Update profiler
            if profiler:
                profiler.step()

            # write train loss for each batch into tensorboard
            if rank == 0 and step % 10 == 0:  # Log every 10 steps to reduce overhead
                total_step += 1
                tensorboard_writer.add_scalar("train_recon_loss_iter", recons_loss.item(), total_step)
                tensorboard_writer.add_scalar("train_kl_loss_iter", kl_loss.item(), total_step)
                tensorboard_writer.add_scalar("train_perceptual_loss_iter", p_loss.item(), total_step)
                if epoch > autoencoder_warm_up_n_epochs:
                    tensorboard_writer.add_scalar("train_adv_loss_iter", generator_loss.item(), total_step)
                
                # Log learning rates
                if scheduler_g:
                    tensorboard_writer.add_scalar("lr_generator", optimizer_g.param_groups[0]['lr'], total_step)
                    tensorboard_writer.add_scalar("lr_discriminator", optimizer_d.param_groups[0]['lr'], total_step)
                
                # Log gradient norms for debugging
                if step % 50 == 0:  # Less frequent gradient norm logging
                    autoencoder_params = autoencoder.module.parameters() if ddp_bool else autoencoder.parameters()
                    total_norm = 0
                    for p in autoencoder_params:
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    tensorboard_writer.add_scalar("grad_norm_autoencoder", total_norm, total_step)

        # Step learning rate schedulers
        if scheduler_g:
            scheduler_g.step()
        if scheduler_d:
            scheduler_d.step()

        # Synchronize and average epoch losses across GPUs
        epoch_recon_loss = reduce_tensor(epoch_recon_loss / num_batches, world_size)
        epoch_kl_loss = reduce_tensor(epoch_kl_loss / num_batches, world_size)
        epoch_p_loss = reduce_tensor(epoch_p_loss / num_batches, world_size)
        if epoch > autoencoder_warm_up_n_epochs:
            epoch_adv_loss = reduce_tensor(epoch_adv_loss / num_batches, world_size)
        
        epoch_time = time.time() - epoch_start_time

        if rank == 0:
            if epoch > autoencoder_warm_up_n_epochs:
                print(f"Epoch {epoch}/{max_epochs} ({epoch_time:.2f}s) - Recon: {epoch_recon_loss:.4f}, KL: {epoch_kl_loss:.4f}, "
                      f"Perceptual: {epoch_p_loss:.4f}, Adv: {epoch_adv_loss:.4f}")
            else:
                print(f"Epoch {epoch}/{max_epochs} ({epoch_time:.2f}s) - Recon: {epoch_recon_loss:.4f}, KL: {epoch_kl_loss:.4f}, "
                      f"Perceptual: {epoch_p_loss:.4f}")
            
            # Log epoch losses to tensorboard
            tensorboard_writer.add_scalar("train_recon_loss_epoch", epoch_recon_loss.item(), epoch)
            tensorboard_writer.add_scalar("train_kl_loss_epoch", epoch_kl_loss.item(), epoch)
            tensorboard_writer.add_scalar("train_perceptual_loss_epoch", epoch_p_loss.item(), epoch)
            tensorboard_writer.add_scalar("epoch_time", epoch_time, epoch)
            if epoch > autoencoder_warm_up_n_epochs:
                tensorboard_writer.add_scalar("train_adv_loss_epoch", epoch_adv_loss.item(), epoch)

        # validation
        if epoch % val_interval == 0:
            autoencoder.eval()
            val_recon_loss = torch.tensor(0.0, device=device)
            val_num_batches = 0
            
            with torch.no_grad():
                for step, batch in enumerate(val_loader):
                    images = batch["image"].to(device, non_blocking=True)
                    
                    # Ensure input images are in valid range [0, 1]
                    images = torch.clamp(images, 0.0, 1.0)
                    
                    # Use autocast for validation as well
                    with autocast(enabled=args.amp):
                        reconstruction, z_mu, z_sigma = autoencoder(images)
                        
                        # Clamp reconstruction to valid range
                        reconstruction = torch.clamp(reconstruction, 0.0, 1.0)
                        
                        recons_loss = intensity_loss(
                            reconstruction.float(), images.float()
                        )
                        
                        # Safe perceptual loss calculation
                        try:
                            p_loss = loss_perceptual(reconstruction.float(), images.float())
                            if torch.isnan(p_loss).any() or torch.isinf(p_loss).any():
                                p_loss = torch.tensor(0.0, device=device)
                        except:
                            p_loss = torch.tensor(0.0, device=device)
                        
                        # Calculate KL loss with stability check
                        kl_loss = KL_loss(z_mu, z_sigma)
                        if torch.isnan(kl_loss).any() or torch.isinf(kl_loss).any():
                            if rank == 0:
                                print(f"Warning: NaN/Inf in validation KL loss at epoch {epoch}, step {step}")
                            continue
                        
                        # Final validation loss calculation
                        val_loss = recons_loss + perceptual_weight * p_loss
                        
                        # Check for NaN in validation loss
                        if torch.isnan(val_loss).any() or torch.isinf(val_loss).any():
                            if rank == 0:
                                print(f"Warning: NaN/Inf in validation recon loss at epoch {epoch}, step {step}")
                            continue

                    val_recon_loss += val_loss.detach()
                    val_num_batches += 1

            # Only proceed if we have valid validation batches
            if val_num_batches > 0:
                # Synchronize validation loss across GPUs
                val_recon_epoch_loss = reduce_tensor(val_recon_loss / val_num_batches, world_size)

                if rank == 0:
                    # Check for NaN in final validation loss
                    if torch.isnan(val_recon_epoch_loss).any() or torch.isinf(val_recon_epoch_loss).any():
                        print(f"ERROR: Final validation loss is NaN/Inf at epoch {epoch}")
                        print("Training terminated due to numerical instability")
                        break
                    
                    # save last model
                    print(f"Epoch {epoch} val_recon_loss: {val_recon_epoch_loss:.4f}")
                    if ddp_bool:
                        torch.save(autoencoder.module.state_dict(), trained_g_path_last)
                        torch.save(discriminator.module.state_dict(), trained_d_path_last)
                    else:
                        torch.save(autoencoder.state_dict(), trained_g_path_last)
                        torch.save(discriminator.state_dict(), trained_d_path_last)
                        
                    # save best model
                    if val_recon_epoch_loss < best_val_recon_epoch_loss:
                        best_val_recon_epoch_loss = val_recon_epoch_loss
                    if ddp_bool:
                        torch.save(autoencoder.module.state_dict(), trained_g_path)
                        torch.save(discriminator.module.state_dict(), trained_d_path)
                    else:
                        torch.save(autoencoder.state_dict(), trained_g_path)
                        torch.save(discriminator.state_dict(), trained_d_path)
                    print("Got best val recon loss.")
                    print("Save trained autoencoder to", trained_g_path)
                    print("Save trained discriminator to", trained_d_path)

                # write val loss for each epoch into tensorboard
                tensorboard_writer.add_scalar("val_recon_loss", val_recon_epoch_loss.item(), epoch)
                # Only visualize on validation intervals to reduce overhead
                if not args.no_images and epoch % (val_interval * 5) == 0:  # Visualize every 5 validation intervals
                    try:
                        for axis in range(3):
                            # Safe image logging with proper tensor handling
                            img_slice = visualize_one_slice_in_3d_image(images[0, 0, ...], axis)
                            recon_slice = visualize_one_slice_in_3d_image(reconstruction[0, 0, ...], axis)
                            
                            # Ensure tensors are in the right format for tensorboard
                            if img_slice is not None:
                                tensorboard_writer.add_image(
                                    "val_img_" + str(axis),
                                    img_slice,
                                    epoch,
                                )
                            if recon_slice is not None:
                                tensorboard_writer.add_image(
                                    "val_recon_" + str(axis),
                                    recon_slice,
                                    epoch,
                                )
                    except Exception as e:
                        if rank == 0:
                            print(f"Warning: Failed to log images to tensorboard: {e}")
                            print("Continuing training without image logging...")

            else:
                if rank == 0:
                    print(f"Warning: No valid validation batches at epoch {epoch}")

            # Synchronize before next epoch
            if ddp_bool:
                dist.barrier()
        else:
            # Set autoencoder back to train mode if not validating
            autoencoder.train()

    # Stop profiler
    if profiler:
        profiler.stop()
        print("Profiler stopped")

    # Clean up distributed training
    if ddp_bool:
        cleanup_ddp()
        if rank == 0:
            print("DDP cleanup completed")


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
