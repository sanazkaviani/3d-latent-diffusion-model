import argparse
import json
import logging
from pathlib import Path

import os
import sys

import torch
import torch.nn.functional as F
from monai.inferers import LatentDiffusionInferer
from monai.networks.schedulers import DDPMScheduler
from monai.config import print_config
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from utils import define_instance, prepare_dataloader, setup_ddp
from visualize_image import visualize_one_slice_in_3d_image


def main():
    parser = argparse.ArgumentParser(description="PyTorch Latent Diffusion Model Training")
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
    print(f"Using {device}")

    #print_config()
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    set_determinism(42)

    # Step 1: set data loader
    size_divisible = 2 ** (len(args.autoencoder_def["channels"]) + len(args.diffusion_def["channels"]) - 2)
    train_loader, val_loader = prepare_dataloader(
        args,
        args.diffusion_train["batch_size"],
        args.diffusion_train["patch_size"],
        randcrop=False,
        rank=rank,
        world_size=world_size,
        cache=1.0,
        size_divisible=size_divisible,
        amp=True,
    )

    # initialize tensorboard writer
    if rank == 0:
        Path(args.tfevent_path).mkdir(parents=True, exist_ok=True)
        tensorboard_path = os.path.join(args.tfevent_path, "diffusion")
        tensorboard_writer = SummaryWriter(tensorboard_path)

    # Step 2: Define Autoencoder KL network and diffusion model
    # Load Autoencoder KL network
    autoencoder = define_instance(args, "autoencoder_def").to(device)

    trained_g_path = os.path.join(args.model_dir, "autoencoder.pt")

    map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
    autoencoder.load_state_dict(torch.load(trained_g_path, map_location=map_location))
    print(f"Rank {rank}: Load trained autoencoder from {trained_g_path}")

    # Compute Scaling factor for LABELS (not images)
    # We need to compute the scaling factor based on label latents, not image latents
    with torch.no_grad():
        with autocast(enabled=True):
            check_data = first(train_loader)
            # Encode labels instead of images to get the target latent space
            z_labels = autoencoder.encode_stage_2_inputs(check_data["label"].to(device))
            if rank == 0:
                print(f"Label latent feature shape {z_labels.shape}")
                for axis in range(3):
                    tensorboard_writer.add_image(
                        "train_img_" + str(axis),
                        visualize_one_slice_in_3d_image(check_data["image"][0, 0, ...], axis).transpose([2, 1, 0]),
                        1,
                    )
                    tensorboard_writer.add_image(
                        "train_label_" + str(axis),
                        visualize_one_slice_in_3d_image(check_data["label"][0, 0, ...], axis).transpose([2, 1, 0]),
                        1,
                    )
                print(f"Scaling factor set to {1/torch.std(z_labels)}")
    scale_factor = 1 / torch.std(z_labels)
    print(f"Rank {rank}: local scale_factor: {scale_factor}")
    if ddp_bool:
        dist.barrier()
        dist.all_reduce(scale_factor, op=torch.distributed.ReduceOp.AVG)
    print(f"Rank {rank}: final scale_factor -> {scale_factor}")

    # Define Diffusion Model
    unet = define_instance(args, "diffusion_def").to(device)

    trained_diffusion_path = os.path.join(args.model_dir, "diffusion_unet.pt")
    trained_diffusion_path_last = os.path.join(args.model_dir, "diffusion_unet_last.pt")

    if args.resume_ckpt:
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        try:
            unet.load_state_dict(torch.load(trained_diffusion_path, map_location=map_location))
            print(f"Rank {rank}: Load trained diffusion model from", trained_diffusion_path)
        except:
            print(f"Rank {rank}: Train diffusion model from scratch.")

    scheduler = DDPMScheduler(
        num_train_timesteps=args.NoiseScheduler["num_train_timesteps"],
        schedule="scaled_linear_beta",
        beta_start=args.NoiseScheduler["beta_start"],
        beta_end=args.NoiseScheduler["beta_end"],
    )

    if ddp_bool:
        autoencoder = DDP(autoencoder, device_ids=[device], output_device=rank)
        unet = DDP(unet, device_ids=[device], output_device=rank)

    # We define the inferer using the scale factor:
    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

    # Step 3: training config
    optimizer_diff = torch.optim.Adam(params=unet.parameters(), lr=args.diffusion_train["lr"])  # Remove world_size multiplication
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_diff, milestones=[100, 1000], gamma=0.1)

    # Step 4: training
    max_epochs = args.diffusion_train["max_epochs"]
    val_interval = args.diffusion_train["val_interval"]
    autoencoder.eval()
    scaler = GradScaler()
    total_step = 0
    best_val_recon_epoch_loss = float('inf')  # Use inf instead of 100.0

    for epoch in range(max_epochs):
        unet.train()
        epoch_loss = 0
        if ddp_bool:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device).float()  # Convert to float32
            labels = batch["label"].to(device).float()  # Convert to float32
            optimizer_diff.zero_grad(set_to_none=True)

            with autocast(enabled=False):  # Disable autocast to avoid numerical issues
                # Generate random noise in the shape of LABEL latents
                with torch.no_grad():
                    label_latents = autoencoder.encode_stage_2_inputs(labels) if ddp_bool == False else autoencoder.module.encode_stage_2_inputs(labels)
                noise_shape = label_latents.shape
                noise = torch.randn(noise_shape, dtype=torch.float32).to(device)  # Explicit float32

                # Create timesteps
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps, (labels.shape[0],), device=labels.device
                ).long()

                # Get model prediction - use LABELS as inputs and IMAGE LATENTS as conditioning
                if ddp_bool:
                    inferer_autoencoder = autoencoder.module
                else:
                    inferer_autoencoder = autoencoder
                with torch.no_grad():
                    image_latents = inferer_autoencoder.encode_stage_2_inputs(images)
                    
                noise_pred = inferer(
                    inputs=labels,
                    autoencoder_model=inferer_autoencoder,
                    diffusion_model=unet,
                    noise=noise,
                    timesteps=timesteps,
                    condition=image_latents,
                    mode="concat"
                )

                loss = F.mse_loss(noise_pred.float(), noise.float())
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"NaN loss detected at epoch {epoch}, step {step}")
                    continue

            loss.backward()  # Remove scaler for now
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            
            optimizer_diff.step()
            
            # Move lr_scheduler.step() after optimizer.step()
            if step == len(train_loader) - 1:
                lr_scheduler.step()

            # write train loss for each batch into tensorboard
            if rank == 0:
                total_step += 1
                tensorboard_writer.add_scalar("train_diffusion_loss_iter", loss.item(), total_step)

        # validation
        if epoch % val_interval == 0:
            autoencoder.eval()
            unet.eval()
            val_recon_epoch_loss = 0
            valid_batches = 0
            with torch.no_grad():
                with autocast(enabled=False):  # Disable autocast for validation too
                    # compute val loss - use same setup as training for loss computation
                    for step, batch in enumerate(val_loader):
                        images = batch["image"].to(device).float()  # Convert to float32
                        labels = batch["label"].to(device).float()  # Convert to float32
                        
                        # Generate noise in label latent space
                        if ddp_bool:
                            inferer_autoencoder = autoencoder.module
                        else:
                            inferer_autoencoder = autoencoder
                        
                        label_latents = inferer_autoencoder.encode_stage_2_inputs(labels)
                        noise_shape = label_latents.shape
                        noise = torch.randn(noise_shape, dtype=torch.float32).to(device)  # Explicit float32

                        timesteps = torch.randint(
                            0, inferer.scheduler.num_train_timesteps, (labels.shape[0],), device=labels.device
                        ).long()

                        # Training-style validation: predict noise for high-count given low-count conditioning
                        image_latents = inferer_autoencoder.encode_stage_2_inputs(images)
                        
                        noise_pred = inferer(
                            inputs=labels,
                            autoencoder_model=inferer_autoencoder,
                            diffusion_model=unet,
                            noise=noise,
                            timesteps=timesteps,
                            condition=image_latents,
                            mode="concat"
                        )
                        val_loss = F.mse_loss(noise_pred.float(), noise.float())
                        
                        # Skip NaN losses in validation
                        if not torch.isnan(val_loss):
                            val_recon_epoch_loss += val_loss
                            valid_batches += 1
                    if valid_batches > 0:
                        val_recon_epoch_loss = val_recon_epoch_loss / valid_batches
                    else:
                        val_recon_epoch_loss = torch.tensor(float('inf'))


                    if ddp_bool:
                        dist.barrier()
                        dist.all_reduce(val_recon_epoch_loss, op=torch.distributed.ReduceOp.AVG)

                    val_recon_epoch_loss = val_recon_epoch_loss.item()

                    # write val loss and save best model
                    if rank == 0:
                        tensorboard_writer.add_scalar("val_diffusion_loss", val_recon_epoch_loss, epoch)
                        print(f"Epoch {epoch} val_diffusion_loss: {val_recon_epoch_loss}")
                        # save last model
                        if ddp_bool:
                            torch.save(unet.module.state_dict(), trained_diffusion_path_last)
                        else:
                            torch.save(unet.state_dict(), trained_diffusion_path_last)

                        # save best model
                        if val_recon_epoch_loss < best_val_recon_epoch_loss and rank == 0:
                            best_val_recon_epoch_loss = val_recon_epoch_loss
                            if ddp_bool:
                                torch.save(unet.module.state_dict(), trained_diffusion_path)
                            else:
                                torch.save(unet.state_dict(), trained_diffusion_path)
                            print("Got best val noise pred loss.")
                            print("Save trained latent diffusion model to", trained_diffusion_path)

                        # Test denoising capability: generate high-count from low-count without conditioning
                        if (epoch) % (2 * val_interval) == 0:
                            # Generate noise in the same shape as high-count latents
                            test_noise_shape = inferer_autoencoder.encode_stage_2_inputs(labels[0:1, ...]).shape
                            test_noise = torch.randn(test_noise_shape, dtype=torch.float32).to(device)  # Explicit float32
                            
                            # Test unconditional generation (pure denoising capability)
                            # denoised_images = inferer.sample(
                            #     input_noise=test_noise,
                            #     autoencoder_model=inferer_autoencoder,
                            #     diffusion_model=unet,
                            #     scheduler=scheduler,
                            #     conditioning=None,  # No conditioning - test pure denoising
                            #     mode="crossattn"
                            # )
                            
                            # Also test conditional generation for comparison
                            image_latents_sample = inferer_autoencoder.encode_stage_2_inputs(images[0:1, ...])
                            
                            conditional_denoised = inferer.sample(
                                input_noise=test_noise,
                                autoencoder_model=inferer_autoencoder,
                                diffusion_model=unet,
                                scheduler=scheduler,
                                conditioning=image_latents_sample,  # Use raw image latents
                                mode="concat"  # Use concatenation instead of cross-attention
                            )
                            
                            for axis in range(3):
                                # Original low-count image
                                tensorboard_writer.add_image(
                                    "val_lowcount_input_" + str(axis),
                                    visualize_one_slice_in_3d_image(images[0, 0, ...], axis).transpose([2, 1, 0]),
                                    epoch,
                                )
                                # Ground truth high-count image
                                tensorboard_writer.add_image(
                                    "val_highcount_gt_" + str(axis),
                                    visualize_one_slice_in_3d_image(labels[0, 0, ...], axis).transpose([2, 1, 0]),
                                    epoch,
                                )
                                # Unconditional denoised (pure model capability)
                                # tensorboard_writer.add_image(
                                #     "val_denoised_uncond_" + str(axis),
                                #     visualize_one_slice_in_3d_image(denoised_images[0, 0, ...], axis).transpose([2, 1, 0]),
                                #     epoch,
                                # )
                                # Conditional denoised (with low-count guidance)
                                tensorboard_writer.add_image(
                                    "val_denoised_cond_" + str(axis),
                                    visualize_one_slice_in_3d_image(conditional_denoised[0, 0, ...], axis).transpose([2, 1, 0]),
                                    epoch,
                                )


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()