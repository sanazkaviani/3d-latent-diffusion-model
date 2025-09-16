# 3D Latent Diffusion Model Utilities
import os
from glob import glob
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from monai.bundle import ConfigParser
from monai.data import DataLoader
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    RandSpatialCropd,
    ScaleIntensityRangePercentilesd,
)
from torch.utils.data.dataloader import default_collate

# Fix PIL compatibility issue
try:
    from PIL import Image
    if not hasattr(Image, 'ANTIALIAS'):
        Image.ANTIALIAS = Image.LANCZOS
except ImportError:
    pass

def safe_tensorboard_add_image(writer, tag, img_tensor, global_step, dataformats='CHW'):
    """
    Safely add image to tensorboard with PIL compatibility handling
    """
    try:
        # Ensure tensor is on CPU and detached
        if hasattr(img_tensor, 'detach'):
            img_tensor = img_tensor.detach().cpu()
        
        # Convert to numpy if it's a tensor
        if torch.is_tensor(img_tensor):
            img_tensor = img_tensor.numpy()
        
        # Ensure valid range [0, 1]
        img_tensor = np.clip(img_tensor, 0, 1)
        
        # Check for valid shape
        if img_tensor.ndim < 2:
            print(f"Warning: Invalid image shape {img_tensor.shape} for tensorboard logging")
            return
            
        writer.add_image(tag, img_tensor, global_step, dataformats=dataformats)
    except Exception as e:
        print(f"Warning: Failed to log image {tag} to tensorboard: {e}")


def setup_ddp(rank, world_size):
    print(f"Running DDP diffusion example on rank {rank}/world_size {world_size}.")
    print(f"Initing to IP {os.environ['MASTER_ADDR']}")
    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(seconds=36000), rank=rank, world_size=world_size
    )  # gloo, nccl
    dist.barrier()
    device = torch.device(f"cuda:{rank}")
    return dist, device


def prepare_dataloader(
    args,
    batch_size,
    patch_size,
    randcrop=True,
    rank=0,
    world_size=1,
    cache=1.0,
    download=False,
    size_divisible=16,
    amp=False,
):
    ddp_bool = world_size > 1

    if amp:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    # define crop sizes
    if randcrop:
        train_crop = RandSpatialCropd(keys=["image", "label"], roi_size=patch_size, random_size=False)
        val_patch_size = [int(np.ceil(1.5 * p / size_divisible) * size_divisible) for p in patch_size]
    else:
        train_crop = CenterSpatialCropd(keys=["image", "label"], roi_size=patch_size)
        val_patch_size = patch_size

    # MONAI transforms for paired low/high volumes
    train_transforms = Compose(
        [
            train_crop,
            ScaleIntensityRangePercentilesd(keys=["image", "label"], lower=0.0, upper=99.5, b_min=0.0, b_max=1.0),
            EnsureTyped(keys=["image", "label"], dtype=compute_dtype),
        ]
    )
    val_transforms = Compose(
        [
            CenterSpatialCropd(keys=["image", "label"], roi_size=val_patch_size),
            ScaleIntensityRangePercentilesd(keys=["image", "label"], lower=0.0, upper=99.5, b_min=0.0, b_max=1.0),
            EnsureTyped(keys=["image", "label"], dtype=compute_dtype),
        ]
    )

    # NPZ dataset: image=low-count, label=high-count from arr0/arr_0
    npz_dir_train = getattr(args, "npz_dir_train", None)
    npz_dir_val = getattr(args, "npz_dir_val", None)
    npz_dir = getattr(args, "npz_dir", None)
    val_fraction = float(getattr(args, "val_fraction", 0.1))
    seed = int(getattr(args, "seed", 0))

    class NPZPairDataset(torch.utils.data.Dataset):
        def __init__(self, files, transforms=None):
            self.files = files
            self.transforms = transforms

        def __len__(self):
            return len(self.files)

        def _load_pair(self, path):
            data = np.load(path)
            key = None
            if "arr0" in data:
                key = "arr0"
            elif "arr_0" in data:
                key = "arr_0"
            else:
                keys = list(data.keys())
                if len(keys) >= 1:
                    key = keys[0]
            if key is None:
                raise RuntimeError(f"NPZ {path} does not contain expected 'arr0' or 'arr_0'")
            arr = data[key]
            if arr.ndim < 4 or arr.shape[0] < 2:
                raise RuntimeError(f"NPZ {path} expected shape (2, D, H, W), got {arr.shape}")
            low = np.array(arr[0], dtype=np.float32)
            high = np.array(arr[1], dtype=np.float32)
            
            return low, high

        def __getitem__(self, idx):
            path = self.files[idx]
            low, high = self._load_pair(path)

            # Add channel dimension manually (if not already there)
            if low.ndim == 3:
                low = low[None, ...]  # shape becomes (1, D, H, W)
            if high.ndim == 3:
                high = high[None, ...]

            sample = {"image": low.astype(np.float32).copy(), "label": high.astype(np.float32).copy()}
            
            if self.transforms is not None:
                sample = self.transforms(sample)
            return sample

    # build file lists
    if npz_dir_train and npz_dir_val and os.path.isdir(npz_dir_train) and os.path.isdir(npz_dir_val):
        train_files = sorted(glob(os.path.join(npz_dir_train, "*.npz")))
        val_files = sorted(glob(os.path.join(npz_dir_val, "*.npz")))
        if len(train_files) == 0:
            raise ValueError(f"No .npz files found in train dir: {npz_dir_train}")
        if len(val_files) == 0:
            raise ValueError(f"No .npz files found in val dir: {npz_dir_val}")
    else:
        if not npz_dir or not os.path.isdir(npz_dir):
            raise ValueError(
                "Provide (npz_dir_train and npz_dir_val) or set npz_dir to a folder containing .npz files."
            )
        all_files = sorted(glob(os.path.join(npz_dir, "*.npz")))
        if len(all_files) == 0:
            raise ValueError(f"No .npz files found in {npz_dir}")
        rng = np.random.RandomState(seed)
        indices = np.arange(len(all_files))
        rng.shuffle(indices)
        n_val = int(len(indices) * val_fraction)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        train_files = [all_files[i] for i in train_idx]
        val_files = [all_files[i] for i in val_idx] if n_val > 0 else [all_files[i] for i in indices[:1]]

    train_ds = NPZPairDataset(train_files, transforms=train_transforms)
    val_ds = NPZPairDataset(val_files, transforms=val_transforms)
    if ddp_bool:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    # Improved DataLoader config for multi-GPU
    # Optimize num_workers based on system and GPU count
    if ddp_bool:
        # Use more workers for multi-GPU to keep GPUs fed with data
        num_workers = min(8, os.cpu_count() // world_size) if os.cpu_count() else 4
        pin_memory = True
        persistent_workers = True if num_workers > 0 else False
    else:
        num_workers = min(4, os.cpu_count() // 2) if os.cpu_count() else 0
        pin_memory = torch.cuda.is_available()
        persistent_workers = False

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=(train_sampler is None), 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        sampler=train_sampler, 
        collate_fn=default_collate,
        drop_last=ddp_bool,  # Ensure consistent batch sizes across GPUs
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else 2
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        sampler=val_sampler, 
        collate_fn=default_collate,
        drop_last=ddp_bool,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else 2
    )
    
    if rank == 0:
        print(f"DataLoader config: num_workers={num_workers}, pin_memory={pin_memory}, "
              f"persistent_workers={persistent_workers}")
        print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
        try:
            print(f'Image shape {train_ds[0]["image"].shape}')
        except Exception:
            pass
    return train_loader, val_loader


def define_instance(args, instance_def_key):
    parser = ConfigParser(vars(args))
    parser.parse(True)
    return parser.get_parsed_content(instance_def_key, instantiate=True)


def KL_loss(z_mu, z_sigma):
    # Add numerical stability
    eps = 1e-8
    z_sigma_clamped = torch.clamp(z_sigma, min=eps)
    
    # More numerically stable KL divergence calculation
    kl_loss = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma_clamped.pow(2) - torch.log(z_sigma_clamped.pow(2) + eps) - 1,
        dim=list(range(1, len(z_sigma.shape))),
    )
    
    # Clamp the final result to prevent explosion
    kl_loss_per_sample = kl_loss / kl_loss.shape[0]
    return torch.clamp(kl_loss_per_sample, 0.0, 1000.0)
