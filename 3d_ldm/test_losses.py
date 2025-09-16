#!/usr/bin/env python3
"""
Quick test script to validate tensor operations and loss computations
Run this before the full training to catch tensor shape issues early
"""

import torch
import torch.nn as nn
from monai.losses import PatchAdversarialLoss, PerceptualLoss

def test_loss_scalars():
    """Test that all loss functions return scalars"""
    print("Testing loss function outputs...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    
    # Create dummy tensors
    images = torch.randn(batch_size, 1, 32, 32, 32).to(device)
    reconstruction = torch.randn(batch_size, 1, 32, 32, 32).to(device)
    z_mu = torch.randn(batch_size, 8, 4, 4, 4).to(device)
    z_sigma = torch.randn(batch_size, 8, 4, 4, 4).to(device)
    
    # Test basic losses
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    
    recons_l1 = l1_loss(reconstruction, images)
    recons_mse = mse_loss(reconstruction, images)
    
    print(f"L1 Loss: {recons_l1}, shape: {recons_l1.shape}, numel: {recons_l1.numel()}")
    print(f"MSE Loss: {recons_mse}, shape: {recons_mse.shape}, numel: {recons_mse.numel()}")
    
    # Test KL loss
    def KL_loss(z_mu, z_sigma):
        eps = 1e-8
        z_sigma_clamped = torch.clamp(z_sigma, min=eps)
        kl_loss = 0.5 * torch.sum(
            z_mu.pow(2) + z_sigma_clamped.pow(2) - torch.log(z_sigma_clamped.pow(2) + eps) - 1,
            dim=list(range(1, len(z_sigma.shape))),
        )
        return torch.clamp(kl_loss / kl_loss.shape[0], 0.0, 1000.0)
    
    kl_loss_val = KL_loss(z_mu, z_sigma)
    print(f"KL Loss: {kl_loss_val}, shape: {kl_loss_val.shape}, numel: {kl_loss_val.numel()}")
    
    # Test perceptual loss
    try:
        loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
        loss_perceptual.to(device)
        p_loss = loss_perceptual(reconstruction.float(), images.float())
        print(f"Perceptual Loss: {p_loss}, shape: {p_loss.shape}, numel: {p_loss.numel()}")
    except Exception as e:
        print(f"Perceptual loss test failed: {e}")
    
    # Test adversarial loss
    try:
        adv_loss = PatchAdversarialLoss(criterion="least_squares")
        logits = torch.randn(batch_size, 1, 4, 4, 4).to(device)
        adv_loss_val = adv_loss(logits, target_is_real=True, for_discriminator=False)
        print(f"Adversarial Loss: {adv_loss_val}, shape: {adv_loss_val.shape}, numel: {adv_loss_val.numel()}")
    except Exception as e:
        print(f"Adversarial loss test failed: {e}")
    
    print("\nTesting tensor operations...")
    
    # Test NaN checking
    test_tensor = torch.tensor([1.0, float('nan'), 3.0])
    nan_check = torch.isnan(test_tensor).any().item()
    print(f"NaN check result: {nan_check}")
    
    # Test scalar operations
    scalar_tensor = torch.tensor(5.0)
    multi_tensor = torch.tensor([1.0, 2.0, 3.0])
    
    print(f"Scalar tensor numel: {scalar_tensor.numel()}")
    print(f"Multi tensor numel: {multi_tensor.numel()}")
    
    # Test mean reduction
    reduced = multi_tensor.mean()
    print(f"Reduced tensor: {reduced}, numel: {reduced.numel()}")
    
    print("✅ All tests completed!")

if __name__ == "__main__":
    test_loss_scalars()
