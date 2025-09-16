"""
Scoring Script for 3D Latent Diffusion Model Deployment
This script handles model inference requests for the deployed model
"""

import os
import json
import logging
import numpy as np
import torch
import mlflow
import mlflow.pytorch
from typing import Dict, List, Any
import base64
import io
import nibabel as nib
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init():
    """
    Initialize the model for inference
    This function is called when the container is initialized/started
    """
    global model, device, config
    
    try:
        # Get model path from environment variable
        model_path = os.getenv("AZUREML_MODEL_DIR")
        if model_path is None:
            model_path = "./model"
        
        logger.info(f"Loading model from: {model_path}")
        
        # Load configuration
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Default configuration
            config = {
                "patch_size": [64, 64, 64],
                "num_inference_steps": 100,
                "guidance_scale": 7.5
            }
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load model using MLflow
        try:
            model = mlflow.pytorch.load_model(model_path)
            model.to(device)
            model.eval()
            logger.info("Model loaded successfully via MLflow")
        except Exception as e:
            logger.error(f"Error loading model via MLflow: {e}")
            # Fallback to direct PyTorch loading
            model_file = os.path.join(model_path, "model.pth")
            if os.path.exists(model_file):
                model = torch.load(model_file, map_location=device)
                model.eval()
                logger.info("Model loaded successfully via PyTorch")
            else:
                raise Exception(f"No model found at {model_path}")
        
        logger.info("Model initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Error during model initialization: {str(e)}")
        raise

def run(raw_data: str) -> str:
    """
    Run inference on the input data
    
    Args:
        raw_data: JSON string containing the input data
        
    Returns:
        JSON string containing the generated images
    """
    try:
        # Parse input data
        data = json.loads(raw_data)
        logger.info(f"Received request with keys: {list(data.keys())}")
        
        # Extract parameters
        num_samples = data.get("num_samples", 1)
        inference_steps = data.get("inference_steps", config.get("num_inference_steps", 100))
        guidance_scale = data.get("guidance_scale", config.get("guidance_scale", 7.5))
        seed = data.get("seed", None)
        output_format = data.get("output_format", "nii")  # "nii" or "base64"
        
        logger.info(f"Generating {num_samples} samples with {inference_steps} steps")
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Generate images
        with torch.no_grad():
            generated_images = []
            
            for i in range(num_samples):
                # Generate random noise
                patch_size = config.get("patch_size", [64, 64, 64])
                noise = torch.randn(1, 1, *patch_size, device=device)
                
                # Generate image using the model
                if hasattr(model, 'generate'):
                    # If model has a generate method
                    generated_image = model.generate(
                        noise,
                        num_inference_steps=inference_steps,
                        guidance_scale=guidance_scale
                    )
                else:
                    # Direct model forward pass
                    generated_image = model(noise)
                
                # Convert to numpy
                generated_image = generated_image.cpu().numpy().squeeze()
                
                # Normalize to [0, 1] range
                generated_image = (generated_image - generated_image.min()) / (generated_image.max() - generated_image.min())
                
                if output_format == "base64":
                    # Convert to base64 for JSON response
                    image_bytes = generated_image.astype(np.float32).tobytes()
                    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                    
                    generated_images.append({
                        "image_data": image_b64,
                        "shape": generated_image.shape,
                        "dtype": "float32",
                        "sample_id": i
                    })
                else:
                    # Save as NIfTI file and return metadata
                    nii_img = nib.Nifti1Image(generated_image, affine=np.eye(4))
                    
                    # Create temporary file path
                    temp_path = f"/tmp/generated_sample_{i}.nii.gz"
                    nib.save(nii_img, temp_path)
                    
                    # Read file and encode
                    with open(temp_path, 'rb') as f:
                        file_bytes = f.read()
                    file_b64 = base64.b64encode(file_bytes).decode('utf-8')
                    
                    generated_images.append({
                        "nii_data": file_b64,
                        "shape": generated_image.shape,
                        "sample_id": i,
                        "filename": f"generated_sample_{i}.nii.gz"
                    })
                    
                    # Clean up temporary file
                    os.remove(temp_path)
        
        # Prepare response
        response = {
            "status": "success",
            "num_samples": num_samples,
            "inference_steps": inference_steps,
            "guidance_scale": guidance_scale,
            "output_format": output_format,
            "generated_images": generated_images,
            "model_info": {
                "patch_size": patch_size,
                "device": str(device)
            }
        }
        
        logger.info(f"Successfully generated {num_samples} images")
        return json.dumps(response)
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        error_response = {
            "status": "error",
            "error_message": str(e),
            "error_type": type(e).__name__
        }
        return json.dumps(error_response)

def get_model_info() -> Dict[str, Any]:
    """Get model information for health checks"""
    try:
        return {
            "model_loaded": model is not None,
            "device": str(device),
            "config": config,
            "status": "healthy"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Health check endpoint
def health_check():
    """Health check function"""
    try:
        info = get_model_info()
        if info["status"] == "healthy":
            return "Model is healthy and ready for inference"
        else:
            return f"Model health check failed: {info.get('error', 'Unknown error')}"
    except Exception as e:
        return f"Health check error: {str(e)}"

# Additional utility functions for debugging
def get_system_info():
    """Get system information for debugging"""
    import platform
    import psutil
    
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "torch_version": torch.__version__
    }

# Test function for local development
def test_local():
    """Test function for local development"""
    print("Testing scoring script locally...")
    
    # Initialize model
    init()
    
    # Test inference
    test_data = {
        "num_samples": 1,
        "inference_steps": 10,  # Reduced for testing
        "output_format": "base64"
    }
    
    result = run(json.dumps(test_data))
    response = json.loads(result)
    
    if response["status"] == "success":
        print("✅ Local test successful!")
        print(f"Generated {response['num_samples']} images")
        print(f"Image shape: {response['generated_images'][0]['shape']}")
    else:
        print("❌ Local test failed!")
        print(f"Error: {response.get('error_message', 'Unknown error')}")

if __name__ == "__main__":
    # Run local test
    test_local()