"""
FastAPI server for 3D Latent Diffusion Model
Production-ready API with async support, monitoring, and proper error handling
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import torch
from pathlib import Path
import traceback
import time
import psutil

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Monitoring
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('model_requests_total', 'Total model requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('model_request_duration_seconds', 'Request duration in seconds')
INFERENCE_DURATION = Histogram('model_inference_duration_seconds', 'Model inference duration in seconds')
ACTIVE_REQUESTS = Gauge('model_active_requests', 'Number of active requests')
MEMORY_USAGE = Gauge('model_memory_usage_bytes', 'Memory usage in bytes')
GPU_MEMORY_USAGE = Gauge('model_gpu_memory_usage_bytes', 'GPU memory usage in bytes')

# Pydantic models
class GenerationRequest(BaseModel):
    num_samples: int = Field(default=1, ge=1, le=10, description="Number of samples to generate")
    inference_steps: int = Field(default=100, ge=10, le=1000, description="Number of inference steps")
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0, description="Guidance scale for generation")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    output_format: str = Field(default="base64", regex="^(base64|nii)$", description="Output format")
    
    @validator('num_samples')
    def validate_num_samples(cls, v):
        if v > 5:
            logger.warning(f"Large number of samples requested: {v}")
        return v

class GenerationResponse(BaseModel):
    status: str
    num_samples: int
    inference_steps: int
    guidance_scale: float
    output_format: str
    generated_images: List[Dict[str, Any]]
    model_info: Dict[str, Any]
    request_id: str
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    gpu_available: bool
    memory_usage: Dict[str, float]
    uptime_seconds: float

# Global variables for model
model = None
model_config = None
device = None
start_time = datetime.now()

class ModelServer:
    """Model server with async support and proper resource management"""
    
    def __init__(self):
        self.model = None
        self.device = None
        self.config = None
        self.is_ready = False
        
    async def load_model(self):
        """Asynchronously load the model"""
        try:
            logger.info("Loading 3D Latent Diffusion Model...")
            
            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Load configuration
            config_path = "/app/config/config_train_32g.json"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                self.config = {
                    "patch_size": [64, 64, 64],
                    "num_inference_steps": 100,
                    "guidance_scale": 7.5
                }
            
            # Load model (placeholder - replace with actual model loading)
            model_path = "/app/models/3d_ldm_model.pth"
            if os.path.exists(model_path):
                self.model = torch.load(model_path, map_location=self.device)
                self.model.eval()
                logger.info("Model loaded from checkpoint")
            else:
                # Create dummy model for testing
                logger.warning("No model checkpoint found, using dummy model")
                self.model = self._create_dummy_model()
            
            self.is_ready = True
            logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _create_dummy_model(self):
        """Create a dummy model for testing purposes"""
        class DummyModel(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.device = device
                
            def forward(self, x):
                # Simulate processing time
                time.sleep(0.1)
                return torch.randn_like(x)
                
            def generate(self, noise, num_inference_steps=100, guidance_scale=7.5):
                # Simulate generation process
                time.sleep(num_inference_steps * 0.01)  # Simulate inference time
                return torch.randn_like(noise)
        
        return DummyModel(self.device).to(self.device)
    
    async def generate(self, request: GenerationRequest, request_id: str) -> Dict[str, Any]:
        """Generate images asynchronously"""
        
        if not self.is_ready:
            raise HTTPException(status_code=503, detail="Model not ready")
        
        start_time = time.time()
        
        try:
            # Set random seed if provided
            if request.seed is not None:
                torch.manual_seed(request.seed)
                np.random.seed(request.seed)
            
            generated_images = []
            
            with torch.no_grad():
                for i in range(request.num_samples):
                    # Generate random noise
                    patch_size = self.config.get("patch_size", [64, 64, 64])
                    noise = torch.randn(1, 1, *patch_size, device=self.device)
                    
                    # Generate image
                    if hasattr(self.model, 'generate'):
                        generated_image = self.model.generate(
                            noise,
                            num_inference_steps=request.inference_steps,
                            guidance_scale=request.guidance_scale
                        )
                    else:
                        generated_image = self.model(noise)
                    
                    # Convert to numpy
                    generated_image = generated_image.cpu().numpy().squeeze()
                    
                    # Normalize to [0, 1] range
                    generated_image = (generated_image - generated_image.min()) / (generated_image.max() - generated_image.min())
                    
                    if request.output_format == "base64":
                        # Convert to base64
                        import base64
                        image_bytes = generated_image.astype(np.float32).tobytes()
                        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                        
                        generated_images.append({
                            "image_data": image_b64,
                            "shape": generated_image.shape,
                            "dtype": "float32",
                            "sample_id": i
                        })
                    else:
                        # Save as NIfTI (simplified for demo)
                        generated_images.append({
                            "shape": generated_image.shape,
                            "sample_id": i,
                            "message": "NIfTI format not implemented in demo"
                        })
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Update metrics
            INFERENCE_DURATION.observe(processing_time_ms / 1000)
            
            return {
                "status": "success",
                "num_samples": request.num_samples,
                "inference_steps": request.inference_steps,
                "guidance_scale": request.guidance_scale,
                "output_format": request.output_format,
                "generated_images": generated_images,
                "model_info": {
                    "patch_size": patch_size,
                    "device": str(self.device)
                },
                "request_id": request_id,
                "processing_time_ms": processing_time_ms
            }
            
        except Exception as e:
            logger.error(f"❌ Error during generation: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

# Initialize FastAPI app
app = FastAPI(
    title="3D Latent Diffusion Model API",
    description="Production API for 3D medical image generation using Latent Diffusion Models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Initialize model server
model_server = ModelServer()

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    try:
        await model_server.load_model()
        logger.info("🚀 API server started successfully")
    except Exception as e:
        logger.error(f"❌ Failed to start server: {str(e)}")
        raise

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add request processing time to response headers"""
    start_time = time.time()
    ACTIVE_REQUESTS.inc()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # Update metrics
        REQUEST_DURATION.observe(process_time)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        return response
    finally:
        ACTIVE_REQUESTS.dec()

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "3D Latent Diffusion Model API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Get system metrics
        memory = psutil.virtual_memory()
        uptime = (datetime.now() - start_time).total_seconds()
        
        # Update Prometheus metrics
        MEMORY_USAGE.set(memory.used)
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated()
            GPU_MEMORY_USAGE.set(gpu_memory)
        
        return HealthResponse(
            status="healthy" if model_server.is_ready else "unhealthy",
            timestamp=datetime.now().isoformat(),
            model_loaded=model_server.is_ready,
            gpu_available=torch.cuda.is_available(),
            memory_usage={
                "used_gb": round(memory.used / (1024**3), 2),
                "total_gb": round(memory.total / (1024**3), 2),
                "percentage": memory.percent
            },
            uptime_seconds=uptime
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Health check failed")

@app.post("/generate", response_model=GenerationResponse)
async def generate_images(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    request_id: str = None
):
    """Generate medical images using the 3D LDM"""
    
    if request_id is None:
        request_id = f"req_{int(time.time())}_{np.random.randint(1000, 9999)}"
    
    logger.info(f"📥 Generation request {request_id}: {request.num_samples} samples")
    
    try:
        result = await model_server.generate(request, request_id)
        logger.info(f"✅ Generation completed {request_id}: {result['processing_time_ms']:.2f}ms")
        return GenerationResponse(**result)
        
    except Exception as e:
        logger.error(f"❌ Generation failed {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )

@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    if not model_server.is_ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    return {
        "model_loaded": model_server.is_ready,
        "device": str(model_server.device),
        "config": model_server.config,
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/model/reload")
async def reload_model():
    """Reload the model (admin endpoint)"""
    try:
        logger.info("🔄 Reloading model...")
        await model_server.load_model()
        return {"status": "success", "message": "Model reloaded successfully"}
    except Exception as e:
        logger.error(f"❌ Model reload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info",
        access_log=True
    )