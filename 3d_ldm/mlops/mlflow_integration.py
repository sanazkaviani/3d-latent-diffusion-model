"""
MLflow Integration for 3D Latent Diffusion Model
Provides experiment tracking, model versioning, and model registry functionality
"""

import mlflow
import mlflow.pytorch
import os
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import yaml

class MLflowTracker:
    """MLflow experiment tracking and model management"""
    
    def __init__(self, 
                 experiment_name: str = "3d-ldm-experiment",
                 tracking_uri: Optional[str] = None,
                 azure_ml_workspace: Optional[str] = None):
        """
        Initialize MLflow tracker
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking URI (if None, uses Azure ML)
            azure_ml_workspace: Azure ML workspace name for integration
        """
        self.experiment_name = experiment_name
        self.azure_ml_workspace = azure_ml_workspace
        
        # Set up MLflow tracking
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        elif azure_ml_workspace:
            # Use Azure ML as tracking backend
            self._setup_azure_ml_tracking()
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        self.run = None
        self.model_name = "3d-latent-diffusion-model"
    
    def _setup_azure_ml_tracking(self):
        """Set up Azure ML as MLflow tracking backend"""
        try:
            # Load workspace config
            config_path = Path(__file__).parent / "azure" / "workspace_config.yml"
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Create ML client
            credential = DefaultAzureCredential()
            ml_client = MLClient(
                credential=credential,
                subscription_id=config['subscription_id'],
                resource_group_name=config['resource_group'],
                workspace_name=config['workspace_name']
            )
            
            # Set MLflow tracking URI to Azure ML
            mlflow.set_tracking_uri(ml_client.workspaces.get().mlflow_tracking_uri)
            print(f"✅ Connected to Azure ML workspace: {config['workspace_name']}")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not connect to Azure ML workspace: {e}")
            print("Using local MLflow tracking")
    
    def start_run(self, 
                  run_name: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run"""
        
        default_tags = {
            "model_type": "3d_latent_diffusion",
            "framework": "pytorch",
            "library": "monai"
        }
        
        if tags:
            default_tags.update(tags)
        
        self.run = mlflow.start_run(run_name=run_name, tags=default_tags)
        return self.run
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration parameters"""
        if not self.run:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
        # Flatten nested config for MLflow
        flattened_config = self._flatten_dict(config)
        mlflow.log_params(flattened_config)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log training metrics"""
        if not self.run:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_artifacts(self, artifacts_dir: str):
        """Log artifacts directory"""
        if not self.run:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
        mlflow.log_artifacts(artifacts_dir)
    
    def log_model(self, 
                  model: torch.nn.Module,
                  model_path: str = "model",
                  signature: Optional[mlflow.types.Schema] = None,
                  input_example: Optional[np.ndarray] = None,
                  pip_requirements: Optional[list] = None):
        """Log PyTorch model to MLflow"""
        if not self.run:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
        # Default pip requirements
        if pip_requirements is None:
            pip_requirements = [
                "torch>=2.0.0",
                "torchvision>=0.15.0",
                "monai>=1.3.0",
                "numpy>=1.21.0",
                "nibabel>=5.0.0"
            ]
        
        # Log model
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=model_path,
            signature=signature,
            input_example=input_example,
            pip_requirements=pip_requirements
        )
        
        print(f"✅ Model logged to MLflow: {model_path}")
    
    def register_model(self, 
                      model_uri: str,
                      model_name: Optional[str] = None,
                      model_version: Optional[str] = None,
                      description: Optional[str] = None,
                      tags: Optional[Dict[str, str]] = None) -> mlflow.entities.ModelVersion:
        """Register model in MLflow Model Registry"""
        
        if model_name is None:
            model_name = self.model_name
        
        if description is None:
            description = "3D Latent Diffusion Model for medical image synthesis"
        
        default_tags = {
            "model_type": "generative",
            "domain": "medical_imaging",
            "architecture": "latent_diffusion"
        }
        
        if tags:
            default_tags.update(tags)
        
        # Register model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            tags=default_tags
        )
        
        # Update model version description
        client = mlflow.tracking.MlflowClient()
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=description
        )
        
        print(f"✅ Model registered: {model_name} v{model_version.version}")
        return model_version
    
    def log_figure(self, figure, filename: str):
        """Log matplotlib figure"""
        if not self.run:
            raise RuntimeError("No active MLflow run. Call start_run() first.")
        
        mlflow.log_figure(figure, filename)
    
    def end_run(self):
        """End the current MLflow run"""
        if self.run:
            mlflow.end_run()
            self.run = None
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary for MLflow logging"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                # Convert complex types to strings
                if isinstance(v, (list, tuple)):
                    v = str(v)
                elif isinstance(v, (np.ndarray, torch.Tensor)):
                    v = str(v.shape) if hasattr(v, 'shape') else str(v)
                items.append((new_key, v))
        return dict(items)

class ModelRegistry:
    """Model registry operations for production deployment"""
    
    def __init__(self, model_name: str = "3d-latent-diffusion-model"):
        self.model_name = model_name
        self.client = mlflow.tracking.MlflowClient()
    
    def get_latest_model_version(self, stage: str = "Production") -> Optional[mlflow.entities.ModelVersion]:
        """Get latest model version in specified stage"""
        try:
            versions = self.client.get_latest_versions(
                name=self.model_name,
                stages=[stage]
            )
            return versions[0] if versions else None
        except Exception as e:
            print(f"Error getting model version: {e}")
            return None
    
    def promote_model(self, version: str, stage: str = "Production") -> bool:
        """Promote model version to specified stage"""
        try:
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage=stage
            )
            print(f"✅ Model v{version} promoted to {stage}")
            return True
        except Exception as e:
            print(f"Error promoting model: {e}")
            return False
    
    def load_model(self, version: Optional[str] = None, stage: Optional[str] = "Production") -> Any:
        """Load model from registry"""
        try:
            if version:
                model_uri = f"models:/{self.model_name}/{version}"
            else:
                model_uri = f"models:/{self.model_name}/{stage}"
            
            model = mlflow.pytorch.load_model(model_uri)
            print(f"✅ Model loaded from registry: {model_uri}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

# Example usage and utilities
def create_model_signature(input_shape: tuple) -> mlflow.types.Schema:
    """Create MLflow model signature for 3D LDM"""
    from mlflow.types import Schema, TensorSpec
    import numpy as np
    
    input_schema = Schema([
        TensorSpec(np.dtype(np.float32), input_shape, name="input_volume")
    ])
    
    output_schema = Schema([
        TensorSpec(np.dtype(np.float32), input_shape, name="generated_volume")
    ])
    
    return mlflow.types.ModelSignature(
        inputs=input_schema,
        outputs=output_schema
    )

def log_training_run(tracker: MLflowTracker,
                    config: Dict[str, Any],
                    model: torch.nn.Module,
                    metrics: Dict[str, float],
                    figures_dir: Optional[str] = None,
                    model_path: str = "model"):
    """Complete training run logging"""
    
    # Start run
    run_name = f"3d_ldm_training_{config.get('experiment_id', 'default')}"
    tracker.start_run(run_name=run_name)
    
    try:
        # Log configuration
        tracker.log_config(config)
        
        # Log final metrics
        tracker.log_metrics(metrics)
        
        # Log figures if available
        if figures_dir and os.path.exists(figures_dir):
            tracker.log_artifacts(figures_dir)
        
        # Create model signature
        input_shape = tuple(config.get("patch_size", [64, 64, 64]))
        signature = create_model_signature((-1, 1) + input_shape)
        
        # Log model
        tracker.log_model(
            model=model,
            model_path=model_path,
            signature=signature
        )
        
        # Get model URI for registration
        model_uri = f"runs:/{tracker.run.info.run_id}/{model_path}"
        
        # Register model
        model_version = tracker.register_model(
            model_uri=model_uri,
            description=f"3D LDM trained with config: {config.get('experiment_id', 'default')}"
        )
        
        print(f"✅ Training run completed and logged: {run_name}")
        return model_version
        
    except Exception as e:
        print(f"Error during training run logging: {e}")
        raise
    finally:
        tracker.end_run()

if __name__ == "__main__":
    # Example usage
    tracker = MLflowTracker(
        experiment_name="3d-ldm-development",
        azure_ml_workspace="3d-ldm-workspace"
    )
    
    # Test connection
    print("MLflow tracker initialized successfully!")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {tracker.experiment_name}")