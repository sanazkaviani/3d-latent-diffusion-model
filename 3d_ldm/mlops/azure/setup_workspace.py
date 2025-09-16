"""
Azure ML Workspace Setup Script
This script creates and configures Azure ML workspace with compute targets
"""

import os
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    Workspace, 
    AmlCompute, 
    Environment,
    BuildContext
)
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
import yaml

def load_config(config_path):
    """Load workspace configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def create_workspace(config):
    """Create Azure ML workspace"""
    try:
        # Get credential
        credential = DefaultAzureCredential()
        
        # Create ML client
        ml_client = MLClient(
            credential=credential,
            subscription_id=config['subscription_id'],
            resource_group_name=config['resource_group']
        )
        
        # Create workspace
        workspace = Workspace(
            name=config['workspace_name'],
            location=config['location'],
            display_name="3D Latent Diffusion Model Workspace",
            description="MLOps workspace for 3D LDM training and deployment",
            tags={"project": "3d-ldm", "environment": "production"}
        )
        
        print(f"Creating workspace: {config['workspace_name']}")
        workspace = ml_client.workspaces.begin_create(workspace).result()
        print(f"Workspace created successfully: {workspace.name}")
        
        return ml_client
        
    except Exception as e:
        print(f"Error creating workspace: {str(e)}")
        raise

def create_compute_targets(ml_client, config):
    """Create compute targets for training and inference"""
    
    # Training compute cluster (GPU)
    training_config = config['compute_targets']['training_cluster']
    training_compute = AmlCompute(
        name=training_config['name'],
        type="amlcompute",
        size=training_config['vm_size'],
        min_instances=training_config['min_nodes'],
        max_instances=training_config['max_nodes'],
        idle_time_before_scale_down=training_config['idle_seconds_before_scaledown'],
        description="GPU cluster for 3D LDM training",
        tags={"purpose": "training", "gpu": "true"}
    )
    
    print(f"Creating training compute: {training_config['name']}")
    ml_client.compute.begin_create_or_update(training_compute)
    
    # Inference compute cluster (CPU)
    inference_config = config['compute_targets']['inference_cluster']
    inference_compute = AmlCompute(
        name=inference_config['name'],
        type="amlcompute",
        size=inference_config['vm_size'],
        min_instances=inference_config['min_nodes'],
        max_instances=inference_config['max_nodes'],
        idle_time_before_scale_down=inference_config['idle_seconds_before_scaledown'],
        description="CPU cluster for 3D LDM inference",
        tags={"purpose": "inference", "gpu": "false"}
    )
    
    print(f"Creating inference compute: {inference_config['name']}")
    ml_client.compute.begin_create_or_update(inference_compute)
    
    print("Compute targets created successfully")

def create_environment(ml_client, config):
    """Create Azure ML environment for training and inference"""
    
    env_config = config['environment']
    
    # Create environment from conda file and dockerfile
    environment = Environment(
        name=env_config['name'],
        description="3D Latent Diffusion Model environment with PyTorch and MONAI",
        conda_file="./conda_env.yml",
        image=env_config['docker_base_image'],
        tags={"framework": "pytorch", "purpose": "3d-ldm"}
    )
    
    print(f"Creating environment: {env_config['name']}")
    ml_client.environments.create_or_update(environment)
    print("Environment created successfully")

def main():
    """Main function to set up Azure ML workspace"""
    
    # Load configuration
    config_path = "workspace_config.yml"
    config = load_config(config_path)
    
    print("Setting up Azure ML Workspace for 3D Latent Diffusion Model...")
    print(f"Workspace: {config['workspace_name']}")
    print(f"Resource Group: {config['resource_group']}")
    print(f"Location: {config['location']}")
    
    # Create workspace
    ml_client = create_workspace(config)
    
    # Create compute targets
    create_compute_targets(ml_client, config)
    
    # Create environment
    create_environment(ml_client, config)
    
    print("\n✅ Azure ML Workspace setup completed successfully!")
    print(f"📍 Workspace URL: https://ml.azure.com/workspaces/{config['workspace_name']}")
    print("\nNext steps:")
    print("1. Upload your training data to Azure storage")
    print("2. Run the training pipeline")
    print("3. Deploy the trained model")

if __name__ == "__main__":
    main()