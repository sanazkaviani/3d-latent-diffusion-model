"""
Azure ML Model Deployment Script
Handles deployment of 3D Latent Diffusion Model to Azure ML endpoints
"""

import os
import json
from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration
)
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
import yaml

class ModelDeployment:
    """Handle model deployment to Azure ML"""
    
    def __init__(self, config_path: str = None):
        """Initialize deployment manager"""
        
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent / "../azure/workspace_config.yml"
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize ML client
        credential = DefaultAzureCredential()
        self.ml_client = MLClient(
            credential=credential,
            subscription_id=self.config['subscription_id'],
            resource_group_name=self.config['resource_group'],
            workspace_name=self.config['workspace_name']
        )
        
        print(f"✅ Connected to Azure ML workspace: {self.config['workspace_name']}")
    
    def register_model(self, 
                      model_path: str,
                      model_name: str = "3d-ldm-model",
                      model_version: str = None,
                      description: str = None) -> Model:
        """Register model in Azure ML"""
        
        if description is None:
            description = "3D Latent Diffusion Model for medical image synthesis"
        
        # Create model
        model = Model(
            path=model_path,
            type=AssetTypes.CUSTOM_MODEL,
            name=model_name,
            description=description,
            version=model_version,
            tags={
                "framework": "pytorch",
                "domain": "medical_imaging",
                "architecture": "latent_diffusion"
            }
        )
        
        # Register model
        registered_model = self.ml_client.models.create_or_update(model)
        print(f"✅ Model registered: {registered_model.name} v{registered_model.version}")
        
        return registered_model
    
    def create_environment(self, 
                          env_name: str = "3d-ldm-inference-env",
                          conda_file: str = None) -> Environment:
        """Create deployment environment"""
        
        if conda_file is None:
            conda_file = Path(__file__).parent / "../azure/conda_env.yml"
        
        # Create environment
        environment = Environment(
            name=env_name,
            description="Environment for 3D LDM inference",
            conda_file=conda_file,
            image="mcr.microsoft.com/azureml/pytorch-2.0-ubuntu20.04-py38-cuda11.7-gpu:latest",
            tags={"purpose": "inference", "model": "3d-ldm"}
        )
        
        # Create or update environment
        env = self.ml_client.environments.create_or_update(environment)
        print(f"✅ Environment created: {env.name}")
        
        return env
    
    def create_endpoint(self, 
                       endpoint_name: str,
                       description: str = None) -> ManagedOnlineEndpoint:
        """Create managed online endpoint"""
        
        if description is None:
            description = "3D Latent Diffusion Model inference endpoint"
        
        # Create endpoint
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description=description,
            auth_mode="key",  # Use key-based authentication
            tags={
                "model": "3d-ldm",
                "purpose": "inference",
                "environment": "production"
            }
        )
        
        # Create endpoint
        endpoint = self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        print(f"✅ Endpoint created: {endpoint.name}")
        print(f"🔗 Endpoint URL: {endpoint.scoring_uri}")
        
        return endpoint
    
    def create_deployment(self,
                         endpoint_name: str,
                         deployment_name: str,
                         model_name: str,
                         model_version: str = None,
                         instance_type: str = "Standard_DS3_v2",
                         instance_count: int = 1,
                         environment_name: str = "3d-ldm-inference-env") -> ManagedOnlineDeployment:
        """Create model deployment"""
        
        # Get the latest model version if not specified
        if model_version is None:
            model = self.ml_client.models.get(model_name, label="latest")
            model_version = model.version
        
        # Create deployment
        deployment = ManagedOnlineDeployment(
            name=deployment_name,
            endpoint_name=endpoint_name,
            model=f"{model_name}:{model_version}",
            environment=environment_name,
            code_configuration=CodeConfiguration(
                code=Path(__file__).parent,
                scoring_script="score.py"
            ),
            instance_type=instance_type,
            instance_count=instance_count,
            request_settings={
                "request_timeout_ms": 300000,  # 5 minutes
                "max_concurrent_requests_per_instance": 1,
                "max_queue_wait_ms": 120000  # 2 minutes
            },
            liveness_probe={
                "failure_threshold": 3,
                "success_threshold": 1,
                "timeout": 30,
                "period": 10,
                "initial_delay": 10
            },
            readiness_probe={
                "failure_threshold": 3,
                "success_threshold": 1,
                "timeout": 30,
                "period": 10,
                "initial_delay": 10
            },
            tags={
                "model": model_name,
                "version": model_version,
                "environment": "production"
            }
        )
        
        # Create deployment
        deployment = self.ml_client.online_deployments.begin_create_or_update(deployment).result()
        print(f"✅ Deployment created: {deployment.name}")
        
        # Set traffic to 100% for this deployment
        endpoint = self.ml_client.online_endpoints.get(endpoint_name)
        endpoint.traffic = {deployment_name: 100}
        self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        
        print(f"✅ Traffic set to 100% for deployment: {deployment_name}")
        
        return deployment
    
    def deploy_model(self,
                    model_path: str,
                    endpoint_name: str,
                    deployment_name: str = "default",
                    model_name: str = "3d-ldm-model",
                    instance_type: str = "Standard_DS3_v2",
                    instance_count: int = 1) -> dict:
        """Complete model deployment workflow"""
        
        print(f"🚀 Starting deployment of {model_name}...")
        
        # Step 1: Register model
        model = self.register_model(
            model_path=model_path,
            model_name=model_name
        )
        
        # Step 2: Create environment
        environment = self.create_environment()
        
        # Step 3: Create endpoint
        endpoint = self.create_endpoint(
            endpoint_name=endpoint_name,
            description=f"Endpoint for {model_name}"
        )
        
        # Step 4: Create deployment
        deployment = self.create_deployment(
            endpoint_name=endpoint_name,
            deployment_name=deployment_name,
            model_name=model_name,
            model_version=model.version,
            instance_type=instance_type,
            instance_count=instance_count,
            environment_name=environment.name
        )
        
        # Get endpoint details
        endpoint_details = self.ml_client.online_endpoints.get(endpoint_name)
        
        deployment_info = {
            "endpoint_name": endpoint_name,
            "deployment_name": deployment_name,
            "model_name": model_name,
            "model_version": model.version,
            "scoring_uri": endpoint_details.scoring_uri,
            "swagger_uri": endpoint_details.openapi_uri,
            "instance_type": instance_type,
            "instance_count": instance_count
        }
        
        print("\n🎉 Deployment completed successfully!")
        print(f"📍 Endpoint: {endpoint_details.scoring_uri}")
        print(f"📋 Swagger: {endpoint_details.openapi_uri}")
        print(f"🔑 Use endpoint keys for authentication")
        
        return deployment_info
    
    def test_endpoint(self, endpoint_name: str, sample_data: dict = None):
        """Test the deployed endpoint"""
        
        if sample_data is None:
            sample_data = {
                "num_samples": 1,
                "inference_steps": 50,
                "output_format": "base64"
            }
        
        try:
            # Get endpoint
            endpoint = self.ml_client.online_endpoints.get(endpoint_name)
            
            # Test the endpoint
            response = self.ml_client.online_endpoints.invoke(
                endpoint_name=endpoint_name,
                request_file=None,
                deployment_name=None,
                request_data=json.dumps(sample_data)
            )
            
            print("✅ Endpoint test successful!")
            print(f"Response status: {response.get('status', 'unknown')}")
            
            return response
            
        except Exception as e:
            print(f"❌ Endpoint test failed: {str(e)}")
            return None
    
    def get_endpoint_info(self, endpoint_name: str) -> dict:
        """Get detailed endpoint information"""
        
        try:
            endpoint = self.ml_client.online_endpoints.get(endpoint_name)
            deployments = self.ml_client.online_deployments.list(endpoint_name)
            
            info = {
                "endpoint_name": endpoint.name,
                "scoring_uri": endpoint.scoring_uri,
                "swagger_uri": endpoint.openapi_uri,
                "auth_mode": endpoint.auth_mode,
                "provisioning_state": endpoint.provisioning_state,
                "traffic": endpoint.traffic,
                "deployments": []
            }
            
            for deployment in deployments:
                deployment_info = {
                    "name": deployment.name,
                    "model": deployment.model,
                    "instance_type": deployment.instance_type,
                    "instance_count": deployment.instance_count,
                    "provisioning_state": deployment.provisioning_state
                }
                info["deployments"].append(deployment_info)
            
            return info
            
        except Exception as e:
            print(f"Error getting endpoint info: {str(e)}")
            return None
    
    def delete_endpoint(self, endpoint_name: str):
        """Delete endpoint and all its deployments"""
        
        try:
            self.ml_client.online_endpoints.begin_delete(endpoint_name).result()
            print(f"✅ Endpoint deleted: {endpoint_name}")
        except Exception as e:
            print(f"Error deleting endpoint: {str(e)}")

def main():
    """Main function for deployment"""
    
    # Initialize deployment manager
    deployer = ModelDeployment()
    
    # Example deployment
    model_path = "./models/3d-ldm-model"  # Path to your trained model
    endpoint_name = "3d-ldm-endpoint"
    
    # Deploy model
    deployment_info = deployer.deploy_model(
        model_path=model_path,
        endpoint_name=endpoint_name,
        deployment_name="production",
        instance_type="Standard_DS3_v2",  # Adjust based on your needs
        instance_count=1
    )
    
    # Test endpoint
    deployer.test_endpoint(endpoint_name)
    
    print("\nDeployment completed! Your model is now available via REST API.")

if __name__ == "__main__":
    main()