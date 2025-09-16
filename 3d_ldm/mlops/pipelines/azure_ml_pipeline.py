"""
Azure ML Training Pipeline for 3D Latent Diffusion Model
Automated training pipeline with MLflow integration and model registration
"""

import os
import sys
from pathlib import Path
from azure.ai.ml import MLClient, Input, Output, command
from azure.ai.ml.entities import (
    Pipeline, 
    Job,
    Environment,
    Data
)
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential
import yaml

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

class AzureMLPipeline:
    """Azure ML Pipeline for 3D LDM training"""
    
    def __init__(self, config_path: str = None):
        """Initialize Azure ML pipeline"""
        
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent / "../mlops/azure/workspace_config.yml"
        
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
    
    def create_data_asset(self, 
                         data_path: str, 
                         name: str = "brats_dataset",
                         description: str = "BraTS dataset for 3D LDM training"):
        """Create data asset in Azure ML"""
        
        data_asset = Data(
            path=data_path,
            type=AssetTypes.URI_FOLDER,
            description=description,
            name=name
        )
        
        self.ml_client.data.create_or_update(data_asset)
        print(f"✅ Data asset created: {name}")
        return data_asset
    
    @pipeline(
        display_name="3D LDM Training Pipeline",
        description="Complete pipeline for training 3D Latent Diffusion Model"
    )
    def create_training_pipeline(self,
                               data_input: Input,
                               config_input: Input,
                               model_name: str = "3d-ldm",
                               experiment_name: str = "3d-ldm-training"):
        """Create the training pipeline"""
        
        # Environment for training
        env = Environment(
            name="3d-ldm-training-env",
            conda_file=Path(__file__).parent / "../azure/conda_env.yml",
            image="mcr.microsoft.com/azureml/pytorch-2.0-ubuntu20.04-py38-cuda11.7-gpu:latest"
        )
        
        # Step 1: Data preprocessing
        data_prep_step = command(
            name="data_preprocessing",
            display_name="Data Preprocessing",
            description="Preprocess BraTS data for training",
            code=Path(__file__).parent.parent,  # Root directory
            command="""
            python -m mlops.pipelines.data_preprocessing \\
                --data_path ${{inputs.data_input}} \\
                --config_path ${{inputs.config_input}} \\
                --output_path ${{outputs.processed_data}}
            """,
            environment=env,
            inputs={
                "data_input": data_input,
                "config_input": config_input
            },
            outputs={
                "processed_data": Output(
                    type=AssetTypes.URI_FOLDER,
                    mode=InputOutputModes.RW_MOUNT
                )
            },
            compute=self.config['compute_targets']['training_cluster']['name']
        )
        
        # Step 2: Autoencoder training
        autoencoder_step = command(
            name="autoencoder_training",
            display_name="Autoencoder Training",
            description="Train 3D autoencoder",
            code=Path(__file__).parent.parent,
            command="""
            python train_autoencoder.py \\
                --data_path ${{inputs.processed_data}} \\
                --config_path ${{inputs.config_input}} \\
                --output_path ${{outputs.autoencoder_model}} \\
                --mlflow_experiment_name {experiment_name} \\
                --azure_ml_workspace {workspace_name}
            """.format(
                experiment_name=experiment_name,
                workspace_name=self.config['workspace_name']
            ),
            environment=env,
            inputs={
                "processed_data": data_prep_step.outputs.processed_data,
                "config_input": config_input
            },
            outputs={
                "autoencoder_model": Output(
                    type=AssetTypes.CUSTOM_MODEL,
                    mode=InputOutputModes.RW_MOUNT
                )
            },
            compute=self.config['compute_targets']['training_cluster']['name']
        )
        
        # Step 3: Diffusion model training
        diffusion_step = command(
            name="diffusion_training",
            display_name="Diffusion Model Training", 
            description="Train diffusion model in latent space",
            code=Path(__file__).parent.parent,
            command="""
            python train_diffusion.py \\
                --data_path ${{inputs.processed_data}} \\
                --config_path ${{inputs.config_input}} \\
                --autoencoder_path ${{inputs.autoencoder_model}} \\
                --output_path ${{outputs.diffusion_model}} \\
                --mlflow_experiment_name {experiment_name} \\
                --azure_ml_workspace {workspace_name}
            """.format(
                experiment_name=experiment_name,
                workspace_name=self.config['workspace_name']
            ),
            environment=env,
            inputs={
                "processed_data": data_prep_step.outputs.processed_data,
                "config_input": config_input,
                "autoencoder_model": autoencoder_step.outputs.autoencoder_model
            },
            outputs={
                "diffusion_model": Output(
                    type=AssetTypes.CUSTOM_MODEL,
                    mode=InputOutputModes.RW_MOUNT
                )
            },
            compute=self.config['compute_targets']['training_cluster']['name']
        )
        
        # Step 4: Model evaluation and registration
        evaluation_step = command(
            name="model_evaluation",
            display_name="Model Evaluation and Registration",
            description="Evaluate model and register in MLflow",
            code=Path(__file__).parent.parent,
            command="""
            python -m mlops.pipelines.model_evaluation \\
                --autoencoder_path ${{inputs.autoencoder_model}} \\
                --diffusion_path ${{inputs.diffusion_model}} \\
                --test_data_path ${{inputs.processed_data}} \\
                --config_path ${{inputs.config_input}} \\
                --model_name {model_name} \\
                --mlflow_experiment_name {experiment_name}
            """.format(
                model_name=model_name,
                experiment_name=experiment_name
            ),
            environment=env,
            inputs={
                "autoencoder_model": autoencoder_step.outputs.autoencoder_model,
                "diffusion_model": diffusion_step.outputs.diffusion_model,
                "processed_data": data_prep_step.outputs.processed_data,
                "config_input": config_input
            },
            compute=self.config['compute_targets']['inference_cluster']['name']
        )
        
        return {
            "processed_data": data_prep_step.outputs.processed_data,
            "autoencoder_model": autoencoder_step.outputs.autoencoder_model,
            "diffusion_model": diffusion_step.outputs.diffusion_model
        }
    
    def submit_pipeline(self, 
                       data_path: str,
                       config_path: str,
                       model_name: str = "3d-ldm",
                       experiment_name: str = "3d-ldm-training"):
        """Submit the training pipeline"""
        
        # Create data inputs
        data_input = Input(
            type=AssetTypes.URI_FOLDER,
            path=data_path
        )
        
        config_input = Input(
            type=AssetTypes.URI_FILE,
            path=config_path
        )
        
        # Create pipeline
        pipeline_job = self.create_training_pipeline(
            data_input=data_input,
            config_input=config_input,
            model_name=model_name,
            experiment_name=experiment_name
        )
        
        # Submit pipeline
        pipeline_job = self.ml_client.jobs.create_or_update(
            pipeline_job,
            experiment_name=experiment_name
        )
        
        print(f"✅ Pipeline submitted: {pipeline_job.name}")
        print(f"🔗 Studio URL: {pipeline_job.studio_url}")
        
        return pipeline_job
    
    def create_batch_inference_pipeline(self):
        """Create pipeline for batch inference"""
        
        @pipeline(
            display_name="3D LDM Batch Inference",
            description="Batch inference pipeline for generating medical images"
        )
        def batch_inference_pipeline(
            model_input: Input,
            config_input: Input,
            num_samples: int = 10
        ):
            
            env = Environment(
                name="3d-ldm-inference-env",
                conda_file=Path(__file__).parent / "../azure/conda_env.yml",
                image="mcr.microsoft.com/azureml/pytorch-2.0-ubuntu20.04-py38-cuda11.7-gpu:latest"
            )
            
            inference_step = command(
                name="batch_inference",
                display_name="Batch Image Generation",
                description="Generate synthetic medical images",
                code=Path(__file__).parent.parent,
                command=f"""
                python inference.py \\
                    --model_path ${{inputs.model_input}} \\
                    --config_path ${{inputs.config_input}} \\
                    --num_samples {num_samples} \\
                    --output_path ${{outputs.generated_images}}
                """,
                environment=env,
                inputs={
                    "model_input": model_input,
                    "config_input": config_input
                },
                outputs={
                    "generated_images": Output(
                        type=AssetTypes.URI_FOLDER,
                        mode=InputOutputModes.RW_MOUNT
                    )
                },
                compute=self.config['compute_targets']['training_cluster']['name']
            )
            
            return {
                "generated_images": inference_step.outputs.generated_images
            }
        
        return batch_inference_pipeline

def main():
    """Main function to test pipeline creation"""
    
    # Initialize pipeline
    pipeline_manager = AzureMLPipeline()
    
    # Example data and config paths (adjust as needed)
    data_path = "./dataset"  # Local path or Azure storage path
    config_path = "./config/config_train_32g.json"
    
    print("Creating 3D LDM training pipeline...")
    
    # Submit pipeline
    job = pipeline_manager.submit_pipeline(
        data_path=data_path,
        config_path=config_path,
        model_name="3d-ldm-v1",
        experiment_name="3d-ldm-production"
    )
    
    print(f"Pipeline job created: {job.name}")
    print("Monitor the pipeline in Azure ML Studio")

if __name__ == "__main__":
    main()