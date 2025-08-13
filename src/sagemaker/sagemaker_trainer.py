"""
SageMaker training utilities for plastic bag detection model.
"""

import sagemaker
from sagemaker import get_execution_role, image_uris
from sagemaker.tuner import CategoricalParameter, ContinuousParameter, HyperparameterTuner
import os
import sys
from typing import Dict, Optional, Tuple, Any
from dotenv import load_dotenv
import boto3


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import (
    load_config,
    get_data_config,
    get_aws_config,
    get_training_config,
    get_hyperparameters_config,
    get_tuning_config,
    get_runtime_config,
)


load_dotenv()


class SageMakerTrainer:
    """Handles SageMaker model training for object detection."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SageMaker trainer with configuration.
        
        Args:
            config: Configuration dictionary loaded from YAML file
        """
        self.config = config or {}
        
        # Extract AWS configuration
        aws_config = get_aws_config(self.config)
        self.bucket = aws_config.get('bucket') or os.getenv('BUCKET')
        self.prefix = aws_config.get('prefix', 'plastic-bag-detection')
        self.role_arn = aws_config.get('role_arn') or os.getenv('ROLE_ARN')
        region = aws_config.get('region')
        
        if not self.bucket:
            raise ValueError("Bucket must be specified in config.yaml or BUCKET environment variable")
            
        self.sess = sagemaker.Session()
        self.region = region or self.sess.boto_region_name
        
        # Handle role for local development
        if self.role_arn:
            self.role = self.role_arn
        else:
            try:
                self.role = get_execution_role()
            except:
                raise ValueError(
                    "Must provide role_arn when running locally. "
                    f"Use: --role-arn {self.role_arn}"
                )
        
        # S3 paths - read directly from config
        data_config = get_data_config(self.config)
        self.s3_train_data = data_config.get('s3_train_path')
        self.s3_validation_data = data_config.get('s3_validation_path')
        self.s3_output_location = f"s3://{self.bucket}/{self.prefix}/output"
        
        # Validate S3 paths
        if not self.s3_train_data:
            raise ValueError("s3_train_path must be specified in config.yaml under 'data' section")
        if not self.s3_validation_data:
            raise ValueError("s3_validation_path must be specified in config.yaml under 'data' section")
        
        # Training components
        self.od_model = None
        self.tuner = None
        
        print(f"Initialized SageMaker trainer for bucket: {self.bucket}, prefix: {self.prefix}")
        print(f"Output location: {self.s3_output_location}")
        print(f"S3 data paths from config:")
        print(f"  Train: {self.s3_train_data}")
        print(f"  Validation: {self.s3_validation_data}")
    
    def create_estimator(self) -> sagemaker.estimator.Estimator:
        """
        Create SageMaker estimator for object detection.
        
        Returns:
            Configured SageMaker estimator
        """
        # Get training configuration
        training_config = get_training_config(self.config)
        instance_type = training_config.get('instance_type', 'ml.p3.2xlarge')
        instance_count = training_config.get('instance_count', 1)
        volume_size = training_config.get('volume_size', 50)
        max_run = training_config.get('max_run', 360000)
        # Get the object detection container image
        training_image = image_uris.retrieve(
            region=self.region,
            framework="object-detection", 
            version="1"
        )
        
        print(f"Using training image: {training_image}")
        
        # Create estimator
        self.od_model = sagemaker.estimator.Estimator(
            training_image,
            self.role,
            instance_count=instance_count,
            instance_type=instance_type,
            volume_size=volume_size,
            max_run=max_run,
            input_mode="File",
            output_path=self.s3_output_location,
            sagemaker_session=self.sess,
        )
        
        print(f"Created estimator with instance type: {instance_type}")
        return self.od_model
    
    def set_hyperparameters(self):
        """
        Set hyperparameters for object detection model using configuration.
        """
        if self.od_model is None:
            raise ValueError("Must create estimator first using create_estimator()")
        
        # Get hyperparameters from config
        hyperparams_config = get_hyperparameters_config(self.config)
        data_config = get_data_config(self.config)
        training_config = get_training_config(self.config)
        
        # Build hyperparameters from config with defaults
        hyperparams = {
            "base_network": hyperparams_config.get('base_network', 'resnet-50'),
            "use_pretrained_model": hyperparams_config.get('use_pretrained_model', 1),
            "num_classes": hyperparams_config.get('num_classes', 1),
            "epochs": training_config.get('epochs', 100),
            "lr_scheduler_step": hyperparams_config.get('lr_scheduler_step', '50,70,80,90,95'),
            "lr_scheduler_factor": hyperparams_config.get('lr_scheduler_factor', 0.1),
            "momentum": hyperparams_config.get('momentum', 0.9),
            "weight_decay": hyperparams_config.get('weight_decay', 0.0005),
            "nms_threshold": hyperparams_config.get('nms_threshold', 0.45),
            "image_shape": hyperparams_config.get('image_shape', 512),
            "num_training_samples": data_config.get('num_training_samples', 44)
        }
        
        self.od_model.set_hyperparameters(**hyperparams)
        
        print("Set hyperparameters:")
        for key, value in hyperparams.items():
            print(f"  {key}: {value}")
    
    def create_hyperparameter_tuner(self) -> HyperparameterTuner:
        """
        Create hyperparameter tuner for automatic hyperparameter optimization.
        
        Returns:
            Configured hyperparameter tuner
        """
        if self.od_model is None:
            raise ValueError("Must create estimator first using create_estimator()")
        
        # Get tuning configuration
        tuning_config = get_tuning_config(self.config)
        max_jobs = tuning_config.get('max_jobs', 8)
        max_parallel_jobs = tuning_config.get('max_parallel_jobs', 1)
        objective_metric = tuning_config.get('objective_metric', 'validation:mAP')
        objective_type = tuning_config.get('objective_type', 'Maximize')
        
        # Build hyperparameter ranges from config
        hyperparameter_ranges = {}
        ranges_config = tuning_config.get('hyperparameter_ranges', {})
        
        for param_name, param_config in ranges_config.items():
            if param_config['type'] == 'continuous':
                hyperparameter_ranges[param_name] = ContinuousParameter(
                    param_config['min'], param_config['max']
                )
            elif param_config['type'] == 'categorical':
                hyperparameter_ranges[param_name] = CategoricalParameter(
                    param_config['values']
                )
        
        # Create tuner
        self.tuner = HyperparameterTuner(
            estimator=self.od_model,
            objective_metric_name=objective_metric,
            hyperparameter_ranges=hyperparameter_ranges,
            objective_type=objective_type,
            max_jobs=max_jobs,
            max_parallel_jobs=max_parallel_jobs
        )
        
        print(f"Created hyperparameter tuner:")
        print(f"  Max jobs: {max_jobs}")
        print(f"  Max parallel jobs: {max_parallel_jobs}")
        print(f"  Objective: {objective_type} {objective_metric}")
        print(f"  Hyperparameter ranges: {list(hyperparameter_ranges.keys())}")
        
        return self.tuner
    
    def prepare_training_data(self) -> Dict[str, sagemaker.inputs.TrainingInput]:
        """
        Prepare training and validation data inputs.
        
        Returns:
            Dictionary of data channels for training
        """
        if not self.s3_train_data or not self.s3_validation_data:
            raise ValueError("S3 data paths not found. Please ensure s3_train_path and s3_validation_path are set in config.yaml")
        
        train_data = sagemaker.inputs.TrainingInput(
            self.s3_train_data,
            distribution="FullyReplicated",
            content_type="application/x-recordio",
            s3_data_type="S3Prefix"
        )
        
        validation_data = sagemaker.inputs.TrainingInput(
            self.s3_validation_data,
            distribution="FullyReplicated",
            content_type="application/x-recordio",
            s3_data_type="S3Prefix"
        )
        
        data_channels = {"train": train_data, "validation": validation_data}
        
        print("Prepared training data channels:")
        print(f"  Train: {self.s3_train_data}")
        print(f"  Validation: {self.s3_validation_data}")
        
        return data_channels
    
    def start_training(self):
        """
        Start training job using configuration settings.
        """
        data_channels = self.prepare_training_data()
        
        # Get runtime configuration
        runtime_config = get_runtime_config(self.config)
        tuning_config = get_tuning_config(self.config)
        wait = runtime_config.get('wait_for_completion', True)
        logs = runtime_config.get('show_logs', True)
        use_tuner = tuning_config.get('enabled', True)
        
        if use_tuner:
            if self.tuner is None:
                raise ValueError("Must create tuner first using create_hyperparameter_tuner()")
            
            print("Starting hyperparameter tuning job...")
            self.tuner.fit(inputs=data_channels, wait=wait, logs=logs)
            
            if wait:
                print("Hyperparameter tuning completed!")
                # Get best training job
                best_training_job = self.tuner.best_training_job()
                print(f"Best training job: {best_training_job}")
        else:
            if self.od_model is None:
                raise ValueError("Must create estimator first using create_estimator()")
            
            print("Starting training job...")
            self.od_model.fit(inputs=data_channels, wait=wait, logs=logs)
            
            if wait:
                print("Training completed!")
    
    def get_model_artifacts(self) -> str:
        """
        Get the S3 path to trained model artifacts.
        
        Returns:
            S3 path to model artifacts
        """
        if self.tuner and hasattr(self.tuner, 'best_training_job'):
            # Get best model from hyperparameter tuning
            best_job = self.tuner.best_training_job()
            model_data = self.sess.describe_training_job(best_job)['ModelArtifacts']['S3ModelArtifacts']
        elif self.od_model and hasattr(self.od_model, 'model_data'):
            # Get model from regular training
            model_data = self.od_model.model_data
        else:
            raise ValueError("No trained model found. Complete training first.")
        
        print(f"Model artifacts location: {model_data}")
        return model_data


def main():
    """Configuration-driven SageMaker training. Assumes data already uploaded to S3."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train plastic bag detection model on SageMaker")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    print("Note: This script assumes training data is already uploaded to S3!")
    
    # Create trainer with configuration
    trainer = SageMakerTrainer(config)
    
    # Create estimator
    print("Creating SageMaker estimator...")
    trainer.create_estimator()
    
    # Set hyperparameters
    print("Setting hyperparameters...")
    trainer.set_hyperparameters()
    
    # Create tuner if enabled
    tuning_config = get_tuning_config(config)
    if tuning_config.get('enabled', True):
        print("Creating hyperparameter tuner...")
        trainer.create_hyperparameter_tuner()
    
    # Start training
    print("Starting training job...")
    trainer.start_training()
    
    # Get model artifacts
    print("Getting model artifacts...")
    model_path = trainer.get_model_artifacts()
    print(f"Training complete! Model saved to: {model_path}")


if __name__ == "__main__":
    main()

    """
    Configuration-driven SageMaker training
    
    This script handles training only. Data upload is handled separately.
    
    Prerequisites:
    1. Upload training data to S3.
    2. Ensure config.yaml has correct AWS settings
    
    Usage:
    python src/sagemaker/sagemaker_trainer.py [--config CONFIG_FILE]
    
    Options:
      --config CONFIG_FILE       Configuration file path (default: config.yaml)
    """
    