"""
SageMaker training utilities for plastic bag detection model.
"""

import sagemaker
from sagemaker import get_execution_role, image_uris
from sagemaker.tuner import CategoricalParameter, ContinuousParameter, HyperparameterTuner
import os
from typing import Dict, Optional, Tuple
from dotenv import load_dotenv
import boto3

from consts import TRAIN_REC_PATH, VALIDATION_REC_PATH

load_dotenv()

class SageMakerTrainer:
    """Handles SageMaker model training for object detection."""
    
    def __init__(self, 
                 prefix: str = "plastic-bag-detection", 
                 bucket: str = os.getenv('BUCKET'), 
                 role_arn: Optional[str] = None, 
                 region: Optional[str] = None
                 ):
        """
        Initialize SageMaker trainer.
        
        Args:
            bucket: S3 bucket name for storing data and models
            prefix: S3 prefix for organizing data
            role_arn: SageMaker execution role ARN (required for local development)
            region: AWS region (defaults to session region)
        """
        self.bucket = bucket
        self.prefix = prefix
        self.sess = sagemaker.Session()
        self.region = region or self.sess.boto_region_name
        self.role_arn = role_arn or os.getenv('ROLE_ARN')
        
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
        
        # S3 paths
        self.s3_train_data = None
        self.s3_validation_data = None
        self.s3_output_location = f"s3://{bucket}/{prefix}/output"
        
        # Training components
        self.od_model = None
        self.tuner = None
        
        print(f"Initialized SageMaker trainer for bucket: {bucket}, prefix: {prefix}")
        print(f"Output location: {self.s3_output_location}")
    
    def upload_data_to_s3(self, 
                          train_rec_path: str = TRAIN_REC_PATH, 
                          test_rec_path: str = VALIDATION_REC_PATH
                          ) -> Tuple[str, str]:
        """
        Upload training and validation data to S3.
        
        Args:
            train_rec_path: Local path to training record file
            test_rec_path: Local path to test/validation record file
            
        Returns:
            Tuple of (train_s3_path, validation_s3_path)
        """
        if not os.path.exists(train_rec_path):
            raise FileNotFoundError(f"Training data not found: {train_rec_path}")
        if not os.path.exists(test_rec_path):
            raise FileNotFoundError(f"Test data not found: {test_rec_path}")
        
        # Upload training data
        train_channel = f"{self.prefix}/train"
        self.sess.upload_data(path=train_rec_path, bucket=self.bucket, key_prefix=train_channel)
        self.s3_train_data = f"s3://{self.bucket}/{train_channel}"
        
        # Upload validation data
        validation_channel = f"{self.prefix}/validation"
        self.sess.upload_data(path=test_rec_path, bucket=self.bucket, key_prefix=validation_channel)
        self.s3_validation_data = f"s3://{self.bucket}/{validation_channel}"
        
        print(f"Training data uploaded to: {self.s3_train_data}")
        print(f"Validation data uploaded to: {self.s3_validation_data}")
        
        return self.s3_train_data, self.s3_validation_data
    
    def create_estimator(self, 
                         instance_type: str = "ml.p3.2xlarge", 
                         instance_count: int = 1, 
                         volume_size: int = 50,
                         max_run: int = 360000
                         ) -> sagemaker.estimator.Estimator:
        """
        Create SageMaker estimator for object detection.
        
        Args:
            instance_type: EC2 instance type for training
            instance_count: Number of instances
            volume_size: EBS volume size in GB
            max_run: Maximum training time in seconds
            
        Returns:
            Configured SageMaker estimator
        """
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
    
    def set_hyperparameters(self, 
                            num_training_samples: int = 44, # number of images in plastic_bag/images/train
                            num_epochs: int = 100, 
                            lr_steps: str = "50,70,80,90,95",
                            **kwargs
                            ):
        """
        Set hyperparameters for object detection model.
        
        Args:
            num_training_samples: Number of training samples
            num_epochs: Number of training epochs
            lr_steps: Learning rate schedule steps
            **kwargs: Additional hyperparameters to override defaults
        """
        if self.od_model is None:
            raise ValueError("Must create estimator first using create_estimator()")
        
        # Default hyperparameters
        hyperparams = {
            "base_network": "resnet-50", # ResNet-50 backbone
            "use_pretrained_model": 1, # transfer learning | use pre-trained weights
            "num_classes": 1,  # Only plastic bags
            "epochs": num_epochs,
            "lr_scheduler_step": lr_steps,
            "lr_scheduler_factor": 0.1,
            "momentum": 0.9, # SGD momentum
            "weight_decay": 0.0005,
            "nms_threshold": 0.45,
            "image_shape": 512,
            "num_training_samples": num_training_samples
        }
        
        # Override with any provided kwargs
        hyperparams.update(kwargs)
        
        self.od_model.set_hyperparameters(**hyperparams)
        
        print("Set hyperparameters:")
        for key, value in hyperparams.items():
            print(f"  {key}: {value}")
    
    def create_hyperparameter_tuner(self, 
                                    max_jobs: int = 8,
                                    max_parallel_jobs: int = 1
                                    ) -> HyperparameterTuner:
        """
        Create hyperparameter tuner for automatic hyperparameter optimization.
        
        Args:
            max_jobs: Maximum number of tuning jobs
            max_parallel_jobs: Maximum parallel tuning jobs
            
        Returns:
            Configured hyperparameter tuner
        """
        if self.od_model is None:
            raise ValueError("Must create estimator first using create_estimator()")
        
        # Define hyperparameter ranges for tuning
        hyperparameter_ranges = {
            "learning_rate": ContinuousParameter(0.001, 0.1), # Learning rate range
            "mini_batch_size": CategoricalParameter([2, 4]), # Batch size options [8, 16, 32] if more images
            "optimizer": CategoricalParameter(["sgd", "adam"]) # Optimizer options
        }
        
        # Create tuner
        self.tuner = HyperparameterTuner(
            estimator=self.od_model,
            objective_metric_name="validation:mAP",
            hyperparameter_ranges=hyperparameter_ranges,
            objective_type="Maximize", # We want to maximize mAP
            max_jobs=max_jobs,
            max_parallel_jobs=max_parallel_jobs
        )
        
        print(f"Created hyperparameter tuner:")
        print(f"  Max jobs: {max_jobs}")
        print(f"  Max parallel jobs: {max_parallel_jobs}")
        print(f"  Objective: Maximize validation:mAP")
        
        return self.tuner
    
    def prepare_training_data(self) -> Dict[str, sagemaker.inputs.TrainingInput]:
        """
        Prepare training and validation data inputs.
        
        Returns:
            Dictionary of data channels for training
        """
        if not self.s3_train_data or not self.s3_validation_data:
            raise ValueError("Must upload data to S3 first using upload_data_to_s3()")
        
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
    
    def start_training(self, 
                       use_tuner: bool = True, 
                       wait: bool = True, 
                       logs: bool = True
                       ):
        """
        Start training job.
        
        Args:
            use_tuner: Whether to use hyperparameter tuning
            wait: Whether to wait for job completion
            logs: Whether to show training logs
        """
        data_channels = self.prepare_training_data()
        
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
    """Example usage of SageMaker trainer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train plastic bag detection model on SageMaker")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--prefix", required=True, help="S3 prefix")
    parser.add_argument("--train-data", default=TRAIN_REC_PATH, help="Path to training data")
    parser.add_argument("--test-data", default=VALIDATION_REC_PATH, help="Path to test data")
    parser.add_argument("--instance-type", default="ml.p3.2xlarge", help="Training instance type")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--use-tuner", action="store_true", help="Use hyperparameter tuning")
    parser.add_argument("--max-jobs", type=int, default=8, help="Max tuning jobs")
    parser.add_argument("--role-arn", help="SageMaker execution role ARN (required for local development)")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = SageMakerTrainer(bucket=args.bucket, prefix=args.prefix, role_arn=args.role_arn)
    
    # Upload data
    trainer.upload_data_to_s3(args.train_data, args.test_data)
    
    # Create estimator
    trainer.create_estimator(instance_type=args.instance_type)
    
    # Set hyperparameters
    trainer.set_hyperparameters(num_epochs=args.epochs)
    
    # Optionally create tuner
    if args.use_tuner:
        trainer.create_hyperparameter_tuner(max_jobs=args.max_jobs)
    
    # Start training
    trainer.start_training(use_tuner=args.use_tuner)
    
    # Get model artifacts
    model_path = trainer.get_model_artifacts()
    print(f"Training complete! Model saved to: {model_path}")


if __name__ == "__main__":
    main()

    """
    How to use:
    python sagemaker_trainer.py --bucket <bucket-name> --prefix <prefix> --instance-type <instance-type> --epochs <epochs> --use-tuner --max-jobs <max-jobs> --role-arn <role-arn>
    
    Example:
    python src/sagemaker/sagemaker_trainer.py --bucket cyudhist-pipeline-mxnet-503561429929 --prefix plastic-bag-detection --instance-type ml.p3.2xlarge --epochs 100 --use-tuner --max-jobs 8 --role-arn arn:aws:iam::503561429929:role/SageMakerExecutionRole-Pipeline-MXNet
    
    Note:
    - The train.rec and test.rec files should be in the same directory as the script.
    - The instance type should be a valid SageMaker instance type.
    - The role ARN is required when running locally (not on SageMaker notebook instance).
    """
    