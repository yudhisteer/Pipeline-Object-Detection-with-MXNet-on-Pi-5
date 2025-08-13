"""
SageMaker data management utilities for plastic bag detection model.
Handles S3 uploads, data preparation, and S3 path management.
"""

import sagemaker
import boto3
import os
import sys
import argparse
from typing import Dict, Optional, Tuple, Any
from dotenv import load_dotenv
from botocore.exceptions import ClientError

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import (
    load_config,
    get_data_config,
    get_aws_config,
)

load_dotenv()


class SageMakerDataManager:
    """Handles S3 data operations for SageMaker training."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SageMaker data manager with configuration.
        
        Args:
            config: Configuration dictionary loaded from YAML file
        """
        self.config = config or {}
        
        # Extract AWS configuration
        aws_config = get_aws_config(self.config)
        self.bucket = aws_config.get('bucket') or os.getenv('BUCKET')
        self.prefix = aws_config.get('prefix', 'plastic-bag-detection')
        region = aws_config.get('region')
        
        if not self.bucket:
            raise ValueError("Bucket must be specified in config.yaml or BUCKET environment variable")
            
        self.sess = sagemaker.Session()
        self.region = region or self.sess.boto_region_name
        self.s3_client = boto3.client('s3', region_name=self.region)
        
        # S3 paths
        self.s3_train_data = None
        self.s3_validation_data = None
        
        print(f"Initialized SageMaker data manager for bucket: {self.bucket}, prefix: {self.prefix}")
    
    def _s3_object_exists(self, s3_path: str) -> bool:
        """
        Check if an S3 object exists.
        
        Args:
            s3_path: Full S3 path (s3://bucket/key)
            
        Returns:
            True if object exists, False otherwise
        """
        # Parse S3 path
        if not s3_path.startswith('s3://'):
            return False
            
        path_parts = s3_path[5:].split('/', 1)
        bucket = path_parts[0]
        key = path_parts[1] if len(path_parts) > 1 else ''
        
        try:
            # List objects with the prefix to check if any exist
            response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
            return 'Contents' in response and len(response['Contents']) > 0
        except ClientError:
            return False
    
    def _get_local_file_size(self, file_path: str) -> int:
        """Get the size of a local file in bytes."""
        try:
            return os.path.getsize(file_path)
        except OSError:
            return 0
    
    def _get_s3_object_size(self, s3_path: str) -> int:
        """Get the size of an S3 object in bytes."""
        if not s3_path.startswith('s3://'):
            return 0
            
        path_parts = s3_path[5:].split('/', 1)
        bucket = path_parts[0]
        key = path_parts[1] if len(path_parts) > 1 else ''
        
        try:
            response = self.s3_client.head_object(Bucket=bucket, Key=key)
            return response['ContentLength']
        except ClientError:
            return 0
    
    def check_data_in_s3(self) -> Tuple[bool, bool, Dict[str, str]]:
        """
        Check if training and validation data already exist in S3.
        
        Returns:
            Tuple of (train_exists, validation_exists, s3_paths)
            where s3_paths is a dict with 'train' and 'validation' keys
        """
        # Get S3 paths from config
        data_config = get_data_config(self.config)
        expected_train_s3 = data_config.get('s3_train_path')
        expected_validation_s3 = data_config.get('s3_validation_path')
        
        if not expected_train_s3 or not expected_validation_s3:
            print("Warning: S3 paths not found in config. Using default paths.")
            # Fallback to constructed paths
            train_rec_path = data_config.get('train_path')
            test_rec_path = data_config.get('validation_path')
            
            if train_rec_path and test_rec_path:
                train_filename = os.path.basename(train_rec_path)
                test_filename = os.path.basename(test_rec_path)
                expected_train_s3 = f"s3://{self.bucket}/{self.prefix}/train/{train_filename}"
                expected_validation_s3 = f"s3://{self.bucket}/{self.prefix}/validation/{test_filename}"
            else:
                return False, False, {}
        
        # Check if files exist in S3
        train_exists = self._s3_object_exists(expected_train_s3)
        validation_exists = self._s3_object_exists(expected_validation_s3)
        
        # Also check if local and S3 file sizes match (basic integrity check)
        data_config = get_data_config(self.config)
        train_rec_path = data_config.get('train_path')
        test_rec_path = data_config.get('validation_path')
        
        if train_exists and train_rec_path and os.path.exists(train_rec_path):
            local_size = self._get_local_file_size(train_rec_path)
            s3_size = self._get_s3_object_size(expected_train_s3)
            if local_size != s3_size:
                print(f"Warning: Local training file size ({local_size}) differs from S3 ({s3_size})")
                train_exists = False
        
        if validation_exists and test_rec_path and os.path.exists(test_rec_path):
            local_size = self._get_local_file_size(test_rec_path)
            s3_size = self._get_s3_object_size(expected_validation_s3)
            if local_size != s3_size:
                print(f"Warning: Local validation file size ({local_size}) differs from S3 ({s3_size})")
                validation_exists = False
        
        s3_paths = {
            'train': expected_train_s3 if train_exists else None,
            'validation': expected_validation_s3 if validation_exists else None
        }
        
        print(f"S3 data check results:")
        print(f"  Training data exists: {train_exists} ({expected_train_s3})")
        print(f"  Validation data exists: {validation_exists} ({expected_validation_s3})")
        
        return train_exists, validation_exists, s3_paths
    
    def upload_data_to_s3(self, force_upload: bool = False) -> Tuple[str, str]:
        """
        Upload training and validation data to S3.
        
        Args:
            force_upload: If True, upload even if data already exists in S3
            
        Returns:
            Tuple of (train_s3_path, validation_s3_path)
        """
        # Check if data already exists in S3
        train_exists, validation_exists, existing_s3_paths = self.check_data_in_s3()
        
        if not force_upload and train_exists and validation_exists:
            print("Data already exists in S3. Skipping upload.")
            self.s3_train_data = existing_s3_paths['train']
            self.s3_validation_data = existing_s3_paths['validation']
            return self.s3_train_data, self.s3_validation_data
        
        # Get data paths from config
        data_config = get_data_config(self.config)
        train_rec_path = data_config.get('train_path')
        test_rec_path = data_config.get('validation_path')
        
        if not train_rec_path:
            raise ValueError("train_path must be specified in config.yaml under 'data' section")
        if not test_rec_path:
            raise ValueError("validation_path must be specified in config.yaml under 'data' section")
        if not os.path.exists(train_rec_path):
            raise FileNotFoundError(f"Training data not found: {train_rec_path}")
        if not os.path.exists(test_rec_path):
            raise FileNotFoundError(f"Test data not found: {test_rec_path}")
        
        # Get expected S3 paths from config or construct them
        data_config = get_data_config(self.config)
        expected_train_s3 = data_config.get('s3_train_path')
        expected_validation_s3 = data_config.get('s3_validation_path')
        
        # If not in config, construct them based on local file paths
        if not expected_train_s3:
            train_filename = os.path.basename(train_rec_path)
            expected_train_s3 = f"s3://{self.bucket}/{self.prefix}/train/{train_filename}"
        
        if not expected_validation_s3:
            test_filename = os.path.basename(test_rec_path)
            expected_validation_s3 = f"s3://{self.bucket}/{self.prefix}/validation/{test_filename}"
        
        # Upload training data if needed
        if force_upload or not train_exists:
            print(f"Uploading training data: {train_rec_path}")
            train_channel = f"{self.prefix}/train"
            self.sess.upload_data(path=train_rec_path, bucket=self.bucket, key_prefix=train_channel)
            self.s3_train_data = expected_train_s3
            print(f"Training data uploaded to: {self.s3_train_data}")
        else:
            self.s3_train_data = existing_s3_paths['train']
            print(f"Using existing training data: {self.s3_train_data}")
        
        # Upload validation data if needed
        if force_upload or not validation_exists:
            print(f"Uploading validation data: {test_rec_path}")
            validation_channel = f"{self.prefix}/validation"
            self.sess.upload_data(path=test_rec_path, bucket=self.bucket, key_prefix=validation_channel)
            self.s3_validation_data = expected_validation_s3
            print(f"Validation data uploaded to: {self.s3_validation_data}")
        else:
            self.s3_validation_data = existing_s3_paths['validation']
            print(f"Using existing validation data: {self.s3_validation_data}")
        
        # Print reminder about config
        if not data_config.get('s3_train_path') or not data_config.get('s3_validation_path'):
            print(f"\n💡 Consider updating your config.yaml with these S3 paths:")
            print(f"   s3_train_path: \"{self.s3_train_data}\"")
            print(f"   s3_validation_path: \"{self.s3_validation_data}\"")
        
        return self.s3_train_data, self.s3_validation_data
    
    def get_s3_data_paths(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Get S3 paths for training and validation data.
        
        Returns:
            Tuple of (train_s3_path, validation_s3_path)
        """
        return self.s3_train_data, self.s3_validation_data
    
    def set_s3_paths(self, train_s3_path: str, validation_s3_path: str):
        """
        Manually set S3 paths for training and validation data.
        Useful when data is already uploaded by external processes.
        
        Args:
            train_s3_path: S3 path to training data
            validation_s3_path: S3 path to validation data
        """
        self.s3_train_data = train_s3_path
        self.s3_validation_data = validation_s3_path
        
        print(f"Set S3 data paths:")
        print(f"  Train: {self.s3_train_data}")
        print(f"  Validation: {self.s3_validation_data}")


def main():
    """Standalone data upload utility."""
    
    parser = argparse.ArgumentParser(description="Upload plastic bag detection data to S3")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--force", action="store_true", help="Force upload even if data exists in S3")
    parser.add_argument("--check-only", action="store_true", help="Only check if data exists, don't upload")
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Create data manager
    data_manager = SageMakerDataManager(config)
    
    if args.check_only:
        print("Checking data in S3...")
        train_exists, validation_exists, s3_paths = data_manager.check_data_in_s3()
        
        if train_exists and validation_exists:
            print("All data exists in S3")
            print(f"Train: {s3_paths['train']}")
            print(f"Validation: {s3_paths['validation']}")
        else:
            print("Some data missing from S3")
            if not train_exists:
                print("Missing: Training data")   
            if not validation_exists:
                print("Missing: Validation data")
    else:
        print("Uploading data to S3...")
        train_s3, validation_s3 = data_manager.upload_data_to_s3(force_upload=args.force)
        print(f"Upload complete!")
        print(f"Train: {train_s3}")
        print(f"Validation: {validation_s3}")


if __name__ == "__main__":
    main()
