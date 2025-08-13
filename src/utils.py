"""
Utility functions for the plastic bag detection pipeline.
"""

import os
import yaml
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        if config is None:
            raise ValueError(f"Configuration file is empty: {config_path}")
            
        return config
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")


def get_data_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract data configuration section with defaults.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Data configuration with default values
    """
    return config.get('data', {})


def get_aws_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract AWS configuration section.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        AWS configuration dictionary
    """
    return config.get('aws', {})


def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract training configuration section.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Training configuration dictionary
    """
    return config.get('training', {})


def get_hyperparameters_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract hyperparameters configuration section.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Hyperparameters configuration dictionary
    """
    return config.get('hyperparameters', {})


def get_tuning_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract tuning configuration section.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Tuning configuration dictionary
    """
    return config.get('tuning', {})


def get_runtime_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract runtime configuration section.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Runtime configuration dictionary
    """
    return config.get('runtime', {})


def get_inference_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract inference configuration section.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Inference configuration dictionary
    """
    return config.get('inference', {})
