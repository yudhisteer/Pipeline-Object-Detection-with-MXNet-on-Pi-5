"""
Data loading and preprocessing utilities for plastic bag detection dataset.
"""

import json
import pandas as pd
import os
import glob
from pathlib import Path
from typing import Tuple, List, Dict, Any

    
from utils import load_config, get_data_config


class DataLoader:
    """Handles loading and preprocessing of the plastic bag detection dataset."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize DataLoader with configuration.
        
        Args:
            config: Configuration dictionary from YAML file
        """
        self.config = config or load_config()
        data_config = get_data_config(self.config)
        
        self.base_dir = data_config.get('base_dir', 'dataset')
        self.train_json_path = data_config.get('train_labels_path', 'dataset/train/labels.json')
        self.val_json_path = data_config.get('validation_labels_path', 'dataset/validation/labels.json')
        self.train_images_dir = data_config.get('train_images_dir', 'dataset/train/data')
        self.val_images_dir = data_config.get('validation_images_dir', 'dataset/validation/data')
        self.target_class = data_config.get('target_class', 'Plastic bag')
    
    def load_filtered_annotations(self, json_path: str, target_class: str = None) -> pd.DataFrame:
        """
        Load and filter annotations for only the target class.
        
        Args:
            json_path: Path to the JSON annotations file
            target_class: Name of the target class to filter for (uses config if None)
            
        Returns:
            DataFrame with filtered annotations
        """
        if target_class is None:
            target_class = self.target_class
        with open(json_path) as f:
            data = json.load(f)
        
        images = {img['id']: img for img in data['images']}
        categories = {cat['id']: cat['name'] for cat in data['categories']}
        
        records = []
        for ann in data['annotations']:
            img = images[ann['image_id']]
            class_name = categories[ann['category_id']]
            
            if class_name == target_class: 
                records.append({
                    'ImageID': Path(img['file_name']).stem,
                    'LabelName': '/m/05gqfk',
                    'XMin': ann['bbox'][0] / img['width'],
                    'YMin': ann['bbox'][1] / img['height'],
                    'XMax': (ann['bbox'][0] + ann['bbox'][2]) / img['width'],
                    'YMax': (ann['bbox'][1] + ann['bbox'][3]) / img['height']
                })
        
        return pd.DataFrame(records)
    
    def get_image_ids(self, images_dir: str) -> List[str]:
        """
        Get list of image IDs from directory.
        
        Args:
            images_dir: Directory containing images
            
        Returns:
            List of image IDs (filenames without extension)
        """
        data_path = os.path.join(images_dir, '*.jpg')
        folder = glob.glob(data_path)
        img_ids = [Path(f).stem for f in folder]
        return img_ids
    
    def get_class_info(self, json_path: str) -> pd.DataFrame:
        """
        Load and analyze class information from JSON annotations.
        
        Args:
            json_path: Path to the JSON annotations file
            
        Returns:
            DataFrame with class information and counts
        """
        with open(json_path) as f:
            data = json.load(f)
        
        categories = {cat['id']: cat['name'] for cat in data['categories']}
        
        # Count occurrences of each class
        class_counts = {}
        for ann in data['annotations']:
            class_name = categories[ann['category_id']]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return pd.DataFrame({
            'className': list(categories.values()),
            'Count': [class_counts.get(name, 0) for name in categories.values()]
        })
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
        """
        Load all training and validation data.
        
        Returns:
            Tuple of (train_df, validation_df, train_image_ids, validation_image_ids)
        """
        print(f"Loading training data from: {self.train_json_path}")
        df_train = self.load_filtered_annotations(self.train_json_path)
        
        print(f"Loading validation data from: {self.val_json_path}")
        df_validation = self.load_filtered_annotations(self.val_json_path)
        
        # Get image IDs
        train_image_ids = self.get_image_ids(self.train_images_dir)
        validation_image_ids = self.get_image_ids(self.val_images_dir)
        
        print(f"\nData loaded successfully!")
        print(f"Found {len(df_train)} plastic bag training annotations")
        print(f"Found {len(df_validation)} plastic bag validation annotations")
        print(f"Found {len(train_image_ids)} training images")
        print(f"Found {len(validation_image_ids)} validation images")
        print(f"Average annotations per training image: {len(df_train)/len(train_image_ids):.2f}")
        
        return df_train, df_validation, train_image_ids, validation_image_ids
    
    def save_processed_data(self, df_train: pd.DataFrame, df_validation: pd.DataFrame, 
                          train_image_ids: List[str], validation_image_ids: List[str]):
        """
        Save processed data to files for later use.
        
        Args:
            df_train: Training DataFrame
            df_validation: Validation DataFrame  
            train_image_ids: List of training image IDs
            validation_image_ids: List of validation image IDs
        """
        data_config = get_data_config(self.config)
        output_dir = data_config.get('output_dir', 'output')
        os.makedirs(output_dir, exist_ok=True)
        df_train.to_csv(f'{output_dir}/processed_train_data.csv', index=False)
        df_validation.to_csv(f'{output_dir}/processed_validation_data.csv', index=False)
        
        print(f"Saving train image ids to {output_dir}/train_image_ids.txt")
        with open(f'{output_dir}/train_image_ids.txt', 'w') as f:
            f.write('\n'.join(train_image_ids))
            
        print(f"Saving validation image ids to {output_dir}/validation_image_ids.txt")
        with open(f'{output_dir}/validation_image_ids.txt', 'w') as f:
            f.write('\n'.join(validation_image_ids))
        
        print("Processed data saved successfully!")


def main():
    """Configuration-driven data loading."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and process plastic bag detection data")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    loader = DataLoader(config)
    
    # Load all data
    df_train, df_validation, train_ids, val_ids = loader.load_all_data()
    
    # Get class information
    train_classes = loader.get_class_info(loader.train_json_path)
    val_classes = loader.get_class_info(loader.val_json_path)
    
    print("\nTraining set class information:")
    print(train_classes)
    print("\nValidation set class information:")
    print(val_classes)
    print(f"Total training images: {len(train_ids)}")
    print(f"Total validation images: {len(val_ids)}")
    print("Train dataframe: ")
    print(df_train.head())
    print("Validation dataframe:")
    print(df_validation.head())
    
    # Save processed data
    loader.save_processed_data(df_train, df_validation, train_ids, val_ids)


if __name__ == "__main__":
    main()