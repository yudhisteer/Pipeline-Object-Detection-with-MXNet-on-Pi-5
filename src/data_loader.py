"""
Data loading and preprocessing utilities for plastic bag detection dataset.
"""

import json
import pandas as pd
import os
import glob
from pathlib import Path
from typing import Tuple, List


from consts import VALIDATION_LABEL_PATH, TRAIN_LABEL_PATH, TRAIN_DATA_IMAGES, VALIDATION_DATA_IMAGES


class DataLoader:
    """Handles loading and preprocessing of the plastic bag detection dataset."""
    
    def __init__(self, base_dir: str = None):
        """
        Initialize DataLoader with base directory path.
        
        Args:
            base_dir: Base directory containing the dataset. Defaults to 'dataset'
        """
        if base_dir is None:
            base_dir = 'dataset'
        
        self.base_dir = base_dir
        self.train_json_path = TRAIN_LABEL_PATH
        self.val_json_path = VALIDATION_LABEL_PATH
        self.train_images_dir = TRAIN_DATA_IMAGES
        self.val_images_dir = VALIDATION_DATA_IMAGES
    
    def load_filtered_annotations(self, json_path: str, target_class: str = 'Plastic bag') -> pd.DataFrame:
        """
        Load and filter annotations for only the target class.
        
        Args:
            json_path: Path to the JSON annotations file
            target_class: Name of the target class to filter for
            
        Returns:
            DataFrame with filtered annotations
        """
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
        os.makedirs('output', exist_ok=True)
        df_train.to_csv('output/processed_train_data.csv', index=False)
        df_validation.to_csv('output/processed_validation_data.csv', index=False)
        
        print(f"Saving train image ids to output/train_image_ids.txt")
        with open('output/train_image_ids.txt', 'w') as f:
            f.write('\n'.join(train_image_ids))
            
        print(f"Saving validation image ids to output/validation_image_ids.txt")
        with open('output/validation_image_ids.txt', 'w') as f:
            f.write('\n'.join(validation_image_ids))
        
        print("Processed data saved successfully!")


def main():
    """Main function to demonstrate data loading."""
    loader = DataLoader()
    
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