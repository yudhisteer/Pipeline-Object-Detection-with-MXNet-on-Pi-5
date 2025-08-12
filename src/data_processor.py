"""
Data processing utilities for train-test split and augmentation.
"""

import pandas as pd
import os
import shutil
import glob
import csv
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from typing import Tuple, List


from consts import VALIDATION_LABEL_PATH, TRAIN_LABEL_PATH, TRAIN_DATA_IMAGES, VALIDATION_DATA_IMAGES, CLASS_NAME


class DataProcessor:
    """Handles data processing including train-test split and augmentation."""
    
    def __init__(self, base_dir: str = None):
        """
        Initialize DataProcessor with base directory path.
        
        Args:
            base_dir: Base directory containing the dataset. Defaults to 'dataset'
        """
        if base_dir is None:
            base_dir = 'dataset'
        self.base_dir = base_dir
        self.train_images_dir = TRAIN_DATA_IMAGES
        self.val_images_dir = VALIDATION_DATA_IMAGES
        self.class_name = CLASS_NAME
        self.output_dir = os.path.join(base_dir, self.class_name)
    
    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load processed data from CSV files.
        
        Returns:
            Tuple of (train_df, validation_df)
        """
        df_train = pd.read_csv('output/processed_train_data.csv')
        df_validation = pd.read_csv('output/processed_validation_data.csv')
        
        print(f"Loaded {len(df_train)} training annotations")
        print(f"Loaded {len(df_validation)} validation annotations")
        
        return df_train, df_validation
    
    def create_train_test_split(self, df_train: pd.DataFrame, test_size: float = 0.2, 
                               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split training data into train and test sets.
        
        Args:
            df_train: Training DataFrame
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_split, test_split)
        """
        train, test = train_test_split(df_train, test_size=test_size, random_state=random_state)
        
        print(f"Train set: {len(train)} annotations")
        print(f"Test set: {len(test)} annotations")
        print(f"Total: {len(train) + len(test)} annotations")
        
        return train, test
    
    def setup_directories(self):
        """Create necessary directories for processed data."""
        os.makedirs(os.path.join(self.output_dir, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'images', 'test'), exist_ok=True)
        print("Created output directories")
    
    def copy_images_to_splits(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Copy images to train and test directories.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
        """
        train_ids = train_df["ImageID"].values.tolist()
        test_ids = test_df["ImageID"].values.tolist()
        
        # Source images pattern
        source_pattern = os.path.join(self.train_images_dir, '*.jpg')
        source_images = glob.glob(source_pattern)
        
        # Destination paths
        train_dest = os.path.join(self.output_dir, 'images', 'train')
        test_dest = os.path.join(self.output_dir, 'images', 'test')
        
        # Copy images
        for image_path in source_images:
            image_id = Path(image_path).stem
            
            if image_id in train_ids:
                shutil.copy(image_path, os.path.join(train_dest, f"{image_id}.jpg"))
            
            if image_id in test_ids:
                shutil.copy(image_path, os.path.join(test_dest, f"{image_id}.jpg"))
        
        # Count copied images
        train_count = len(glob.glob(os.path.join(train_dest, '*.jpg')))
        test_count = len(glob.glob(os.path.join(test_dest, '*.jpg')))
        
        print(f"Copied {train_count} training images")
        print(f"Copied {test_count} test images")
    
    def prepare_dataframes_for_export(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare DataFrames for export to list format.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Returns:
            Tuple of prepared DataFrames
        """
        # Create copies and rename columns
        final_train_df = train_df.copy()
        final_test_df = test_df.copy()
        
        final_train_df.rename(columns={"LabelName": "className"}, inplace=True)
        final_test_df.rename(columns={"LabelName": "className"}, inplace=True)
        
        # Add required columns
        final_train_df["header_cols"] = 2
        final_train_df["label_width"] = 5
        final_train_df["className"] = "0.000"  # Class ID for plastic bag
        
        final_test_df["header_cols"] = 2
        final_test_df["label_width"] = 5
        final_test_df["className"] = "0.000"  # Class ID for plastic bag
        
        # Add image paths
        final_train_df["ImagePath"] = final_train_df['ImageID'].apply(
            lambda x: f"{self.class_name}/images/train/{x}.jpg"
        )
        final_test_df["ImagePath"] = final_test_df['ImageID'].apply(
            lambda x: f"{self.class_name}/images/test/{x}.jpg"
        )
        
        # Reorder columns
        columns = ['header_cols', 'label_width', 'className', 'XMin', 'YMin', 'XMax', 'YMax', 'ImagePath']
        final_train_df = final_train_df[columns]
        final_test_df = final_test_df[columns]
        
        return final_train_df, final_test_df
    
    def create_list_file(self, df: pd.DataFrame, split: str):
        """
        Create .lst file for MXNet format.
        
        Args:
            df: DataFrame with annotations
            split: Either 'train' or 'test'
        """
        image_paths = df['ImagePath'].unique()
        final_data = []
        
        for idx, image_path in enumerate(image_paths):
            df_rows = df[df['ImagePath'] == image_path]
            
            # Start with index, header_cols, label_width
            row_data = [idx, 2, 5]
            
            # Add bounding boxes
            for _, annotation in df_rows.iterrows():
                row_data.extend([
                    "0.000",  # class ID
                    str(annotation['XMin']),
                    str(annotation['YMin']),
                    str(annotation['XMax']),
                    str(annotation['YMax'])
                ])
            
            # Add relative image path
            relative_path = f"{split}/{Path(image_path).stem}.jpg"
            row_data.append(relative_path)
            
            final_data.append(row_data)
        
        # Write to file
        output_file = f'{split}.lst'
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            for row in final_data:
                writer.writerow(row)
        
        print(f"Created {output_file} with {len(final_data)} entries")
    
    def augment_data(self, df: pd.DataFrame, split: str) -> pd.DataFrame:
        """
        Apply data augmentation (horizontal flip) to the dataset.
        
        Args:
            df: DataFrame with annotations
            split: Either 'train' or 'test'
            
        Returns:
            DataFrame with original and augmented data combined
        """
        images_path = os.path.join(self.output_dir, 'images', split)
        
        augmented_records = []
        
        for _, row in df.iterrows():
            # Get image info
            image_id = Path(row["ImagePath"]).stem
            image_path = os.path.join(images_path, f"{image_id}.jpg")
            
            if not os.path.exists(image_path):
                continue
            
            # Open and flip image
            img = Image.open(image_path)
            image_width, image_height = img.size
            img_flipped = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            
            # Save flipped image
            flipped_path = os.path.join(images_path, f"flipped_{image_id}.jpg")
            img_flipped.save(flipped_path)
            
            # Calculate new bounding box coordinates for flipped image
            xmin = row['XMin'] * image_width
            xmax = row['XMax'] * image_width
            
            # Flip coordinates: new_x = image_width - old_x
            new_xmin = (image_width - xmax) / image_width
            new_xmax = (image_width - xmin) / image_width
            
            # Create augmented record
            augmented_record = {
                'header_cols': 2,
                'label_width': 5,
                'className': '0.000',
                'XMin': new_xmin,
                'YMin': row['YMin'],  # Y coordinates don't change for horizontal flip
                'XMax': new_xmax,
                'YMax': row['YMax'],
                'ImagePath': f"{self.class_name}/images/{split}/flipped_{image_id}.jpg"
            }
            augmented_records.append(augmented_record)
        
        # Create DataFrame with augmented data
        augmented_df = pd.DataFrame(augmented_records)
        
        # Combine original and augmented data
        combined_df = pd.concat([df, augmented_df], ignore_index=True)
        
        print(f"Original {split} data: {len(df)} annotations")
        print(f"Augmented {split} data: {len(augmented_df)} annotations")
        print(f"Combined {split} data: {len(combined_df)} annotations")
        
        return combined_df
    
    def process_complete_dataset(self):
        """Complete processing pipeline."""
        print("Starting complete dataset processing...")
        
        # Load data
        df_train, df_validation = self.load_processed_data()
        
        # Create train-test split
        train_split, test_split = self.create_train_test_split(df_train)
        
        # Setup directories
        self.setup_directories()
        
        # Copy images
        self.copy_images_to_splits(train_split, test_split)
        
        # Prepare DataFrames
        final_train_df, final_test_df = self.prepare_dataframes_for_export(train_split, test_split)
        
        # Create initial list files
        self.create_list_file(final_train_df, 'train')
        self.create_list_file(final_test_df, 'test')
        
        # Apply data augmentation
        augmented_train_df = self.augment_data(final_train_df, 'train')
        augmented_test_df = self.augment_data(final_test_df, 'test')
        
        # Create final augmented list files
        augmented_train_df.to_csv(
            os.path.join(self.output_dir, 'train.lst'), 
            sep='\t', float_format='%.4f', header=None, index=False
        )
        augmented_test_df.to_csv(
            os.path.join(self.output_dir, 'test.lst'), 
            sep='\t', float_format='%.4f', header=None, index=False
        )
        
        # Move list files to output directory
        for filename in ['train.lst', 'test.lst']:
            if os.path.exists(filename):
                shutil.move(filename, os.path.join(self.output_dir, filename))
        
        print(f"Processing complete! Output saved to: {self.output_dir}")
        
        # Final statistics
        train_images = len(glob.glob(os.path.join(self.output_dir, 'images', 'train', '*.jpg')))
        test_images = len(glob.glob(os.path.join(self.output_dir, 'images', 'test', '*.jpg')))
        
        print(f"\nFinal dataset statistics:")
        print(f"Training images: {train_images}")
        print(f"Test images: {test_images}")
        print(f"Training annotations: {len(augmented_train_df)}")
        print(f"Test annotations: {len(augmented_test_df)}")


def main():
    """Main function to demonstrate data processing."""
    processor = DataProcessor()
    processor.process_complete_dataset()


if __name__ == "__main__":
    main()