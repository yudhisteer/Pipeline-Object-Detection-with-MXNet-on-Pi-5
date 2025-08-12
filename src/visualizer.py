"""
Visualization utilities for plastic bag detection dataset.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import pandas as pd
import random
import glob
import os
from PIL import Image
from pathlib import Path
from typing import List


from consts import TRAIN_DATA_IMAGES, VALIDATION_DATA_IMAGES



class DataVisualizer:
    """Handles visualization of the plastic bag detection dataset."""
    
    def __init__(self, base_dir: str = None):
        """
        Initialize DataVisualizer with base directory path.
        
        Args:
            base_dir: Base directory containing the dataset. Defaults to 'dataset'
        """
        if base_dir is None:
            base_dir = 'dataset'
        
        self.base_dir = base_dir
        self.train_images_path = TRAIN_DATA_IMAGES
        self.validation_images_path = VALIDATION_DATA_IMAGES
        os.makedirs('output', exist_ok=True)
    
    def load_processed_data(self):
        """Load processed data from CSV files."""
        self.df_train = pd.read_csv('output/processed_train_data.csv')
        self.df_validation = pd.read_csv('output/processed_validation_data.csv')
        
        with open('output/train_image_ids.txt', 'r') as f:
            self.train_list_ids = [line.strip() for line in f.readlines()]
            
        with open('output/validation_image_ids.txt', 'r') as f:
            self.validation_list_ids = [line.strip() for line in f.readlines()]
        
        print(f"Loaded {len(self.df_train)} training annotations")
        print(f"Loaded {len(self.df_validation)} validation annotations")
    
    def visualize_random_image(self, split: str):
        """
        Display a random image from the specified split.
        
        Args:
            split: Either 'train' or 'validation'
        """
        if split == 'train':
            images_path = self.train_images_path
        elif split == 'validation':
            images_path = self.validation_images_path
        else:
            raise ValueError("Split must be either 'train' or 'validation'")
        
        images_paths = glob.glob(os.path.join(images_path, '*.jpg'))
        if not images_paths:
            raise ValueError(f"No images found in {images_path}")
        
        random_image = random.choice(images_paths)
        img = mpimg.imread(random_image)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img)
        plt.title(f"Random {split} image: {Path(random_image).name}")
        plt.axis('off')
        print(f"Saving random image to output/{split}_random_image.png")
        plt.savefig(f'output/{split}_random_image.png')
    
    def visualize_with_bounding_boxes(self, dataset: str):
        """
        Visualize a random image with bounding boxes.
        
        Args:
            dataset: Either 'train' or 'validation'
        """
        if dataset == 'train':
            images_path = self.train_images_path
            df = self.df_train
        elif dataset == 'validation':
            images_path = self.validation_images_path
            df = self.df_validation
        else:
            raise ValueError("Dataset must be either 'train' or 'validation'")
        
        image_folder = os.path.join(images_path, '*.jpg')
        images_paths = glob.glob(image_folder)
        
        if not images_paths:
            raise ValueError(f"No images found in {images_path}")
        
        random_image = random.choice(images_paths)
        img = Image.open(random_image)
        id_of_image = Path(random_image).stem
        
        df_rows = df.loc[(df.ImageID == id_of_image) & (df.LabelName == '/m/05gqfk')]
        
        image_width, image_height = img.size
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img)
        
        for index, row in df_rows.iterrows():
            # Convert normalized coordinates to pixel coordinates
            xmin = row['XMin'] * image_width
            xmax = row['XMax'] * image_width
            ymin = row['YMin'] * image_height
            ymax = row['YMax'] * image_height
            
            width = xmax - xmin
            height = ymax - ymin
            
            rect = patches.Rectangle(
                (xmin, ymin), width, height, 
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
        
        plt.title(f"{dataset.capitalize()} image: {id_of_image} ({len(df_rows)} annotations)")
        plt.axis('off')
        print(f"Saving bounding boxes image to output/{dataset}_bounding_boxes.png")
        plt.savefig(f'output/{dataset}_bounding_boxes.png')
    
    def visualize_multiple_images(self, from_num: int, to_num: int, dataset: str):
        """
        Visualize multiple images with bounding boxes.
        
        Args:
            from_num: Starting index
            to_num: Ending index (exclusive)
            dataset: Either 'train' or 'validation'
        """
        if dataset == 'train':
            image_ids = self.train_list_ids
            images_path = self.train_images_path
            df = self.df_train
        elif dataset == 'validation':
            image_ids = self.validation_list_ids
            images_path = self.validation_images_path
            df = self.df_validation
        else:
            raise ValueError("Dataset must be 'train' or 'validation'")
        
        total_images = len(image_ids)
        print(f"Visualizing {dataset} set (has {total_images} total images)")
        
        if from_num >= total_images:
            print(f"Start number {from_num} >= total images {total_images}")
            return
        
        to_num = min(to_num, total_images)
        num_images = to_num - from_num
        
        columns = 4
        rows = (num_images + columns - 1) // columns
        
        fig = plt.figure(figsize=(15, 5*rows))
        plt.suptitle(f"{dataset.capitalize()} Images {from_num}-{to_num-1}", y=1.02)
        
        for i in range(from_num, to_num):
            try:
                img_id = image_ids[i]
                img_path = os.path.join(images_path, f"{img_id}.jpg")
                
                if not os.path.exists(img_path):
                    print(f"Image not found: {img_path}")
                    continue
                    
                img = mpimg.imread(img_path)
                ax = fig.add_subplot(rows, columns, i-from_num+1)
                ax.imshow(img)
                ax.set_title(f"ID: {img_id}")
                ax.axis('off')
                
                # Add bounding boxes
                boxes = df[df['ImageID'] == img_id]
                for _, row in boxes.iterrows():
                    xmin = row['XMin'] * img.shape[1]
                    xmax = row['XMax'] * img.shape[1]
                    ymin = row['YMin'] * img.shape[0]
                    ymax = row['YMax'] * img.shape[0]
                    
                    rect = plt.Rectangle(
                        (xmin, ymin), xmax-xmin, ymax-ymin,
                        linewidth=1, edgecolor='r', facecolor='none'
                    )
                    ax.add_patch(rect)
                    
            except Exception as e:
                print(f"Error processing image {i}: {str(e)}")
        
        plt.tight_layout()
        print(f"Saving multiple images to output/{dataset}_multiple_images.png")
        plt.savefig(f'output/{dataset}_multiple_images.png')
    
    def plot_dataset_statistics(self):
        """Plot dataset statistics."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Image counts
        axes[0].bar(['Training', 'Validation'], [len(self.train_list_ids), len(self.validation_list_ids)])
        axes[0].set_title('Number of Images')
        axes[0].set_ylabel('Count')
        
        # Annotation counts
        axes[1].bar(['Training', 'Validation'], [len(self.df_train), len(self.df_validation)])
        axes[1].set_title('Number of Annotations')
        axes[1].set_ylabel('Count')
        
        # Average annotations per image
        avg_train = len(self.df_train) / len(self.train_list_ids)
        avg_val = len(self.df_validation) / len(self.validation_list_ids)
        axes[2].bar(['Training', 'Validation'], [avg_train, avg_val])
        axes[2].set_title('Average Annotations per Image')
        axes[2].set_ylabel('Average Count')
        
        plt.tight_layout()
        print(f"Saving dataset statistics to output/dataset_statistics.png")
        plt.savefig('output/dataset_statistics.png')
        
        print(f"Dataset Summary:")
        print(f"Training: {len(self.train_list_ids)} images, {len(self.df_train)} annotations")
        print(f"Validation: {len(self.validation_list_ids)} images, {len(self.df_validation)} annotations")
        print(f"Average annotations per training image: {avg_train:.2f}")
        print(f"Average annotations per validation image: {avg_val:.2f}")


def main():
    """Main function to demonstrate visualization."""
    visualizer = DataVisualizer()
    visualizer.load_processed_data()
    
    # Show dataset statistics
    visualizer.plot_dataset_statistics()
    
    # Show random images
    visualizer.visualize_random_image("train")
    visualizer.visualize_random_image("validation")
    
    # Show images with bounding boxes
    visualizer.visualize_with_bounding_boxes("train")
    visualizer.visualize_with_bounding_boxes("validation")
    
    # Show multiple images
    visualizer.visualize_multiple_images(0, 8, "train")
    visualizer.visualize_multiple_images(0, 6, "validation")


if __name__ == "__main__":
    main()