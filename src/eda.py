import json
import os
import pandas as pd
import numpy as np
from pathlib import Path

from consts import TRAIN_LABEL_PATH, VALIDATION_LABEL_PATH


def coco_to_dataframes(data_path: str) -> pd.DataFrame:
    """
    Read COCO format JSON data and convert to DataFrame.
    
    Args:
        data_path: Path to JSON file
        
    Returns:
        pd.DataFrame: DataFrame containing the annotations data
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(f"Loaded COCO data from: {data_path}")
    
    # Convert annotations to DataFrame
    if 'annotations' in data and data['annotations']:
        df = pd.DataFrame(data['annotations'])
        # Convert bbox from list to separate columns
        if 'bbox' in df.columns:
            bbox_df = pd.DataFrame(df['bbox'].tolist(), 
                                 columns=['bbox_x', 'bbox_y', 'bbox_width', 'bbox_height'])
            df = pd.concat([df.drop('bbox', axis=1), bbox_df], axis=1)
        
        return df
    else:
        return pd.DataFrame()




if __name__ == "__main__":
    train_labels_df = coco_to_dataframes(TRAIN_LABEL_PATH)
    print("Train Labels DataFrame:")
    print(train_labels_df.head())
    print(f"Shape: {train_labels_df.shape}")
    print(train_labels_df.columns)
    
    print("\n" + "="*50)
    validation_labels_df = coco_to_dataframes(VALIDATION_LABEL_PATH)
    print("Validation Labels DataFrame:")
    print(validation_labels_df.head())
    print(f"Shape: {validation_labels_df.shape}")
    print(validation_labels_df.columns)
