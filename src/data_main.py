"""
Main pipeline for plastic bag detection dataset processing.
"""

import argparse
import sys
from src.data_loader import DataLoader
from src.visualizer import DataVisualizer
from src.data_processor import DataProcessor
from src.mxnet_converter import MXNetConverter


def run_data_loading():
    """Run data loading pipeline."""
    print("=" * 60)
    print("STEP 1: DATA LOADING")
    print("=" * 60)
    
    loader = DataLoader()
    df_train, df_validation, train_ids, val_ids = loader.load_all_data()
    
    # Get and display class information
    train_classes = loader.get_class_info(loader.train_json_path)
    val_classes = loader.get_class_info(loader.val_json_path)
    
    print("\nTraining set class information:")
    print(train_classes)
    print("\nValidation set class information:")
    print(val_classes)
    
    # Save processed data
    loader.save_processed_data(df_train, df_validation, train_ids, val_ids)
    
    return True


def run_visualization():
    """Run visualization pipeline."""
    print("\n" + "=" * 60)
    print("STEP 2: DATA VISUALIZATION")
    print("=" * 60)
    
    try:
        visualizer = DataVisualizer()
        visualizer.load_processed_data()
        
        # Show dataset statistics
        print("\nDataset Statistics:")
        visualizer.plot_dataset_statistics()
        
        # Show sample visualizations
        print("\nSample Visualizations:")
        print("- Random training image with bounding boxes")
        visualizer.visualize_with_bounding_boxes("train")
        
        print("- Random validation image with bounding boxes")
        visualizer.visualize_with_bounding_boxes("validation")
        
        return True
        
    except FileNotFoundError as e:
        print(f"Error: Required data files not found. Please run data loading first.")
        print(f"Details: {e}")
        return False


def run_data_processing():
    """Run data processing pipeline."""
    print("\n" + "=" * 60)
    print("STEP 3: DATA PROCESSING")
    print("=" * 60)
    
    try:
        processor = DataProcessor()
        processor.process_complete_dataset()
        return True
        
    except FileNotFoundError as e:
        print(f"Error: Required data files not found. Please run data loading first.")
        print(f"Details: {e}")
        return False


def run_mxnet_conversion():
    """Run MXNet conversion pipeline."""
    print("\n" + "=" * 60)
    print("STEP 4: MXNET CONVERSION")
    print("=" * 60)
    
    converter = MXNetConverter()
    success = converter.convert_all_splits()
    
    if success:
        print("\nMXNet conversion completed successfully!")
    else:
        print("\nMXNet conversion failed!")
    
    return success


def run_full_pipeline():
    """Run the complete processing pipeline."""
    print("Starting complete plastic bag detection dataset processing pipeline...")
    print("This will take several minutes to complete.")
    
    # Step 1: Data Loading
    if not run_data_loading():
        print("Pipeline failed at data loading step")
        return False
    
    # Step 2: Data Processing
    if not run_data_processing():
        print("Pipeline failed at data processing step")
        return False
    
    # Step 3: MXNet Conversion
    if not run_mxnet_conversion():
        print("Pipeline failed at MXNet conversion step")
        return False
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Your dataset is now ready for training with MXNet.")
    print("Generated files are located in: dataset/001.Plastic_bag/")
    
    return True


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Plastic Bag Detection Dataset Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    python main.py --full                 # Run complete pipeline
    python main.py --load                 # Load and preprocess data only
    python main.py --visualize            # Show data visualizations
    python main.py --process              # Process and augment data
    python main.py --convert              # Convert to MXNet format
            """
    )
    
    parser.add_argument(
        '--full', action='store_true',
        help='Run the complete processing pipeline'
    )
    parser.add_argument(
        '--load', action='store_true',
        help='Load and preprocess data only'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Show data visualizations'
    )
    parser.add_argument(
        '--process', action='store_true',
        help='Process data (train-test split and augmentation)'
    )
    parser.add_argument(
        '--convert', action='store_true',
        help='Convert to MXNet record format'
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return 0
    
    success = True
    
    try:
        if args.full:
            success = run_full_pipeline()
        else:
            if args.load:
                success = success and run_data_loading()
            
            if args.visualize:
                success = success and run_visualization()
            
            if args.process:
                success = success and run_data_processing()
            
            if args.convert:
                success = success and run_mxnet_conversion()
    
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return 1
    
    if success:
        print("\nAll requested operations completed successfully!")
        return 0
    else:
        print("\nSome operations failed. Please check the error messages above.")
        return 1


if __name__ == "__main__":
    exit(main())