"""
MXNet record file converter for the plastic bag detection dataset.
"""

import os
import subprocess
import sys
from pathlib import Path

# Fix NumPy compatibility issue with MXNet
try:
    import numpy as np
    if not hasattr(np, 'bool'):
        np.bool = bool
        np.int = int 
        np.float = float
        np.complex = complex
        np.object = object
        np.unicode = str
        np.str = str
except ImportError:
    pass

from consts import CLASS_NAME

class MXNetConverter:
    """Handles conversion of data to MXNet record format."""
    
    def __init__(self, base_dir: str = None, resize_size: int = 256):
        """
        Initialize MXNetConverter.
        
        Args:
            base_dir: Base directory containing the dataset. Defaults to 'dataset'
            resize_size: Size to resize images to
        """
        if base_dir is None:
            base_dir = 'dataset'
        
        self.base_dir = base_dir
        self.class_name = CLASS_NAME
        self.output_dir = os.path.join(base_dir, self.class_name)
        self.resize_size = resize_size
        self.tools_dir = 'tools'
    
    def install_dependencies(self):
        """Install required dependencies for MXNet conversion."""
        print("Installing required dependencies...")
        
        dependencies = [
            'opencv-python',
            'mxnet',
            'distro'
        ]
        
        for dep in dependencies:
            try:
                subprocess.check_call([sys.executable, '-m', 'uv', 'pip', 'install', dep])
                print(f"Successfully installed {dep}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {dep}: {e}")
                return False
        
        # Install system dependencies for Debian-based systems
        try:
            import distro
            if distro.id() == "debian":
                print("Installing system dependencies for Debian...")
                os.system("apt-get update")
                os.system("apt-get install ffmpeg libsm6 libxext6 -y")
        except ImportError:
            print("Could not check system type, skipping system dependencies")
        
        return True
    
    def verify_files_exist(self):
        """Verify that required files exist before conversion."""
        # Get absolute path to tools directory relative to project root
        current_dir = os.getcwd()
        tools_path = os.path.join(current_dir, self.tools_dir)
        
        required_files = [
            os.path.join(self.output_dir, 'train.lst'),
            os.path.join(self.output_dir, 'test.lst'),
            os.path.join(self.output_dir, 'images', 'train'),
            os.path.join(self.output_dir, 'images', 'test'),
            os.path.join(tools_path, 'im2rec.py')
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print("Missing required files:")
            for file_path in missing_files:
                print(f"  - {file_path}")
            return False
        
        print("All required files found")
        return True
    
    def convert_to_record_format(self, split: str):
        """
        Convert dataset split to MXNet record format.
        
        Args:
            split: Either 'train' or 'test'
        """
        print(f"Converting {split} split to MXNet record format...")
        
        # Change to output directory for conversion
        original_dir = os.getcwd()
        os.chdir(self.output_dir)
        
        try:
            # Build command - use absolute path for im2rec.py script
            im2rec_script = os.path.join(original_dir, self.tools_dir, 'im2rec.py')
            cmd = [
                'python3', im2rec_script,
                '--resize', str(self.resize_size),
                '--pack-label',
                f'{split}.lst',
                'images/'
            ]
            
            print(f"Running command: {' '.join(cmd)}")
            
            # Execute conversion
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Successfully converted {split} split")
                if result.stdout:
                    print("Output:", result.stdout)
            else:
                print(f"Error converting {split} split:")
                print("Error:", result.stderr)
                return False
                
        except Exception as e:
            print(f"Exception during conversion: {e}")
            return False
        finally:
            # Return to original directory
            os.chdir(original_dir)
        
        return True
    
    def convert_all_splits(self):
        """Convert all dataset splits to MXNet record format."""
        print("Starting MXNet record conversion...")
        
        # Verify dependencies and files
        if not self.install_dependencies():
            print("Failed to install dependencies")
            return False
        
        if not self.verify_files_exist():
            print("Required files missing")
            return False
        
        # Convert train split
        if not self.convert_to_record_format('train'):
            print("Failed to convert train split")
            return False
        
        # Convert test split
        if not self.convert_to_record_format('test'):
            print("Failed to convert test split")
            return False
        
        print("Successfully converted all splits to MXNet record format")
        
        # List generated files
        record_files = [
            os.path.join(self.output_dir, 'train.rec'),
            os.path.join(self.output_dir, 'train.idx'),
            os.path.join(self.output_dir, 'test.rec'),
            os.path.join(self.output_dir, 'test.idx')
        ]
        
        print("\nGenerated record files:")
        for file_path in record_files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f"  - {file_path} ({size:,} bytes)")
            else:
                print(f"  - {file_path} (NOT FOUND)")
        
        return True


def main():
    """Main function to demonstrate MXNet conversion."""
    converter = MXNetConverter()
    
    success = converter.convert_all_splits()
    
    if success:
        print("\nMXNet conversion completed successfully!")
        print(f"Record files are available in: {converter.output_dir}")
    else:
        print("\nMXNet conversion failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())