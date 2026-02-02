import os
import shutil
from pathlib import Path
import torch
from torchvision import datasets, transforms
from PIL import Image

# Define class names for FashionMNIST
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Clean class names for file system
CLEAN_CLASS_NAMES = [name.replace('/', '_').replace(' ', '_') for name in CLASS_NAMES]

def ingest_data(output_dir: str = "data/images", num_images_per_class: int = 20):
    """
    Downloads FashionMNIST, selects a subset, converts to RGB, resizes to 224x224,
    and saves to disk organized by class.
    
    Args:
        output_dir: Directory to save images.
        num_images_per_class: Number of images to save per class.
    """
    print(f"Starting data ingestion to {output_dir}...")
    
    # Create output directory
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define transforms: Resize to 224x224 (for ResNet) and convert to Tensor
    # Note: We will convert back to PIL to save, but using transforms for resizing is convenient.
    # Actually, let's just use PIL for resizing to keep it simple and avoid unnecessary tensor conversions for saving.
    
    # Download dataset
    print("Downloading FashionMNIST...")
    dataset = datasets.FashionMNIST(root='./data_tmp', train=True, download=True)
    
    counts = {name: 0 for name in CLEAN_CLASS_NAMES}
    total_saved = 0
    
    print("Processing images...")
    for i, (image, label_idx) in enumerate(dataset):
        class_name = CLEAN_CLASS_NAMES[label_idx]
        
        if counts[class_name] >= num_images_per_class:
            continue
            
        # Create class directory
        class_dir = output_path / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Preprocessing:
        # 1. Convert to RGB (FashionMNIST is grayscale)
        image = image.convert("RGB")
        
        # 2. Resize to 224x224 (standard for ResNet50)
        image = image.resize((224, 224), Image.LANCZOS)
        
        # Save image
        # Naming convention: {class_name}_{index}.jpg
        file_name = f"{class_name}_{counts[class_name]}.jpg"
        image.save(class_dir / file_name)
        
        counts[class_name] += 1
        total_saved += 1
        
        # check if we are done
        if all(c >= num_images_per_class for c in counts.values()):
            break
            
    print(f"Data ingestion complete. Saved {total_saved} images to {output_dir}")
    
    # Cleanup temp download
    try:
        shutil.rmtree('./data_tmp')
    except Exception:
        pass

if __name__ == "__main__":
    ingest_data()
