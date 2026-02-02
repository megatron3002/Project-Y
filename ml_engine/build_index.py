import os
import glob
from pathlib import Path
from PIL import Image
from ml_engine.model import FeatureExtractor
from ml_engine.index import VectorIndex
import tqdm

def build_search_index(image_dir: str = "data/images", index_path: str = "data/vector_index.bin"):
    print("Initializing Feature Extractor...")
    extractor = FeatureExtractor()
    
    print("Initializing Vector Index...")
    index = VectorIndex(index_file=index_path)
    
    # Find all images
    image_paths = sorted(glob.glob(os.path.join(image_dir, "**", "*.jpg"), recursive=True))
    print(f"Found {len(image_paths)} images to index.")
    
    features = []
    metadata = []
    
    for path in tqdm.tqdm(image_paths):
        try:
            # Extract features
            img = Image.open(path).convert('RGB')
            feat = extractor.extract(img)
            
            # Metadata
            # path structure: data/images/class_name/filename.jpg
            parts = Path(path).parts
            class_name = parts[-2]
            filename = parts[-1]
            
            meta = {
                "path": path,
                "class": class_name,
                "filename": filename
            }
            
            features.append(feat)
            metadata.append(meta)
            
            if len(features) % 50 == 0:
                print(f"Processed {len(features)} images...")
                
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    if features:
        import numpy as np
        features_np = np.stack(features)
        index.build(features_np, metadata)
        print("Index build complete!")
    else:
        print("No features extracted.")

if __name__ == "__main__":
    build_search_index()
