import faiss
import numpy as np
import pickle
import os

class VectorIndex:
    def __init__(self, index_file: str = "data/vector_index.bin", metadata_file: str = "data/metadata.pkl"):
        self.index_file = index_file
        self.metadata_file = metadata_file
        
        self.dimension = 2048 # ResNet50 feature size
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = [] # List of dicts, index corresponds to FAISS ID

    def build(self, features: np.ndarray, metadata_list: list):
        """
        Build the FAISS index from scratch.
        Args:
            features: Numpy array of shape (N, 2048)
            metadata_list: List of dictionaries containing image info
        """
        if len(features) != len(metadata_list):
            raise ValueError("Features and metadata must have the same length")
            
        print(f"Building index with {len(features)} items...")
        
        # FAISS expects float32
        features = features.astype('float32')
        
        self.index.reset()
        self.index.add(features)
        self.metadata = metadata_list
        
        self.save()

    def add(self, feature: np.ndarray, meta: dict):
        """Add a single item."""
        feature = feature.astype('float32').reshape(1, -1)
        self.index.add(feature)
        self.metadata.append(meta)

    def search(self, query_feature: np.ndarray, k: int = 5):
        """
        Search for top k similar items.
        Returns list of (metadata, distance) tuples.
        """
        query_feature = query_feature.astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_feature, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append((self.metadata[idx], float(dist)))
                
        return results

    def save(self):
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
        
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"Index saved to {self.index_file}")

    def load(self):
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.metadata_file, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"Loaded index with {self.index.ntotal} items.")
        else:
            print("No index found, starting fresh.")
