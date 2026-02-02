import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

class FeatureExtractor:
    def __init__(self):
        # Load pre-trained ResNet50
        print("Loading ResNet50 model...")
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights)
        
        # Remove the classification head (fc layer)
        # ResNet50: The layer before fc is 'avgpool', outputting (2048, 1, 1)
        self.layer_name = 'avgpool'
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        
        self.model.eval()
        
        # Preprocessing transform
        self.preprocess = weights.transforms()

    def extract(self, img: Image.Image) -> np.ndarray:
        """
        Extract features from a PIL Image.
        Returns a numpy array of shape (2048,)
        """
        # Preprocess
        img_tensor = self.preprocess(img).unsqueeze(0) # Batch dimension
        
        with torch.no_grad():
            # Output shape: (1, 2048, 1, 1)
            feature_tensor = self.model(img_tensor)
        
        # Flatten to (2048,)
        feature_vector = feature_tensor.flatten().numpy()
        
        # L2 Normalize (good for cosine similarity/Euclidean distance in FAISS)
        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            feature_vector = feature_vector / norm
            
        return feature_vector
