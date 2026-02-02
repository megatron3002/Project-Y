from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch
from torchvision import models
import numpy as np
from PIL import Image
import cv2

class XAIEngine:
    def __init__(self):
        print("Loading ResNet50 for Grad-CAM...")
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # We need the last convolutional layer
        # For ResNet50, it is layer4[-1]
        self.target_layers = [self.model.layer4[-1]]
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)

    def explain(self, img_pil: Image.Image, target_class_idx: int = None):
        """
        Generate Grad-CAM heatmap.
        Args:
            img_pil: PIL Image (will be resized to 224x224)
            target_class_idx: Index of the class to explain. If None, highest scoring class is used.
        Returns:
            heatmap_overlay: Numpy array (RGB) of the explanation
        """
        try:
            # Prepare image
            img_resized = img_pil.resize((224, 224), Image.LANCZOS)
            rgb_img = np.float32(img_resized) / 255.
            
            # Preprocess for model (channels first, normalize)
            preprocess = models.ResNet50_Weights.DEFAULT.transforms()
            input_tensor = preprocess(img_resized).unsqueeze(0)
            
            # Targets
            targets = [ClassifierOutputTarget(target_class_idx)] if target_class_idx is not None else None
            
            # Generate CAM
            # grayscale_cam shape: (batch_size, height, width)
            grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            # Overlay
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            return visualization
        except Exception as e:
            print(f"Error generating explanation: {e}")
            # Return original image if fail
            return np.array(img_pil)
