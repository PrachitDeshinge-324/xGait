"""
Person Re-Identification (ReID) models for appearance-based tracking
"""
import sys
import os
from pathlib import Path

# Add parent directory to path for utils and config imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from typing import List, Tuple, Optional

from utils.device_utils import DeviceManager, tensor_to_numpy
from config import get_device_config

class ReIDFeatureExtractor(nn.Module):
    """
    CNN-based feature extractor for person re-identification
    """
    def __init__(self, feature_dim: int = 256):
        super(ReIDFeatureExtractor, self).__init__()
        self.feature_dim = feature_dim
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Second conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            
            # Fully connected layers
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, feature_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

class ReIDModel:
    """
    Main ReID model wrapper for person re-identification with device-aware optimization
    """
    def __init__(self, device: str = "cpu", feature_dim: int = 256):
        self.device = device
        self.feature_dim = feature_dim
        
        # Get device-specific configuration
        self.device_config = get_device_config(device)
        self.device_manager = DeviceManager(device, self.device_config["dtype"])
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),  # Standard ReID input size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load feature extractor
        self.model = ReIDFeatureExtractor(feature_dim)
        
        # Prepare model with device-specific optimizations
        self.model = self.device_manager.prepare_model(self.model)
        self.model = self.device_manager.optimize_inference(self.model)
        
        print(f"✅ ReID model initialized")
        print(f"   Device: {device}")
        print(f"   Dtype: {self.device_config['dtype']}")
        print(f"   Autocast: {self.device_config['autocast']}")
        print(f"   Feature dim: {feature_dim}")
    
    def preprocess_crops(self, crops: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocess person crops for feature extraction with device optimization
        
        Args:
            crops: List of person crop images (BGR format)
            
        Returns:
            Preprocessed tensor batch
        """
        if not crops:
            return torch.empty(0, 3, 256, 128, device=self.device, dtype=self.device_config["dtype"])
        
        processed_crops = []
        for crop in crops:
            # Convert BGR to RGB
            if len(crop.shape) == 3:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            else:
                crop_rgb = crop
                
            # Convert to PIL Image and apply transforms
            pil_image = Image.fromarray(crop_rgb)
            tensor_image = self.transform(pil_image)
            processed_crops.append(tensor_image)
        
        # Stack into batch tensor and prepare for device
        batch = torch.stack(processed_crops)
        return self.device_manager.prepare_tensor(batch)
    
    def extract_features(self, crops: List[np.ndarray]) -> torch.Tensor:
        """
        Extract ReID features from person crops with device optimization
        
        Args:
            crops: List of person crop images
            
        Returns:
            Feature tensor of shape (N, feature_dim)
        """
        if not crops:
            return torch.empty(0, self.feature_dim, device=self.device, dtype=self.device_config["dtype"])
        
        # Preprocess crops
        batch = self.preprocess_crops(crops)
        
        # Extract features with autocast if supported
        with self.device_manager.autocast_context():
            with torch.no_grad():
                features = self.model(batch)
                
        return features
    
    def compute_similarity(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between feature vectors
        
        Args:
            features1: First set of features (N, feature_dim)
            features2: Second set of features (M, feature_dim)
            
        Returns:
            Similarity matrix (N, M)
        """
        if features1.numel() == 0 or features2.numel() == 0:
            return torch.empty(0, 0)
        
        # Normalize features
        features1_norm = torch.nn.functional.normalize(features1, p=2, dim=1)
        features2_norm = torch.nn.functional.normalize(features2, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.mm(features1_norm, features2_norm.t())
        
        return similarity

def create_reid_model(device: str = "cpu") -> ReIDModel:
    """
    Factory function to create a ReID model with device optimization
    
    Args:
        device: Device to run the model on
        
    Returns:
        Initialized ReID model with device-specific optimizations
    """
    return ReIDModel(device=device)
