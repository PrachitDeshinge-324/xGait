"""
U²-Net Silhouette Extraction Model with Real Weights Support
Implements U²-Net for person silhouette extraction from RGB images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Union, Optional, Tuple
import logging
import sys

# Add parent directory to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from utils.device_utils import get_global_device
from models.u2net import U2NET  # Import the correct U2NET model

logger = logging.getLogger(__name__)
class SilhouetteExtractor:
    """Silhouette extraction using U²-Net model."""
    
    def __init__(self, weights=None, device=None):
        """Initialize the silhouette extractor."""
        self.device = device if device is not None else get_global_device()
        
        # Set default weights path if not provided
        if weights is None:
            # Go up from xGait/src/models/silhouette_model.py to Project/Person Identification/Weights/
            weights_path = Path(__file__).parent.parent.parent.parent / "Weights" / "u2net.pth"
            if weights_path.exists():
                self.weights = str(weights_path)
            else:
                self.weights = None
        else:
            self.weights = weights
            
        self.model = U2NET(in_ch=3, out_ch=1)
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Try to load real weights
        self.weights_loaded = self.load_real_weights()
        if not self.weights_loaded:
            logger.warning("⚠️  No U²-Net weights found, using random initialization")
    
    def load_real_weights(self):
        """Load real U²-Net weights if available."""
        if self.weights is None:
            logger.warning("⚠️  No weights path specified")
            return False
            
        try:
            state_dict = torch.load(self.weights, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"✅ Successfully loaded U²-Net weights from {self.weights}")
            return True

        except Exception as e:
            logger.warning(f"⚠️  Failed to load weights from {self.weights}: {e}")
            return False

    def extract_silhouettes(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Extract silhouettes from a list of images with efficient batch processing."""
        if not images:
            return []
        
        silhouettes = []
        batch_size = min(8, len(images))  # Process in batches to optimize GPU usage
        
        with torch.no_grad():
            for batch_start in range(0, len(images), batch_size):
                batch_end = min(batch_start + batch_size, len(images))
                batch_images = images[batch_start:batch_end]
                batch_tensors = []
                
                # Preprocess batch
                for i, img in enumerate(batch_images):
                    try:
                        # Preprocess image
                        if len(img.shape) == 2:  # Grayscale
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                        elif img.shape[2] == 4:  # RGBA
                            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                        
                        # Validate image
                        if img is None or img.size == 0:
                            logger.warning(f"⚠️  Invalid image at index {batch_start + i}, using zero silhouette")
                            silhouettes.append(np.zeros((256, 128), dtype=np.uint8))
                            continue
                        
                        # Convert to tensor
                        img_tensor = self.transform(img)
                        batch_tensors.append(img_tensor)
                        
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to preprocess image {batch_start + i}: {e}")
                        silhouettes.append(np.zeros((256, 128), dtype=np.uint8))
                
                if not batch_tensors:
                    continue
                
                # Stack batch and process
                try:
                    batch_tensor = torch.stack(batch_tensors).to(self.device)
                    
                    # Forward pass on batch
                    outputs = self.model(batch_tensor)
                    predictions = outputs[0]  # Main output
                    
                    # Process batch predictions
                    for i, pred in enumerate(predictions):
                        try:
                            # Convert to silhouette with proper normalization
                            silhouette = pred.squeeze().cpu().numpy()
                            
                            # Apply sigmoid to get probabilities (U2Net outputs logits)
                            silhouette = torch.sigmoid(torch.tensor(silhouette)).numpy()
                            
                            # Resize to match parsing mask size (height=256, width=128)
                            silhouette = cv2.resize(silhouette, (128, 256))
                            
                            # Convert to 0-255 range with proper threshold
                            silhouette = (silhouette * 255).astype(np.uint8)
                            
                            # Apply threshold to create binary mask (use lower threshold for better detection)
                            _, silhouette = cv2.threshold(silhouette, 50, 255, cv2.THRESH_BINARY)
                            
                            silhouettes.append(silhouette)
                            
                        except Exception as e:
                            logger.warning(f"⚠️  Failed to process prediction {batch_start + i}: {e}")
                            silhouettes.append(np.zeros((256, 128), dtype=np.uint8))
                    
                    # Clear batch tensor to free GPU memory
                    del batch_tensor, outputs, predictions
                    
                except Exception as e:
                    logger.warning(f"⚠️  Failed to process batch starting at {batch_start}: {e}")
                    # Add fallback silhouettes for the batch
                    for _ in range(len(batch_tensors)):
                        silhouettes.append(np.zeros((256, 128), dtype=np.uint8))
                
                # Clear GPU cache periodically
                if hasattr(torch.cuda, 'empty_cache') and batch_start % 32 == 0:
                    torch.cuda.empty_cache()
        
        return silhouettes
    
    def __call__(self, images: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
        """Extract silhouettes from images."""
        if isinstance(images, np.ndarray):
            if len(images.shape) == 3:  # Single image
                return self.extract_silhouettes([images])[0]
            else:  # Batch of images
                return self.extract_silhouettes(list(images))
        else:  # List of images
            return self.extract_silhouettes(images)
    
    def is_model_loaded(self):
        """Check if real weights were loaded."""
        return self.weights_loaded
    
    def test_extraction(self) -> bool:
        """Test silhouette extraction with a dummy image."""
        try:
            # Create a simple test image
            test_image = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
            
            # Test extraction
            result = self.extract_silhouettes([test_image])
            
            if result and len(result) == 1 and result[0].shape == (256, 128):
                logger.info("✅ Silhouette extraction test passed")
                return True
            else:
                logger.warning("⚠️  Silhouette extraction test failed - incorrect output shape")
                return False
                
        except Exception as e:
            logger.error(f"❌ Silhouette extraction test failed: {e}")
            return False
    
    def get_model_info(self):
        """Get information about the loaded model."""
        return {
            'architecture': 'U²-Net',
            'weights_loaded': self.weights_loaded,
            'device': str(self.device),
            'parameters': sum(p.numel() for p in self.model.parameters())
        }

def create_silhouette_extractor(device=None, model_path=None):
    """Create and return a silhouette extractor instance."""
    if device is None:
        device = get_global_device()
    return SilhouetteExtractor(device=device)
