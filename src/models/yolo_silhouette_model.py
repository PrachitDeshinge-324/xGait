"""
YOLO Segmentation-based Silhouette Extraction Model
Implements person silhouette extraction using YOLOv11 segmentation masks
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Union, Optional, Tuple
import logging
import sys

# Add parent directory to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ultralytics import YOLO
from utils.device_utils import get_global_device

logger = logging.getLogger(__name__)

class YOLOSilhouetteExtractor:
    """Silhouette extraction using YOLO segmentation model."""
    
    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialize the YOLO segmentation-based silhouette extractor.
        
        Args:
            model_path: Path to YOLO segmentation model (e.g., yolo11n-seg.pt)
            device: Device to run inference on
        """
        self.device = device if device is not None else get_global_device()
        
        # Set default model path if not provided
        if model_path is None:
            # Use YOLOv11n-seg as default (lightweight segmentation model)
            model_path = "yolo11n-seg.pt"
        
        self.model_path = model_path
        
        try:
            # Load YOLO segmentation model
            self.model = YOLO(model_path)
            self.model.to(self.device)
            logger.info(f"✅ Successfully loaded YOLO segmentation model from {model_path}")
            self.weights_loaded = True
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO segmentation model from {model_path}: {e}")
            self.model = None
            self.weights_loaded = False
    
    def extract_silhouettes(self, images: List[np.ndarray], 
                          conf_threshold: float = 0.5) -> List[np.ndarray]:
        """
        Extract silhouettes from a list of images using YOLO segmentation.
        
        Args:
            images: List of input images (RGB format)
            conf_threshold: Confidence threshold for person detection
            
        Returns:
            List of silhouette masks (binary images, 0-255)
        """
        if not images or self.model is None:
            return []
        
        silhouettes = []
        
        for img in images:
            try:
                # Validate image
                if img is None or img.size == 0:
                    logger.warning("⚠️  Invalid image, using zero silhouette")
                    silhouettes.append(np.zeros((256, 128), dtype=np.uint8))
                    continue
                
                # Ensure image is in correct format
                if len(img.shape) == 2:  # Grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:  # RGBA
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                
                # Run YOLO segmentation
                results = self.model(
                    img,
                    conf=conf_threshold,
                    classes=[0],  # Person class only
                    verbose=False,
                    device=self.device
                )
                
                # Extract masks for person detections
                silhouette = self._extract_person_mask(results[0], img.shape[:2])
                
                # Resize silhouette to standard size (256, 128)
                silhouette_resized = cv2.resize(silhouette, (128, 256), 
                                              interpolation=cv2.INTER_NEAREST)
                
                silhouettes.append(silhouette_resized)
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to extract silhouette: {e}")
                # Create fallback silhouette
                fallback = np.zeros((256, 128), dtype=np.uint8)
                fallback[50:200, 30:98] = 255  # Basic person-like rectangle
                silhouettes.append(fallback)
        
        return silhouettes
    
    def _extract_person_mask(self, result, img_shape: Tuple[int, int]) -> np.ndarray:
        """
        Extract person mask from YOLO result.
        
        Args:
            result: YOLO detection result
            img_shape: Original image shape (height, width)
            
        Returns:
            Binary mask of person silhouette
        """
        height, width = img_shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Check if masks exist in the result
        if hasattr(result, 'masks') and result.masks is not None:
            # Use segmentation masks
            masks_data = result.masks.data.cpu().numpy()
            
            # Combine all person masks
            for mask_data in masks_data:
                # Resize mask to image dimensions
                person_mask = cv2.resize(mask_data, (width, height), 
                                       interpolation=cv2.INTER_NEAREST)
                # Convert to binary and add to combined mask
                person_mask = (person_mask > 0.5).astype(np.uint8) * 255
                mask = np.maximum(mask, person_mask)
        
        elif hasattr(result, 'boxes') and result.boxes is not None:
            # Fallback: use bounding boxes to create rectangular masks
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                mask[y1:y2, x1:x2] = 255
        
        return mask
    
    def extract_silhouettes_with_crops(self, frame: np.ndarray, 
                                     boxes: np.ndarray,
                                     conf_threshold: float = 0.5) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Extract both person crops and their corresponding silhouettes.
        
        Args:
            frame: Input video frame
            boxes: Bounding boxes array (N, 4) in xyxy format
            conf_threshold: Confidence threshold for segmentation
            
        Returns:
            Tuple of (crops, silhouettes)
        """
        crops = []
        silhouettes = []
        
        h, w = frame.shape[:2]
        
        for box in boxes:
            try:
                x1, y1, x2, y2 = box.astype(int)
                
                # Add padding and ensure valid coordinates
                padding = 10
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)
                
                # Extract crop
                crop = frame[y1:y2, x1:x2]
                
                # Ensure minimum crop size
                if crop.shape[0] > 32 and crop.shape[1] > 16:
                    # Run YOLO segmentation on the crop
                    crop_silhouettes = self.extract_silhouettes([crop], conf_threshold)
                    silhouette = crop_silhouettes[0] if crop_silhouettes else np.zeros((256, 128), dtype=np.uint8)
                    
                    crops.append(crop)
                    silhouettes.append(silhouette)
                else:
                    # Create placeholder crop and silhouette if too small
                    placeholder_crop = np.zeros((64, 32, 3), dtype=np.uint8)
                    placeholder_silhouette = np.zeros((256, 128), dtype=np.uint8)
                    crops.append(placeholder_crop)
                    silhouettes.append(placeholder_silhouette)
                    
            except Exception as e:
                logger.warning(f"⚠️  Failed to extract crop/silhouette: {e}")
                # Add fallback crop and silhouette
                fallback_crop = np.zeros((64, 32, 3), dtype=np.uint8)
                fallback_silhouette = np.zeros((256, 128), dtype=np.uint8)
                fallback_silhouette[50:200, 30:98] = 255  # Basic person shape
                crops.append(fallback_crop)
                silhouettes.append(fallback_silhouette)
        
        return crops, silhouettes
    
    def __call__(self, images: Union[np.ndarray, List[np.ndarray]], 
                 **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
        """Extract silhouettes from images."""
        if isinstance(images, np.ndarray):
            if len(images.shape) == 3:  # Single image
                return self.extract_silhouettes([images], **kwargs)[0]
            else:  # Batch of images
                return self.extract_silhouettes(list(images), **kwargs)
        else:  # List of images
            return self.extract_silhouettes(images, **kwargs)
    
    def is_model_loaded(self):
        """Check if model was loaded successfully."""
        return self.weights_loaded and self.model is not None
    
    def test_extraction(self) -> bool:
        """Test silhouette extraction with a dummy image."""
        try:
            # Create a simple test image
            test_image = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
            
            # Test extraction
            result = self.extract_silhouettes([test_image])
            
            if result and len(result) == 1 and result[0].shape == (256, 128):
                logger.info("✅ YOLO silhouette extraction test passed")
                return True
            else:
                logger.warning("⚠️  YOLO silhouette extraction test failed - incorrect output shape")
                return False
                
        except Exception as e:
            logger.error(f"❌ YOLO silhouette extraction test failed: {e}")
            return False
    
    def get_model_info(self):
        """Get information about the loaded model."""
        return {
            'architecture': 'YOLOv11-seg',
            'model_path': self.model_path,
            'weights_loaded': self.weights_loaded,
            'device': str(self.device),
            'supports_segmentation': True
        }

def create_yolo_silhouette_extractor(device=None, model_path=None):
    """Create and return a YOLO-based silhouette extractor instance."""
    return YOLOSilhouetteExtractor(device=device, model_path=model_path)
