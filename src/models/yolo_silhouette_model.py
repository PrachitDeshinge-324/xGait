"""
YOLO Segmentation-based Silhouette Extractor
A replacement for U²-Net that uses YOLO11 segmentation models for person silhouette extraction.
"""

import torch
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import List, Union, Optional


class YOLOSilhouetteExtractor:
    """
    YOLO Segmentation implementation for person silhouette extraction
    Replaces U²-Net with YOLO11 instance segmentation for better accuracy and speed
    """
    
    def __init__(self, device: Optional[str] = None, model_path: Optional[str] = None):
        """
        Initialize YOLO Silhouette Extractor
        
        Args:
            device: Device to run inference on ('cuda', 'mps', 'cpu')
            model_path: Path to YOLO segmentation model weights
        """
        # Set device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
            
        # Flag to track if we need to use CPU fallback due to MPS issues
        self.use_cpu_fallback = False

        # Find available YOLO segmentation model
        if model_path is None:
            possible_paths = [
                "weights/yolo11m-seg.pt",  # Preferred segmentation model
                "weights/yolo11s-seg.pt",  # Smaller segmentation model
                "weights/yolo11n-seg.pt",  # Nano segmentation model
                "yolo11m-seg.pt",          # Auto-download medium segmentation model
                "yolo11s-seg.pt",          # Auto-download small segmentation model
                "weights/yolo11m.pt",      # Detection model fallback
                "weights/yolov8n.pt"       # Old detection model fallback
            ]
            
            model_path = None
            for path in possible_paths:
                if Path(path).exists() or not path.startswith("weights/"):
                    model_path = path
                    break
                    
            if model_path is None:
                # Default to auto-download
                model_path = "yolo11m-seg.pt"
                print("📥 Will auto-download YOLO11m segmentation model")
        
        try:
            self.model = YOLO(model_path)
            print(f"✅ Loaded YOLO model from: {model_path}")
            
            # Set model device and ensure it's in evaluation mode
            if hasattr(self.model.model, 'to'):
                try:
                    self.model.model.to(self.device)
                except Exception as device_error:
                    print(f"⚠️ Device setup issue: {device_error}, using default device")
                    
            if hasattr(self.model.model, 'eval'):
                self.model.model.eval()
            
            # Check if the model supports segmentation
            self.has_segmentation = self._check_segmentation_support()
            
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            raise

    def _check_segmentation_support(self) -> bool:
        """Check if the loaded YOLO model supports segmentation"""
        try:
            # Check model name first - most reliable method
            model_name = str(self.model.ckpt_path if hasattr(self.model, 'ckpt_path') else "")
            if 'seg' in model_name.lower():
                print("✅ YOLO model supports segmentation (detected from filename)")
                return True
            
            # Try a test prediction to see if segmentation is available
            test_image = np.zeros((640, 640, 3), dtype=np.uint8)
            results = self.model.predict(test_image, verbose=False)
            
            # Check if the result object has the masks attribute, even if empty
            if hasattr(results[0], 'masks'):
                print("✅ YOLO model supports segmentation (detected from prediction)")
                return True
            else:
                print("⚠️ YOLO model does not support segmentation, using bounding boxes")
                return False
                
        except Exception as e:
            print(f"⚠️ Could not determine YOLO segmentation support ({e}), using bounding boxes")
            return False

    def extract_silhouettes(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract silhouettes from a list of images (batch processing)
        
        Args:
            images: List of input images (BGR format)
            
        Returns:
            List of binary silhouette masks (0-255 grayscale)
        """
        if not images:
            return []
            
        results = []
        
        for i, image in enumerate(images):
            # Validate image
            if image is None or image.size == 0 or len(image.shape) < 2:
                original_h, original_w = 1, 1  # Minimum size for invalid images
                results.append(np.zeros((original_h, original_w), dtype=np.uint8))
                continue
                
            original_h, original_w = image.shape[:2]
            
            # Skip images with invalid dimensions
            if original_h <= 0 or original_w <= 0 or original_h < 10 or original_w < 10:
                results.append(np.zeros((max(1, original_h), max(1, original_w)), dtype=np.uint8))
                continue
                
            try:
                mask = self._extract_single_silhouette(image)
                results.append(mask)
            except Exception as e:
                print(f"⚠️ Error processing image {i} in YOLO batch: {e}")
                results.append(np.zeros((original_h, original_w), dtype=np.uint8))
        
        return results

    def _extract_single_silhouette(self, image: np.ndarray) -> np.ndarray:
        """Extract silhouette from a single image using YOLO segmentation"""
        original_h, original_w = image.shape[:2]
        mask = np.zeros((original_h, original_w), dtype=np.uint8)
        
        try:
            # Validate input image
            if image is None or image.size == 0:
                return mask
            
            # Ensure image has correct shape and data type
            if len(image.shape) != 3 or image.shape[2] != 3:
                print(f"⚠️ Invalid image shape: {image.shape}, expected (H, W, 3)")
                return mask
                
            # Ensure image is in correct format (uint8)
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            
            # Run YOLO prediction with error handling
            # Use CPU fallback if MPS has persistent issues
            prediction_device = 'cpu' if self.use_cpu_fallback else self.device
            
            # Handle MPS-specific issues with batch normalization
            if self.device == 'mps' and not self.use_cpu_fallback:
                try:
                    results = self.model.predict(image, verbose=False, device=prediction_device)
                except (AttributeError, RuntimeError) as mps_error:
                    if 'bn' in str(mps_error) or 'batch' in str(mps_error).lower():
                        print(f"⚠️ MPS batch norm issue detected, switching to CPU fallback for all future predictions")
                        self.use_cpu_fallback = True
                        results = self.model.predict(image, verbose=False, device='cpu')
                    else:
                        raise mps_error
            else:
                results = self.model.predict(image, verbose=False, device=prediction_device)
            
            if not results or len(results) == 0:
                return mask
                
            result = results[0]
            
            # Check if we have detection results
            if result.boxes is None or len(result.boxes) == 0:
                return mask
            
            # Filter for person class (class 0 in COCO dataset)
            person_indices = []
            for i, cls in enumerate(result.boxes.cls):
                if int(cls) == 0:  # Person class
                    person_indices.append(i)
            
            if not person_indices:
                return mask
            
            if self.has_segmentation and result.masks is not None:
                # Use segmentation masks
                for idx in person_indices:
                    if idx < len(result.masks.data):
                        try:
                            # Get mask data with proper error handling
                            mask_data = result.masks.data[idx]
                            
                            # Convert to numpy if it's a tensor
                            if hasattr(mask_data, 'cpu'):
                                mask_data = mask_data.cpu().numpy()
                            
                            # Ensure mask_data is 2D
                            if len(mask_data.shape) > 2:
                                mask_data = mask_data.squeeze()
                            
                            # Validate mask dimensions
                            if mask_data.size == 0:
                                continue
                                
                            # Resize mask to original image size
                            mask_resized = cv2.resize(mask_data, (original_w, original_h), 
                                                    interpolation=cv2.INTER_NEAREST)
                            
                            # Convert to binary mask
                            binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
                            
                            # Combine with existing mask (union of all person masks)
                            mask = np.maximum(mask, binary_mask)
                            
                        except Exception as mask_error:
                            print(f"⚠️ Error processing mask {idx}: {mask_error}")
                            continue
            else:
                # Fallback: use bounding boxes to create rectangular masks
                for idx in person_indices:
                    if idx < len(result.boxes.xyxy):
                        try:
                            x1, y1, x2, y2 = result.boxes.xyxy[idx].cpu().numpy().astype(int)
                            
                            # Ensure coordinates are within image bounds
                            x1 = max(0, min(x1, original_w - 1))
                            y1 = max(0, min(y1, original_h - 1))
                            x2 = max(x1 + 1, min(x2, original_w))
                            y2 = max(y1 + 1, min(y2, original_h))
                            
                            # Create rectangular mask
                            mask[y1:y2, x1:x2] = 255
                            
                        except Exception as bbox_error:
                            print(f"⚠️ Error processing bbox {idx}: {bbox_error}")
                            continue
                        
        except Exception as e:
            print(f"⚠️ Error in YOLO segmentation: {type(e).__name__}: {str(e)}")
            # Try to provide more specific error information
            if hasattr(e, 'args') and len(e.args) > 0:
                print(f"   Error details: {e.args}")
            return mask
            
        return mask

    def extract_silhouette(self, image: np.ndarray) -> np.ndarray:
        """
        Extract silhouette from a single image (compatible with single image interface)
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Binary silhouette mask (0-255 grayscale)
        """
        return self._extract_single_silhouette(image)

    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            'architecture': 'YOLO11-Seg',
            'weights_loaded': True,
            'device': self.device,
            'has_segmentation': self.has_segmentation,
            'model_path': str(self.model.ckpt_path) if hasattr(self.model, 'ckpt_path') else 'unknown'
        }

    def __call__(self, images: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Callable interface for the extractor
        
        Args:
            images: Single image or list of images
            
        Returns:
            Single mask or list of masks
        """
        if isinstance(images, list):
            return self.extract_silhouettes(images)
        else:
            return self.extract_silhouette(images)
