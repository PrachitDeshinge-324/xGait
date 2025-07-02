#!/usr/bin/env python3
"""
Human Parsing Model using CDGNet implementation
Based on the official CDGNet model for human parsing
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from pathlib import Path
import sys
from collections import OrderedDict

# Add parent directory to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from utils.device_utils import get_global_device
from .cdgnet_official import Res_Deeplab

class CDGNetParsingModel:
    """Human Parsing Model using CDGNet implementation"""
    
    def __init__(self, model_path='weights/cdgnet.pth', device=None):
        self.input_height = 256
        self.input_width = 256
        self.model_path = model_path
        self.device = device if device is not None else get_global_device()
        
        # CDGNet preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.input_height, self.input_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # CDGNet class definitions (12 classes for human parsing)
        self.get_class_names = [
            'background', 'class1', 'class2', 'class3', 'class4',
            'class5', 'class6', 'class7', 'class8', 'class9',
            'class10', 'class11'
        ]
        
        self.get_class_colors = [
            [0, 0, 0],        # 0: background
            [128, 0, 0],      # 1
            [0, 128, 0],      # 2
            [128, 128, 0],    # 3
            [0, 0, 128],      # 4
            [128, 0, 128],    # 5
            [0, 128, 128],    # 6
            [128, 128, 128],  # 7
            [64, 0, 0],       # 8
            [192, 0, 0],      # 9
            [64, 128, 0],     # 10
            [192, 128, 0],    # 11
        ]
        
        self.num_classes = len(self.get_class_names)
        self.palette_idx = np.array(self.get_class_colors)
        
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the CDGNet model with pretrained weights"""
        try:
            # Initialize CDGNet model
            self.model = Res_Deeplab(num_classes=self.num_classes)
            
            # Load pretrained weights
            if Path(self.model_path).exists():
                state_dict = torch.load(self.model_path, map_location=self.device)
                
                # Handle 'module.' prefix if present
                if any(k.startswith('module.') for k in state_dict.keys()):
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] if k.startswith('module.') else k
                        new_state_dict[name] = v
                    state_dict = new_state_dict
                
                self.model.load_state_dict(state_dict)
                print(f"✅ Loaded CDGNet weights from {self.model_path}")
            else:
                print(f"⚠️  Model weights not found at {self.model_path}")
                print("   Using randomly initialized weights")
            
            self.model.eval()
            
            # Handle device compatibility issues (especially MPS)
            try:
                # Test if model works on specified device
                test_input = torch.randn(1, 3, 256, 256).to(self.device)
                self.model.to(self.device)
                with torch.no_grad():
                    _ = self.model(test_input)
                print(f"✅ CDGNet model successfully loaded on {self.device}")
            except Exception as device_error:
                print(f"⚠️  Device {self.device} incompatible with CDGNet: {device_error}")
                print("   Falling back to CPU")
                self.device = 'cpu'
                self.model.to(self.device)
                
        except Exception as e:
            print(f"❌ Failed to load CDGNet parsing model: {e}")
            self.model = None

    def is_model_loaded(self):
        """Check if model is properly loaded"""
        return self.model is not None

    def extract_parsing(self, input_images):
        """Extract human parsing masks from input images"""
        if self.model is None:
            print("❌ CDGNet model not loaded")
            return []

        try:
            batch_imgs = []
            batch_parsing = []
            
            # Prepare batch
            for i, img in enumerate(input_images):
                # Convert to PIL format if needed
                if isinstance(img, np.ndarray):
                    # Resize to model input size
                    crop_img = cv2.resize(img, (self.input_width, self.input_height), 
                                        interpolation=cv2.INTER_LINEAR)
                    
                    # Apply transforms
                    crop_tensor = self.transform(crop_img)
                    crop_tensor = crop_tensor.unsqueeze(0)  # Add batch dimension
                    
                    if i == 0:
                        batch_imgs = crop_tensor
                    else:
                        batch_imgs = torch.cat((batch_imgs, crop_tensor), 0)
            
            # Move to device
            batch_imgs = batch_imgs.to(self.device)
            
            # Forward pass
            with torch.no_grad():
                # CDGNet returns: [[seg0, seg1, seg2], [edge], [fea_h1, fea_w1]]
                outputs = self.model(batch_imgs)
                parsing_logits = outputs[0][-1]  # Use seg2 (the final segmentation)
                parsing_predictions = parsing_logits.argmax(1).cpu().numpy().astype(np.uint8)
            
            # Process outputs
            for i, img in enumerate(input_images):
                img_height, img_width = img.shape[:2]
                prediction = parsing_predictions[i]
                
                # Resize prediction back to original size
                parsing_resized = cv2.resize(prediction, (img_width, img_height), 
                                           interpolation=cv2.INTER_NEAREST)
                
                batch_parsing.append(parsing_resized)
            
            return batch_parsing
            
        except Exception as e:
            print(f"❌ CDGNet parsing extraction failed: {e}")
            # Return empty parsing masks as fallback
            return [np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8) for img in input_images]

    def get_class_names_list(self):
        """Get list of class names"""
        return self.get_class_names.copy()

    def get_num_classes(self):
        """Get number of classes"""
        return self.num_classes

    def get_class_colors_list(self):
        """Get list of class colors"""
        return self.get_class_colors.copy()

if __name__ == "__main__":
    # Simple test
    parser = CDGNetParsingModel()
    print(f"Model loaded: {parser.is_model_loaded()}")
    print(f"Number of classes: {parser.get_num_classes()}")
    print(f"Class names: {parser.get_class_names_list()}")