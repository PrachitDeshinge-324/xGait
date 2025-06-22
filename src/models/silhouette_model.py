"""
Fast Silhouette Extraction Model for XGait Inference
Implements U²-Net architecture for real-time person silhouette extraction
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
import torch.nn.functional as F
import cv2
import numpy as np
from typing import List, Tuple
import torchvision.transforms as transforms
from PIL import Image

from utils.device_utils import DeviceManager
from config import get_device_config

class ConvBlock(nn.Module):
    """Basic convolution block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class RSU7(nn.Module):
    """Residual U-block with 7 layers for U²-Net"""
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()
        
        self.rebnconvin = ConvBlock(in_ch, out_ch)
        
        self.rebnconv1 = ConvBlock(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv2 = ConvBlock(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv3 = ConvBlock(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv4 = ConvBlock(mid_ch, mid_ch)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv5 = ConvBlock(mid_ch, mid_ch)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv6 = ConvBlock(mid_ch, mid_ch)
        
        self.rebnconv7 = ConvBlock(mid_ch, mid_ch)
        
        self.rebnconv6d = ConvBlock(mid_ch*2, mid_ch)
        self.rebnconv5d = ConvBlock(mid_ch*2, mid_ch)
        self.rebnconv4d = ConvBlock(mid_ch*2, mid_ch)
        self.rebnconv3d = ConvBlock(mid_ch*2, mid_ch)
        self.rebnconv2d = ConvBlock(mid_ch*2, mid_ch)
        self.rebnconv1d = ConvBlock(mid_ch*2, out_ch)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        
        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)
        
        hx6 = self.rebnconv6(hx)
        
        hx7 = self.rebnconv7(hx6)
        
        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = F.interpolate(hx6d, size=(hx5.shape[2], hx5.shape[3]), mode='bilinear', align_corners=False)
        
        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=(hx4.shape[2], hx4.shape[3]), mode='bilinear', align_corners=False)
        
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=(hx3.shape[2], hx3.shape[3]), mode='bilinear', align_corners=False)
        
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=(hx2.shape[2], hx2.shape[3]), mode='bilinear', align_corners=False)
        
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=(hx1.shape[2], hx1.shape[3]), mode='bilinear', align_corners=False)
        
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        
        return hx1d + hxin

class RSU4(nn.Module):
    """Residual U-block with 4 layers for U²-Net"""
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()
        
        self.rebnconvin = ConvBlock(in_ch, out_ch)
        
        self.rebnconv1 = ConvBlock(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv2 = ConvBlock(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv3 = ConvBlock(mid_ch, mid_ch)
        
        self.rebnconv4 = ConvBlock(mid_ch, mid_ch)
        
        self.rebnconv3d = ConvBlock(mid_ch*2, mid_ch)
        self.rebnconv2d = ConvBlock(mid_ch*2, mid_ch)
        self.rebnconv1d = ConvBlock(mid_ch*2, out_ch)
    
    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        
        hx3 = self.rebnconv3(hx)
        
        hx4 = self.rebnconv4(hx3)
        
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=(hx2.shape[2], hx2.shape[3]), mode='bilinear', align_corners=False)
        
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=(hx1.shape[2], hx1.shape[3]), mode='bilinear', align_corners=False)
        
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        
        return hx1d + hxin

class U2NET(nn.Module):
    """U²-Net for fast and accurate silhouette extraction"""
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()
        
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage2 = RSU7(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage3 = RSU7(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage4 = RSU7(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage5 = RSU4(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage6 = RSU4(512, 256, 512)
        
        # Decoder
        self.stage5d = RSU4(1024, 256, 512)
        self.stage4d = RSU7(1024, 128, 256)
        self.stage3d = RSU7(512, 64, 128)
        self.stage2d = RSU7(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)
        
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)
        
        self.outconv = nn.Conv2d(6*out_ch, out_ch, 1)
    
    def forward(self, x):
        hx = x
        
        # Encoder
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        
        hx6 = self.stage6(hx)
        hx6up = F.interpolate(hx6, size=(hx5.shape[2], hx5.shape[3]), mode='bilinear', align_corners=False)
        
        # Decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = F.interpolate(hx5d, size=(hx4.shape[2], hx4.shape[3]), mode='bilinear', align_corners=False)
        
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = F.interpolate(hx4d, size=(hx3.shape[2], hx3.shape[3]), mode='bilinear', align_corners=False)
        
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = F.interpolate(hx3d, size=(hx2.shape[2], hx2.shape[3]), mode='bilinear', align_corners=False)
        
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = F.interpolate(hx2d, size=(hx1.shape[2], hx1.shape[3]), mode='bilinear', align_corners=False)
        
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))
        
        # Side outputs
        d1 = self.side1(hx1d)
        
        d2 = self.side2(hx2d)
        d2 = F.interpolate(d2, size=(hx.shape[2], hx.shape[3]), mode='bilinear', align_corners=False)
        
        d3 = self.side3(hx3d)
        d3 = F.interpolate(d3, size=(hx.shape[2], hx.shape[3]), mode='bilinear', align_corners=False)
        
        d4 = self.side4(hx4d)
        d4 = F.interpolate(d4, size=(hx.shape[2], hx.shape[3]), mode='bilinear', align_corners=False)
        
        d5 = self.side5(hx5d)
        d5 = F.interpolate(d5, size=(hx.shape[2], hx.shape[3]), mode='bilinear', align_corners=False)
        
        d6 = self.side6(hx6)
        d6 = F.interpolate(d6, size=(hx.shape[2], hx.shape[3]), mode='bilinear', align_corners=False)
        
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        
        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)

class SilhouetteExtractor:
    """
    Fast silhouette extraction for XGait inference with device optimization
    """
    def __init__(self, device: str = "mps", model_path: str = None):
        self.device = device
        self.device_config = get_device_config(device)
        self.device_manager = DeviceManager(device, self.device_config["dtype"])
        
        # Initialize U²-Net model
        self.model = U2NET(in_ch=3, out_ch=1)
        
        # Load pretrained weights if available
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint)
            print(f"✅ Loaded silhouette model from {model_path}")
        else:
            print("⚠️  Using randomly initialized U²-Net (consider training or downloading pretrained weights)")
        
        # Prepare model with device optimizations
        self.model = self.device_manager.prepare_model(self.model)
        self.model = self.device_manager.optimize_inference(self.model)
        
        # Preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((320, 320)),  # Standard size for U²-Net
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ SilhouetteExtractor initialized")
        print(f"   Device: {device}")
        print(f"   Dtype: {self.device_config['dtype']}")
        print(f"   Autocast: {self.device_config['autocast']}")
    
    def preprocess_crops(self, crops: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocess person crops for silhouette extraction
        
        Args:
            crops: List of person crop images (BGR format)
            
        Returns:
            Preprocessed tensor batch
        """
        if not crops:
            return torch.empty(0, 3, 320, 320, device=self.device, dtype=self.device_config["dtype"])
        
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
    
    def extract_silhouettes(self, crops: List[np.ndarray], target_size: Tuple[int, int] = (64, 44)) -> List[np.ndarray]:
        """
        Extract silhouettes from person crops
        
        Args:
            crops: List of person crop images
            target_size: Target size for output silhouettes (H, W)
            
        Returns:
            List of silhouette masks
        """
        if not crops:
            return []
        
        # Preprocess crops
        batch = self.preprocess_crops(crops)
        
        # Extract silhouettes with autocast if supported
        with self.device_manager.autocast_context():
            with torch.no_grad():
                outputs = self.model(batch)
                # Use main output (first element)
                silhouettes = outputs[0]
        
        # Post-process silhouettes
        processed_silhouettes = []
        for i in range(silhouettes.shape[0]):
            # Convert to numpy
            sil = silhouettes[i, 0].cpu().numpy()  # Remove channel dim
            
            # Threshold
            sil = (sil > 0.5).astype(np.uint8) * 255
            
            # Resize to target size
            sil_resized = cv2.resize(sil, target_size, interpolation=cv2.INTER_NEAREST)
            
            processed_silhouettes.append(sil_resized)
        
        return processed_silhouettes
    
    def extract_silhouettes_batch(self, crops_batch: List[List[np.ndarray]], target_size: Tuple[int, int] = (64, 44)) -> List[List[np.ndarray]]:
        """
        Extract silhouettes for multiple tracks in parallel
        
        Args:
            crops_batch: List of crop lists (one per track)
            target_size: Target size for output silhouettes
            
        Returns:
            List of silhouette lists (one per track)
        """
        # Flatten all crops for batch processing
        all_crops = []
        track_lengths = []
        
        for track_crops in crops_batch:
            all_crops.extend(track_crops)
            track_lengths.append(len(track_crops))
        
        if not all_crops:
            return [[] for _ in crops_batch]
        
        # Extract all silhouettes at once
        all_silhouettes = self.extract_silhouettes(all_crops, target_size)
        
        # Split back into tracks
        result = []
        start_idx = 0
        for length in track_lengths:
            end_idx = start_idx + length
            result.append(all_silhouettes[start_idx:end_idx])
            start_idx = end_idx
        
        return result

def create_silhouette_extractor(device: str = "mps", model_path: str = None) -> SilhouetteExtractor:
    """Create a SilhouetteExtractor instance"""
    return SilhouetteExtractor(device=device, model_path=model_path)
