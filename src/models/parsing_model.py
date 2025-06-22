"""
Human Parsing Model for XGait Inference
Uses pre-trained SCHP ResNet101 model for human body part segmentation
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
from typing import List, Tuple, Dict
import torchvision.transforms as transforms
from PIL import Image

from utils.device_utils import DeviceManager
from config import get_device_config

# Human parsing labels for SCHP model
PARSING_LABELS = {
    0: "Background",
    1: "Hat",
    2: "Hair",
    3: "Glove",
    4: "Sunglasses",
    5: "UpperClothes",
    6: "Dress",
    7: "Coat",
    8: "Socks",
    9: "Pants",
    10: "Jumpsuits",
    11: "Scarf",
    12: "Skirt",
    13: "Face",
    14: "Left-arm",
    15: "Right-arm",
    16: "Left-leg",
    17: "Right-leg",
    18: "Left-shoe",
    19: "Right-shoe"
}

class Bottleneck(nn.Module):
    """ResNet Bottleneck block"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """ResNet backbone for SCHP"""
    def __init__(self, block, layers, output_stride=16, BatchNorm=nn.BatchNorm2d):
        self.inplanes = 64
        super(ResNet, self).__init__()
        
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x, low_level_feat

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if backbone == 'resnet':
            inplanes = 2048
        else:
            inplanes = 320

        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPPConv(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = ASPPConv(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = ASPPConv(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = ASPPConv(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

class ASPPConv(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, padding, dilation, BatchNorm):
        super(ASPPConv, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(outplanes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

class Decoder(nn.Module):
    """DeepLab decoder"""
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet':
            low_level_inplanes = 256
        else:
            low_level_inplanes = 24

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

class SCHPModel(nn.Module):
    """Self-Correction for Human Parsing (SCHP) model"""
    def __init__(self, backbone='resnet', output_stride=16, num_classes=20, freeze_bn=False):
        super(SCHPModel, self).__init__()
        
        BatchNorm = nn.BatchNorm2d

        self.backbone = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm)
        self.aspp = ASPP('resnet', output_stride, BatchNorm)
        self.decoder = Decoder(num_classes, 'resnet', BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

class HumanParsingModel:
    """
    Human parsing model wrapper for XGait inference with device optimization
    """
    def __init__(self, device: str = "mps", model_path: str = "weights/schp_resnet101.pth"):
        self.device = device
        self.device_config = get_device_config(device)
        self.device_manager = DeviceManager(device, self.device_config["dtype"])
        
        # Initialize SCHP model
        self.model = SCHPModel(backbone='resnet', output_stride=16, num_classes=20)
        
        # Load pretrained weights
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            self.model.load_state_dict(new_state_dict, strict=False)
            print(f"✅ Loaded human parsing model from {model_path}")
        else:
            print(f"⚠️  Model file not found: {model_path}")
            print("   Using randomly initialized SCHP model")
        
        # Prepare model with device optimizations
        self.model = self.device_manager.prepare_model(self.model)
        self.model = self.device_manager.optimize_inference(self.model)
        
        # Preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),  # Standard size for SCHP
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ HumanParsingModel initialized")
        print(f"   Device: {device}")
        print(f"   Dtype: {self.device_config['dtype']}")
        print(f"   Autocast: {self.device_config['autocast']}")
    
    def preprocess_crops(self, crops: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocess person crops for human parsing
        
        Args:
            crops: List of person crop images (BGR format)
            
        Returns:
            Preprocessed tensor batch
        """
        if not crops:
            return torch.empty(0, 3, 512, 512, device=self.device, dtype=self.device_config["dtype"])
        
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
    
    def parse_humans(self, crops: List[np.ndarray], target_size: Tuple[int, int] = (64, 44)) -> List[np.ndarray]:
        """
        Extract human parsing maps from person crops
        
        Args:
            crops: List of person crop images
            target_size: Target size for output parsing maps (H, W)
            
        Returns:
            List of parsing maps
        """
        if not crops:
            return []
        
        # Preprocess crops
        batch = self.preprocess_crops(crops)
        
        # Extract parsing maps with autocast if supported
        with self.device_manager.autocast_context():
            with torch.no_grad():
                outputs = self.model(batch)
                # Get class predictions
                parsing_maps = torch.argmax(outputs, dim=1)
        
        # Post-process parsing maps
        processed_maps = []
        for i in range(parsing_maps.shape[0]):
            # Convert to numpy
            parsing_map = parsing_maps[i].cpu().numpy().astype(np.uint8)
            
            # Resize to target size
            parsing_resized = cv2.resize(parsing_map, target_size, interpolation=cv2.INTER_NEAREST)
            
            processed_maps.append(parsing_resized)
        
        return processed_maps
    
    def parse_humans_batch(self, crops_batch: List[List[np.ndarray]], target_size: Tuple[int, int] = (64, 44)) -> List[List[np.ndarray]]:
        """
        Extract parsing maps for multiple tracks in parallel
        
        Args:
            crops_batch: List of crop lists (one per track)
            target_size: Target size for output parsing maps
            
        Returns:
            List of parsing map lists (one per track)
        """
        # Flatten all crops for batch processing
        all_crops = []
        track_lengths = []
        
        for track_crops in crops_batch:
            all_crops.extend(track_crops)
            track_lengths.append(len(track_crops))
        
        if not all_crops:
            return [[] for _ in crops_batch]
        
        # Extract all parsing maps at once
        all_parsing_maps = self.parse_humans(all_crops, target_size)
        
        # Split back into tracks
        result = []
        start_idx = 0
        for length in track_lengths:
            end_idx = start_idx + length
            result.append(all_parsing_maps[start_idx:end_idx])
            start_idx = end_idx
        
        return result
    
    def get_part_mask(self, parsing_map: np.ndarray, part_labels: List[int]) -> np.ndarray:
        """
        Extract specific body part mask from parsing map
        
        Args:
            parsing_map: Human parsing map
            part_labels: List of part label IDs to include
            
        Returns:
            Binary mask for specified parts
        """
        mask = np.zeros_like(parsing_map, dtype=np.uint8)
        for label in part_labels:
            mask[parsing_map == label] = 255
        return mask
    
    def get_upper_body_mask(self, parsing_map: np.ndarray) -> np.ndarray:
        """Get upper body mask (head, arms, upper clothes)"""
        upper_parts = [1, 2, 4, 5, 7, 11, 13, 14, 15]  # Hat, Hair, Sunglasses, UpperClothes, Coat, Scarf, Face, Left-arm, Right-arm
        return self.get_part_mask(parsing_map, upper_parts)
    
    def get_lower_body_mask(self, parsing_map: np.ndarray) -> np.ndarray:
        """Get lower body mask (legs, pants, shoes)"""
        lower_parts = [9, 12, 16, 17, 18, 19]  # Pants, Skirt, Left-leg, Right-leg, Left-shoe, Right-shoe
        return self.get_part_mask(parsing_map, lower_parts)

def create_human_parsing_model(device: str = "mps", model_path: str = "weights/schp_resnet101.pth") -> HumanParsingModel:
    """Create a HumanParsingModel instance"""
    return HumanParsingModel(device=device, model_path=model_path)
