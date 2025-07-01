#!/usr/bin/env python3
"""
Human Parsing Model using GaitParsing U²-Net implementation
Based on: https://github.com/wzb-bupt/GaitParsing
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

# Add parent directory to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from utils.device_utils import get_global_device

# U²-Net Architecture Implementation
class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))
        return xout

def _upsample_like(src, tar):
    """Upsample tensor 'src' to have the same spatial size with tensor 'tar'"""
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=False)
    return src

class RSU7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()
        
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)
        
        self.rebnconv6d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

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
        hx6dup = _upsample_like(hx6d, hx5)
        
        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)
        
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        
        return hx1d + hxin

class RSU6(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()
        
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)
        
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

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
        hx6 = self.rebnconv6(hx5)
        
        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)
        
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        
        return hx1d + hxin

class RSU5(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()
        
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)
        
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

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
        hx5 = self.rebnconv5(hx4)
        
        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        
        return hx1d + hxin

class RSU4(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()
        
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)
        
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

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
        hx3dup = _upsample_like(hx3d, hx2)
        
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        
        return hx1d + hxin

class RSU4F(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()
        
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)
        
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))
        
        return hx1d + hxin

class U2NET(nn.Module):
    """U²-Net for human parsing based on GaitParsing implementation"""
    
    def __init__(self, in_ch, out_ch):
        super(U2NET, self).__init__()
        
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage6 = RSU4F(512, 256, 512)
        
        # Decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
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
        
        # Stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        
        # Stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        
        # Stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        
        # Stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        
        # Stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        
        # Stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)
        
        # Decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)
        
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))
        
        # Side outputs
        d1 = self.side1(hx1d)
        
        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)
        
        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)
        
        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)
        
        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)
        
        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)
        
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        
        return d0, d1, d2, d3, d4, d5, d6

class HumanParsingModel:
    """Human Parsing Model using GaitParsing U²-Net implementation"""
    
    def __init__(self, model_path='weights/parsing_u2net.pth', device=None):
        self.input_height = 144
        self.input_width = 96
        self.transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.model_path = model_path
        self.device = device if device is not None else get_global_device()
        
        # GaitParsing class definitions (7 classes for gait parsing)
        self.get_class_names = [
            'background', 'head', 'body', 'r_arm', 'l_arm', 'r_leg', 'l_leg'
        ]
        
        self.get_class_colors = [
            [0, 0, 0],       # background
            [255, 0, 0],     # head
            [255, 255, 0],   # body  
            [0, 0, 255],     # r_arm
            [255, 0, 255],   # l_arm
            [0, 255, 0],     # r_leg
            [0, 255, 255]    # l_leg
        ]
        
        self.num_classes = len(self.get_class_names)
        self.palette_idx = np.array(self.get_class_colors)
        
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the U²-Net model with pretrained weights"""
        try:
            # Initialize U²-Net with 3 input channels (RGB) and num_classes output channels
            self.model = U2NET(3, self.num_classes)
            
            # Load pretrained weights
            if Path(self.model_path).exists():
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"✅ Loaded GaitParsing weights from {self.model_path}")
            else:
                print(f"⚠️  Model weights not found at {self.model_path}")
                print("   Using randomly initialized weights")
            
            self.model.eval()
            
            # Move to device
            if self.device != 'cpu' and torch.cuda.is_available():
                self.model.cuda()
            else:
                self.model.to(self.device)
                
        except Exception as e:
            print(f"❌ Failed to load parsing model: {e}")
            self.model = None

    def is_model_loaded(self):
        """Check if model is properly loaded"""
        return self.model is not None

    def extract_parsing(self, input_images):
        """Extract human parsing masks from input images"""
        if self.model is None:
            print("❌ Model not loaded")
            return []

        try:
            batch_imgs = []
            batch_parsing = []
            
            # Prepare batch
            for i, img in enumerate(input_images):
                # Resize to model input size
                crop_img = cv2.resize(img, (self.input_width, self.input_height), 
                                    interpolation=cv2.INTER_LINEAR)
                
                # Convert to tensor and normalize
                crop_img = torch.from_numpy(crop_img.transpose((2, 0, 1)))
                crop_img = self.transform(crop_img.float().div(255.0))
                crop_img = crop_img.view(1, crop_img.shape[0], crop_img.shape[1], crop_img.shape[2])
                
                if i == 0:
                    batch_imgs = crop_img
                else:
                    batch_imgs = torch.cat((batch_imgs, crop_img), 0)
            
            # Move to device
            if self.device != 'cpu' and torch.cuda.is_available():
                batch_imgs = batch_imgs.cuda()
            else:
                batch_imgs = batch_imgs.to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs, _, _, _, _, _, _ = self.model(batch_imgs)
                outputs = outputs.cpu()
            
            # Process outputs
            for i, img in enumerate(input_images):
                img_height, img_width = img.shape[:2]
                prediction = outputs[i, :, :, :].data.numpy()
                
                # Convert prediction to colored parsing mask
                result = np.zeros((self.input_height, self.input_width, 3))
                for h in range(prediction.shape[1]):
                    for w in range(prediction.shape[2]):
                        result[h][w] = self.palette_idx[np.argmax(prediction[:, h, w])]
                
                # Resize back to original size
                parsing_im = cv2.resize(result, (img_width, img_height), 
                                      interpolation=cv2.INTER_NEAREST)
                
                # Convert to grayscale class labels for compatibility
                parsing_gray = np.zeros((img_height, img_width), dtype=np.uint8)
                for class_id, color in enumerate(self.get_class_colors):
                    mask = np.all(parsing_im == color, axis=2)
                    parsing_gray[mask] = class_id
                
                batch_parsing.append(parsing_gray)
            
            return batch_parsing
            
        except Exception as e:
            print(f"❌ Parsing extraction failed: {e}")
            # Return empty parsing masks as fallback
            return [np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8) for img in input_images]

    def get_class_names_list(self):
        """Get list of class names"""
        return self.get_class_names.copy()

    def get_num_classes(self):
        """Get number of classes"""
        return self.num_classes

if __name__ == "__main__":
    # Simple test
    parser = HumanParsingModel()
    print(f"Model loaded: {parser.is_model_loaded()}")
    print(f"Number of classes: {parser.get_num_classes()}")
    print(f"Class names: {parser.get_class_names()}")
