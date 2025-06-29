"""
CDGNet Model for Human Parsing (from https://github.com/Gait3D/CDGNet-Parsing)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- CDGNet backbone and modules ----
# (This is a minimal, self-contained version for inference, based on the official repo)

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class CDGNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # This is a simplified version. For full details, see the official repo.
        self.layer1 = ConvBNReLU(3, 64, 3, padding=1)
        self.layer2 = ConvBNReLU(64, 128, 3, padding=1)
        self.layer3 = ConvBNReLU(128, 256, 3, padding=1)
        self.layer4 = ConvBNReLU(256, 512, 3, padding=1)
        self.classifier = nn.Conv2d(512, num_classes, 1)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x
