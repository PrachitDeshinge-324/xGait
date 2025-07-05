"""
Backbone networks for XGait model
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """BasicBlock for ResNet"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet9(nn.Module):
    """ResNet9 backbone as used in official XGait"""
    
    def __init__(self, block=BasicBlock, channels=[64, 128, 256, 512], layers=[1, 1, 1, 1], 
                 strides=[1, 2, 2, 1], maxpool=False):
        super(ResNet9, self).__init__()
        self.inplanes = channels[0]
        
        # Initial convolution - match checkpoint structure (3x3 kernel)
        conv1_layer = nn.Conv2d(1, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Module()
        self.conv1.conv = conv1_layer
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        
        if maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = lambda x: x
        
        # Residual layers
        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=strides[0])
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=strides[1])
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=strides[2])
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=strides[3])
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1.conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def get_backbone(backbone_cfg):
    """Create backbone network"""
    if backbone_cfg['type'] == 'ResNet9':
        return ResNet9(
            block=BasicBlock,
            channels=backbone_cfg['channels'],
            layers=backbone_cfg['layers'],
            strides=backbone_cfg['strides'],
            maxpool=backbone_cfg.get('maxpool', False)
        )
    else:
        raise ValueError(f"Unknown backbone type: {backbone_cfg['type']}")
