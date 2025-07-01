"""
Official XGait Model Implementation
Based on: https://github.com/Gait3D/Gait3D-Benchmark/blob/main/opengait/modeling/models/xgait.py

This implementation matches the official Gait3D-Benchmark XGait model exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Union
import logging

# Configure logging
logger = logging.getLogger(__name__)

def get_valid_args(kwargs, targets):
    """Get valid arguments for a function from kwargs"""
    valid_args = {}
    for target in targets:
        if target in kwargs:
            valid_args[target] = kwargs[target]
    return valid_args


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


class SetBlockWrapper(nn.Module):
    """Wrapper for handling set/sequence data"""
    
    def __init__(self, forward_block):
        super(SetBlockWrapper, self).__init__()
        self.forward_block = forward_block
        
    def forward(self, x, *args, **kwargs):
        """
        x: [N, C, S, H, W] where S is sequence length
        """
        if len(x.size()) == 4:
            return self.forward_block(x, *args, **kwargs)
            
        N, C, S, H, W = x.size()
        x = x.view(N*S, C, H, W)
        output = self.forward_block(x, *args, **kwargs)
        
        # Reshape back to sequence format
        _, new_C, new_H, new_W = output.size()
        output = output.view(N, new_C, S, new_H, new_W)
        return output


class PackSequenceWrapper(nn.Module):
    """Temporal pooling wrapper"""
    
    def __init__(self, pooling_func):
        super(PackSequenceWrapper, self).__init__()
        self.pooling_func = pooling_func
        
    def forward(self, seqs, seqL, options=None, *args, **kwargs):
        """
        seqs: [N, C, S, H, W]
        seqL: sequence lengths
        """
        if options is None:
            options = {"dim": 2}
        
        dim = options.get('dim', 2)
        
        if self.pooling_func == torch.max:
            out, _ = torch.max(seqs, dim=dim)
        else:
            out = self.pooling_func(seqs, dim=dim)
            
        return [out]


class HorizontalPoolingPyramid(nn.Module):
    """Horizontal Pooling Pyramid"""
    
    def __init__(self, bin_num=[16]):
        super(HorizontalPoolingPyramid, self).__init__()
        self.bin_num = bin_num
        
    def forward(self, x):
        """
        x: [N, C, H, W]
        output: [N, C, P] where P is sum(bin_num)
        """
        n, c = x.size()[:2]
        features = []
        
        for num_bin in self.bin_num:
            z = F.adaptive_avg_pool2d(x, (num_bin, 1))  # [N, C, num_bin, 1]
            z = z.view(n, c, num_bin)  # [N, C, num_bin]
            features.append(z)
            
        return torch.cat(features, dim=2)  # [N, C, P]


class SeparateFCs(nn.Module):
    """Separate fully connected layers for each part - Official Structure"""
    
    def __init__(self, parts_num, in_channels, out_channels):
        super(SeparateFCs, self).__init__()
        self.parts_num = parts_num
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Official structure: tensor parameter instead of ModuleList
        self.fc_bin = nn.Parameter(torch.randn(parts_num, in_channels, out_channels))
        
    def forward(self, x):
        """
        x: [N, C, P] where P is parts_num
        """
        N, C, P = x.shape
        outputs = torch.zeros(N, self.out_channels, P, device=x.device)
        
        for p in range(P):
            outputs[:, :, p] = torch.matmul(x[:, :, p], self.fc_bin[p])  # [N, out_channels]
            
        return outputs  # [N, out_channels, P]


class SeparateBNNecks(nn.Module):
    """Separate Batch Normalization necks for each part - Official Structure"""
    
    def __init__(self, class_num, in_channels, parts_num):
        super(SeparateBNNecks, self).__init__()
        self.parts_num = parts_num
        self.in_channels = in_channels
        self.class_num = class_num
        
        # Official structure: unified BN and FC
        self.bn1d = nn.BatchNorm1d(in_channels * parts_num)
        self.fc_bin = nn.Parameter(torch.randn(parts_num, in_channels, class_num))
        
    def forward(self, x):
        """
        x: [N, C, P]
        returns: (features, logits)
        """
        N, C, P = x.shape
        
        # Flatten for unified BN: [N, C*P]
        x_flat = x.view(N, C * P)
        features_flat = self.bn1d(x_flat)
        
        # Reshape back: [N, C, P]
        features = features_flat.view(N, C, P)
        
        # Compute logits using fc_bin: [P, C, class_num]
        logits = torch.zeros(N, self.class_num, P, device=x.device)
        for p in range(P):
            logits[:, :, p] = torch.matmul(features[:, :, p], self.fc_bin[p])  # [N, class_num]
        
        return features, logits


class CALayers(nn.Module):
    """Global Cross-granularity Alignment Module"""
    
    def __init__(self, channels, reduction):
        super(CALayers, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, sil_feat, par_feat):
        """
        Global cross-granularity alignment
        """
        N, C, S, H, W = sil_feat.shape
        
        # Flatten for processing
        sil_flat = sil_feat.contiguous().view(N*S, C, H, W)
        par_flat = par_feat.contiguous().view(N*S, C, H, W)
        
        # Concatenate silhouette and parsing features
        concat_feat = torch.cat([sil_flat, par_flat], dim=1)  # [N*S, 2*C, H, W]
        
        # Generate attention from concatenated features
        attention = self.avg_pool(concat_feat).view(N*S, 2*C)  # [N*S, 2*C]
        attention = self.fc(attention).view(N*S, 2*C, 1, 1)   # [N*S, 2*C, 1, 1]
        
        # Apply attention to concatenated features
        aligned_feat = concat_feat * attention
        
        # Split back to individual streams and combine
        sil_aligned, par_aligned = torch.chunk(aligned_feat, 2, dim=1)
        combined_feat = sil_aligned + par_aligned  # [N*S, C, H, W]
        
        # Reshape back
        combined_feat = combined_feat.view(N, C, S, H, W)
        
        return combined_feat


class CALayersP(nn.Module):
    """Part-based Cross-granularity Alignment Module"""
    
    def __init__(self, channels, reduction, choosed_part='up', with_max_pool=True):
        super(CALayersP, self).__init__()
        self.choosed_part = choosed_part
        self.with_max_pool = with_max_pool
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if with_max_pool:
            self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, sil_feat, par_feat, mask_resize):
        """
        Part-based cross-granularity alignment
        """
        N, C, S, H, W = sil_feat.shape
        
        # Determine part region
        if self.choosed_part == 'up':
            part_h = H // 4
            sil_part = sil_feat[:, :, :, :part_h, :]
            par_part = par_feat[:, :, :, :part_h, :]
        elif self.choosed_part == 'middle':
            start_h, end_h = H // 4, 3 * H // 4
            sil_part = sil_feat[:, :, :, start_h:end_h, :]
            par_part = par_feat[:, :, :, start_h:end_h, :]
        else:  # down
            part_h = 3 * H // 4
            sil_part = sil_feat[:, :, :, part_h:, :]
            par_part = par_feat[:, :, :, part_h:, :]
        
        # Flatten for processing
        sil_flat = sil_part.contiguous().view(N*S, C, -1, W)
        par_flat = par_part.contiguous().view(N*S, C, -1, W)
        
        # Concatenate silhouette and parsing features
        concat_feat = torch.cat([sil_flat, par_flat], dim=1)  # [N*S, 2*C, part_h, W]
        
        # Generate attention from concatenated features
        if self.with_max_pool:
            attention_avg = self.avg_pool(concat_feat).view(N*S, 2*C)
            attention_max = self.max_pool(concat_feat).view(N*S, 2*C)
            attention = self.fc(attention_avg + attention_max).view(N*S, 2*C, 1, 1)
        else:
            attention = self.avg_pool(concat_feat).view(N*S, 2*C)
            attention = self.fc(attention).view(N*S, 2*C, 1, 1)
        
        # Apply attention to concatenated features
        aligned_feat = concat_feat * attention
        
        # Split back to individual streams and combine
        sil_aligned, par_aligned = torch.chunk(aligned_feat, 2, dim=1)
        aligned_part = sil_aligned + par_aligned  # [N*S, C, part_h, W]
        
        # Reshape back to part dimensions
        _, _, _, part_h, part_w = sil_part.shape
        aligned_part = aligned_part.view(N, C, S, part_h, part_w)
        
        return aligned_part


class OfficialXGait(nn.Module):
    """
    Official XGait Model Implementation
    Exactly matches the Gait3D-Benchmark implementation
    """
    
    def __init__(self, model_cfg):
        super(OfficialXGait, self).__init__()
        self.build_network(model_cfg)
        
    def get_backbone(self, backbone_cfg):
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
    
    def build_network(self, model_cfg):
        """Build the complete XGait network"""
        # backbone for silhouette
        self.Backbone_sil = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone_sil = SetBlockWrapper(self.Backbone_sil)

        # backbone for parsing
        self.Backbone_par = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone_par = SetBlockWrapper(self.Backbone_par)

        # Global Cross-granularity Alignment Module
        self.gcm = CALayers(**model_cfg['CALayers'])
        
        # Part Cross-granularity Alignment Module
        self.pcm_up = CALayersP(**model_cfg['CALayersP'], choosed_part='up')
        self.pcm_middle = CALayersP(**model_cfg['CALayersP'], choosed_part='middle')
        self.pcm_down = CALayersP(**model_cfg['CALayersP'], choosed_part='down')
        
        # FCs
        self.FCs_sil = SeparateFCs(**model_cfg['SeparateFCs'])
        self.FCs_par = SeparateFCs(**model_cfg['SeparateFCs'])
        self.FCs_gcm = SeparateFCs(**model_cfg['SeparateFCs'])
        self.FCs_pcm = SeparateFCs(**model_cfg['SeparateFCs'])

        # BNNecks
        self.BNNecks_sil = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.BNNecks_par = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.BNNecks_gcm = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.BNNecks_pcm = SeparateBNNecks(**model_cfg['SeparateBNNecks'])

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

    def forward(self, inputs):
        """
        Forward pass exactly matching official implementation
        
        Args:
            inputs: (ipts, labs, _, _, seqL) where:
                - ipts: [pars, sils] - parsing and silhouettes
                - labs: labels
                - seqL: sequence lengths
        """
        ipts, labs, _, _, seqL = inputs

        pars = ipts[0]  # parsing masks [N, S, H, W] or [N, C, S, H, W]
        sils = ipts[1]  # silhouettes [N, S, H, W] or [N, C, S, H, W]
        
        # Handle input dimensions - support both 4D and 5D inputs
        if len(pars.size()) == 4 and len(sils.size()) == 4:
            # Input is [N, S, H, W] - add channel dimension
            pars = pars.unsqueeze(1)  # [N, 1, S, H, W]
            sils = sils.unsqueeze(1)  # [N, 1, S, H, W]
            vis_channel = 1
        elif len(pars.size()) == 5 and len(sils.size()) == 5:
            # Input is already [N, C, S, H, W]
            if pars.size(1) == 1 and sils.size(1) == 1:
                vis_channel = 1
            else:
                vis_channel = pars.size(1)
        else:
            raise ValueError(f"Unexpected input shapes: pars {pars.shape}, sils {sils.shape}")

        del ipts
        
        # Extract features from dual backbones
        outs_sil = self.Backbone_sil(sils)  # [n, c, s, h, w]
        outs_par = self.Backbone_par(pars)  # [n, c, s, h, w]

        # Global Cross-granularity Alignment
        outs_gcm = self.gcm(outs_sil, outs_par)  # [n, c, s, h, w]

        # Part Cross-granularity Alignment
        # mask_resize: [n, s, h, w]
        n, c, s, h, w = outs_sil.size()
        mask_resize = F.interpolate(input=pars.squeeze(1), size=(h, w), mode='nearest')
        mask_resize = mask_resize.view(n*s, h, w)

        outs_pcm_up = self.pcm_up(outs_sil, outs_par, mask_resize)  # [n, c, s, h/4, w]
        outs_pcm_middle = self.pcm_middle(outs_sil, outs_par, mask_resize)  # [n, c, s, h/2, w]
        outs_pcm_down = self.pcm_down(outs_sil, outs_par, mask_resize)  # [n, c, s, h/4, w]
        outs_pcm = torch.cat((outs_pcm_up, outs_pcm_middle, outs_pcm_down), dim=-2)  # [n, c, s, h, w]

        # Temporal Pooling, TP
        outs_sil = self.TP(outs_sil, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        outs_par = self.TP(outs_par, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        outs_gcm = self.TP(outs_gcm, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        outs_pcm = self.TP(outs_pcm, seqL, options={"dim": 2})[0]  # [n, c, h, w]

        # Horizontal Pooling Matching, HPM
        feat_sil = self.HPP(outs_sil)  # [n, c, p]
        feat_par = self.HPP(outs_par)  # [n, c, p]
        feat_gcm = self.HPP(outs_gcm)  # [n, c, p]
        feat_pcm = self.HPP(outs_pcm)  # [n, c, p]

        # silhouette part features
        embed_sil = self.FCs_sil(feat_sil)  # [n, c, p]
        _, logits_sil = self.BNNecks_sil(embed_sil)  # [n, c, p]

        # parsing part features
        embed_par = self.FCs_par(feat_par)  # [n, c, p]
        _, logits_par = self.BNNecks_par(embed_par)  # [n, c, p]

        # gcm part features
        embed_gcm = self.FCs_gcm(feat_gcm)  # [n, c, p]
        _, logits_gcm = self.BNNecks_gcm(embed_gcm)  # [n, c, p]

        # pcm part features
        embed_pcm = self.FCs_pcm(feat_pcm)  # [n, c, p]
        _, logits_pcm = self.BNNecks_pcm(embed_pcm)  # [n, c, p]

        # concatenate four parts features
        embed_cat = torch.cat((embed_sil, embed_par, embed_gcm, embed_pcm), dim=-1)
        logits_cat = torch.cat((logits_sil, logits_par, logits_gcm, logits_pcm), dim=-1)

        embed = embed_cat

        n, _, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_cat, 'labels': labs},
                'softmax': {'logits': logits_cat, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.reshape(n*s, vis_channel, h, w)
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval


def create_official_xgait_model(num_classes=3000):
    """
    Create XGait model with official Gait3D configuration
    """
    model_cfg = {
        'backbone_cfg': {
            'type': 'ResNet9',
            'block': 'BasicBlock',
            'channels': [64, 128, 256, 512],  # Official uses 512 final channels
            'layers': [1, 1, 1, 1],
            'strides': [1, 2, 2, 1],
            'maxpool': False
        },
        'CALayers': {
            'channels': 1024,  # 2 * 512 (concatenated features)
            'reduction': 32    # 1024 // 32 = 32 intermediate channels
        },
        'CALayersP': {
            'channels': 1024,  # 2 * 512 (concatenated features)
            'reduction': 32,   # 1024 // 32 = 32 intermediate channels
            'with_max_pool': True
        },
        'SeparateFCs': {
            'in_channels': 512,  # Match backbone output
            'out_channels': 256,
            'parts_num': 16
        },
        'SeparateBNNecks': {
            'class_num': num_classes,
            'in_channels': 256,
            'parts_num': 16
        },
        'bin_num': [16]
    }
    
    return OfficialXGait(model_cfg)


class OfficialXGaitInference:
    """
    Official XGait inference engine exactly matching Gait3D-Benchmark
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = None, num_classes: int = 3000):
        if device is None:
            from config import get_xgait_device
            device = get_xgait_device()
        
        self.device = device
        self.num_classes = num_classes
        
        # Create the official XGait model
        self.model = create_official_xgait_model(num_classes=num_classes).to(self.device)
        
        # Sequence parameters from official config
        self.min_sequence_length = 10
        self.target_sequence_length = 30
        
        # Input preprocessing parameters (official Gait3D config)
        self.input_height = 64
        self.input_width = 44
        
        # Load weights if available
        if model_path:
            self.model_loaded = self.load_model_weights(model_path)
        else:
            logger.warning("⚠️ No model weights provided, using random initialization.")
            self.model_loaded = False
        
        # Set model to eval mode
        self.model.eval()
        
        logger.info(f"🎯 Official XGait Model initialized:")
        logger.info(f"   - Input size: {self.input_height}x{self.input_width}")
        logger.info(f"   - Target sequence length: {self.target_sequence_length}")
        logger.info(f"   - Device: {self.device}")
        logger.info(f"   - Weights loaded: {self.model_loaded}")
    
    def load_model_weights(self, model_path: str):
        """Load official XGait weights"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load with strict=True since this should match exactly
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"Missing keys: {len(missing_keys)}")
                logger.debug(f"Missing keys: {missing_keys[:5]}...")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {len(unexpected_keys)}")
                logger.debug(f"Unexpected keys: {unexpected_keys[:5]}...")
            
            logger.info(f"✅ Loaded official XGait weights from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load weights: {e}")
            return False
    
    def preprocess_sequence(self, silhouettes: List[np.ndarray], 
                          parsing_masks: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess silhouettes and parsing masks for official XGait
        Returns: (pars_tensor, sils_tensor) - note the order!
        """
        if not silhouettes or not parsing_masks:
            raise ValueError("Both silhouettes and parsing masks are required for official XGait")
        
        if len(silhouettes) != len(parsing_masks):
            raise ValueError("Silhouettes and parsing masks must have same length")
        
        # Ensure minimum sequence length
        if len(silhouettes) < self.min_sequence_length:
            # Repeat frames to reach minimum length
            repeat_factor = self.min_sequence_length // len(silhouettes) + 1
            silhouettes = (silhouettes * repeat_factor)[:self.min_sequence_length]
            parsing_masks = (parsing_masks * repeat_factor)[:self.min_sequence_length]
        
        # Limit to target sequence length
        if len(silhouettes) > self.target_sequence_length:
            # Sample frames uniformly
            indices = np.linspace(0, len(silhouettes) - 1, self.target_sequence_length, dtype=int)
            silhouettes = [silhouettes[i] for i in indices]
            parsing_masks = [parsing_masks[i] for i in indices]
        
        # Process silhouettes
        processed_sils = []
        for sil in silhouettes:
            if sil.shape[:2] != (self.input_height, self.input_width):
                sil_resized = cv2.resize(sil, (self.input_width, self.input_height))
            else:
                sil_resized = sil.copy()
            
            # Normalize to [0, 1]
            if sil_resized.max() > 1:
                sil_resized = sil_resized.astype(np.float32) / 255.0
            
            processed_sils.append(sil_resized)
        
        # Process parsing masks
        processed_pars = []
        for par in parsing_masks:
            if par.shape[:2] != (self.input_height, self.input_width):
                par_resized = cv2.resize(par, (self.input_width, self.input_height), 
                                       interpolation=cv2.INTER_NEAREST)
            else:
                par_resized = par.copy()
            
            # Normalize parsing masks
            if par_resized.max() > 1:
                par_resized = par_resized.astype(np.float32) / par_resized.max()
            
            processed_pars.append(par_resized)
        
        # Convert to tensors [S, H, W]
        sils_tensor = torch.FloatTensor(np.array(processed_sils)).to(self.device)
        pars_tensor = torch.FloatTensor(np.array(processed_pars)).to(self.device)
        
        # Add batch dimension [1, S, H, W]
        sils_tensor = sils_tensor.unsqueeze(0)
        pars_tensor = pars_tensor.unsqueeze(0)
        
        return pars_tensor, sils_tensor  # Note: parsing first!
    
    def extract_features(self, silhouettes: List[np.ndarray], 
                        parsing_masks: List[np.ndarray]) -> np.ndarray:
        """
        Extract features using official XGait model
        """
        try:
            with torch.no_grad():
                self.model.eval()
                
                # Preprocess
                pars, sils = self.preprocess_sequence(silhouettes, parsing_masks)
                
                # Prepare inputs in official format
                ipts = [pars, sils]
                labs = torch.zeros(1).long().to(self.device)  # Dummy labels for inference
                seqL = [len(silhouettes)]
                
                inputs = (ipts, labs, None, None, seqL)
                
                # Forward pass
                output = self.model(inputs)
                
                # Extract inference features
                features = output['inference_feat']['embeddings']
                
                return features.cpu().numpy()
                
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return np.array([])
    
    def is_model_loaded(self) -> bool:
        """Check if model weights are loaded"""
        return self.model_loaded


def create_official_xgait_inference(model_path: Optional[str] = None, 
                                   device: str = None, 
                                   num_classes: int = 3000) -> OfficialXGaitInference:
    """
    Create official XGait inference engine
    """
    if device is None:
        from config import get_xgait_device
        device = get_xgait_device()
    
    # Look for default model path
    if model_path is None:
        import os
        from pathlib import Path
        current_dir = Path(__file__).parent.parent.parent
        potential_paths = [
            current_dir / "weights" / "Gait3D-XGait-120000.pt",
            current_dir / "weights" / "xgait_gait3d.pth"
        ]
        
        for path in potential_paths:
            if path.exists():
                model_path = str(path)
                break
    
    return OfficialXGaitInference(model_path=model_path, device=device, num_classes=num_classes)


if __name__ == "__main__":
    # Test the official XGait model
    print("🧪 Testing Official XGait Model")
    
    # Create test data
    import cv2
    test_silhouettes = [np.random.randint(0, 255, (64, 44), dtype=np.uint8) for _ in range(30)]
    test_parsing = [np.random.randint(0, 7, (64, 44), dtype=np.uint8) for _ in range(30)]
    
    # Create XGait inference
    xgait = create_official_xgait_inference(device="cpu")
    
    # Extract features
    features = xgait.extract_features(test_silhouettes, test_parsing)
    
    print(f"✅ Extracted features: shape {features.shape}")
    print(f"📊 Model loaded: {xgait.is_model_loaded()}")
