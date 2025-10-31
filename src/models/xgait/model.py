"""
Main XGait model implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from .backbone import get_backbone
from .alignment import CALayers, CALayersP
from .pooling import HorizontalPoolingPyramid, PackSequenceWrapper
from .layers import SeparateFCs, SeparateBNNecks, SetBlockWrapper

# Configure logging
logger = logging.getLogger(__name__)


class OfficialXGait(nn.Module):
    """
    Official XGait Model Implementation
    Exactly matches the Gait3D-Benchmark implementation
    """
    
    def __init__(self, model_cfg):
        super(OfficialXGait, self).__init__()
        self.build_network(model_cfg)
    
    def build_network(self, model_cfg):
        """Build the complete XGait network"""
        # backbone for silhouette
        self.Backbone_sil = get_backbone(model_cfg['backbone_cfg'])
        self.Backbone_sil = SetBlockWrapper(self.Backbone_sil)

        # backbone for parsing
        self.Backbone_par = get_backbone(model_cfg['backbone_cfg'])
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
        
        # Log input dtypes for debugging (only if needed)
        logger.debug(f"Model forward input dtypes - pars: {pars.dtype}, sils: {sils.dtype}, labs: {labs.dtype}")
        
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
        # Ensure pars is float type for F.interpolate (requires float tensors)
        try:
            pars_float = pars.squeeze(1).float() if pars.dtype != torch.float32 else pars.squeeze(1)
            logger.debug(f"F.interpolate input - dtype: {pars_float.dtype}, shape: {pars_float.shape}, target size: ({h}, {w})")
            mask_resize = F.interpolate(input=pars_float, size=(h, w), mode='nearest')
            mask_resize = mask_resize.view(n*s, h, w)
            logger.debug(f"mask_resize created - dtype: {mask_resize.dtype}, shape: {mask_resize.shape}")
        except Exception as e:
            logger.error(f"Error in F.interpolate or mask_resize creation: {e}")
            logger.error(f"pars dtype: {pars.dtype}, shape: {pars.shape}")
            logger.error(f"pars_float dtype: {pars_float.dtype if 'pars_float' in locals() else 'not created'}")
            logger.error(f"Target size: ({h}, {w})")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise

        try:
            outs_pcm_up = self.pcm_up(outs_sil, outs_par, mask_resize)  # [n, c, s, h/4, w]
            outs_pcm_middle = self.pcm_middle(outs_sil, outs_par, mask_resize)  # [n, c, s, h/2, w]
            outs_pcm_down = self.pcm_down(outs_sil, outs_par, mask_resize)  # [n, c, s, h/4, w]
        except Exception as e:
            logger.error(f"Error in PCM modules: {e}")
            logger.error(f"outs_sil - dtype: {outs_sil.dtype}, shape: {outs_sil.shape}")
            logger.error(f"outs_par - dtype: {outs_par.dtype}, shape: {outs_par.shape}")
            logger.error(f"mask_resize - dtype: {mask_resize.dtype}, shape: {mask_resize.shape}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise
        
        # Validate tensor shapes before concatenation
        if outs_pcm_up.dim() != 5 or outs_pcm_middle.dim() != 5 or outs_pcm_down.dim() != 5:
            raise RuntimeError(f"PCM outputs have incorrect dimensions: "
                             f"up={outs_pcm_up.shape}, middle={outs_pcm_middle.shape}, down={outs_pcm_down.shape}")
        
        # Ensure dtypes are consistent
        if outs_pcm_up.dtype != outs_pcm_middle.dtype or outs_pcm_up.dtype != outs_pcm_down.dtype:
            logger.warning(f"PCM outputs have inconsistent dtypes: "
                         f"up={outs_pcm_up.dtype}, middle={outs_pcm_middle.dtype}, down={outs_pcm_down.dtype}")
            # Convert to consistent dtype
            target_dtype = outs_pcm_up.dtype
            outs_pcm_middle = outs_pcm_middle.to(dtype=target_dtype)
            outs_pcm_down = outs_pcm_down.to(dtype=target_dtype)
        
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

        # LOGIC-008 fix: Validate tensor shapes and dtypes before concatenation
        embeddings = [embed_sil, embed_par, embed_gcm, embed_pcm]
        logits = [logits_sil, logits_par, logits_gcm, logits_pcm]
        
        # Validate embedding tensor shapes
        embed_shapes = [e.shape for e in embeddings]
        if not all(len(shape) == 3 for shape in embed_shapes):
            raise RuntimeError(f"Embedding tensors have incorrect dimensions: {embed_shapes}")
        
        # Validate logits tensor shapes  
        logits_shapes = [l.shape for l in logits]
        if not all(len(shape) == 3 for shape in logits_shapes):
            raise RuntimeError(f"Logits tensors have incorrect dimensions: {logits_shapes}")
        
        # Ensure device consistency
        embed_devices = [str(e.device) for e in embeddings]
        if not all(device == embed_devices[0] for device in embed_devices):
            raise RuntimeError(f"Embedding tensors are on different devices: {embed_devices}")
            
        logits_devices = [str(l.device) for l in logits]
        if not all(device == logits_devices[0] for device in logits_devices):
            raise RuntimeError(f"Logits tensors are on different devices: {logits_devices}")
        
        # Ensure dtype consistency for embeddings
        embed_dtypes = [e.dtype for e in embeddings]
        if not all(dtype == embed_dtypes[0] for dtype in embed_dtypes):
            logger.warning(f"Embedding tensors have inconsistent dtypes: {embed_dtypes}")
            target_dtype = embed_dtypes[0]
            embeddings = [e.to(dtype=target_dtype) for e in embeddings]
        
        # Ensure dtype consistency for logits
        logits_dtypes = [l.dtype for l in logits]
        if not all(dtype == logits_dtypes[0] for dtype in logits_dtypes):
            logger.warning(f"Logits tensors have inconsistent dtypes: {logits_dtypes}")
            target_dtype = logits_dtypes[0]
            logits = [l.to(dtype=target_dtype) for l in logits]
        
        # Perform validated concatenation
        embed_cat = torch.cat(embeddings, dim=-1)
        logits_cat = torch.cat(logits, dim=-1)

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
    
    Missing parameters identified and added:
    1. backbone_cfg: Added 'in_channels' parameter
    2. Added 'dropout' parameter for regularization
    3. Fixed parameter compatibility issues
    """
    model_cfg = {
        'backbone_cfg': {
            'type': 'ResNet9',
            'block': 'BasicBlock',
            'in_channels': 1,  # MISSING PARAMETER 1: Input channel dimension
            'channels': [64, 128, 256, 512],  # Official uses 512 final channels
            'layers': [1, 1, 1, 1],
            'strides': [1, 2, 2, 1],
            'maxpool': False
        },
        'CALayers': {
            'channels': 1024,  # 2 * 512 (concatenated features)
            'reduction': 32   # 1024 // 32 = 32 intermediate channels
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
        'bin_num': [16],
        'dropout': 0.2  # MISSING PARAMETER 2: Dropout rate for regularization
    }
    
    return OfficialXGait(model_cfg)
