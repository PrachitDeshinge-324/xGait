"""
XGait Model Package

This package contains the complete XGait implementation with all components:
- Core model components (backbone, alignment, pooling, layers)
- Official model implementation 
- Adapter for backward compatibility
- Deprecated official model wrapper
"""

from .backbone import ResNet9, BasicBlock, get_backbone
from .alignment import CALayers, CALayersP
from .pooling import HorizontalPoolingPyramid, PackSequenceWrapper
from .layers import SeparateFCs, SeparateBNNecks, SetBlockWrapper
from .model import OfficialXGait, create_official_xgait_model
from .inference import OfficialXGaitInference, create_official_xgait_inference
from .utils import get_valid_args

# Import adapter and legacy components
from .adapter import XGaitAdapter, create_xgait_adapter, create_xgait_inference as create_xgait_inference_adapter

__all__ = [
    # Backbone components
    'ResNet9', 'BasicBlock', 'get_backbone',
    
    # Alignment modules
    'CALayers', 'CALayersP',
    
    # Pooling modules
    'HorizontalPoolingPyramid', 'PackSequenceWrapper',
    
    # Layer modules
    'SeparateFCs', 'SeparateBNNecks', 'SetBlockWrapper',
    
    # Main model
    'OfficialXGait', 'create_official_xgait_model',
    
    # Inference
    'OfficialXGaitInference', 'create_official_xgait_inference',
    
    # Adapter (recommended interface)
    'XGaitAdapter', 'create_xgait_adapter', 'create_xgait_inference_adapter',
    
    # Legacy components
    'LegacyOfficialXGaitInference', 'create_legacy_xgait_inference',
    
    # Utilities
    'get_valid_args'
]
