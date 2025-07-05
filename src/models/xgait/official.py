"""
Official XGait Model Implementation
DEPRECATED: This file is now modularized into the xgait package.
Please use: from src.models.xgait import OfficialXGaitInference, create_official_xgait_inference

Based on: https://github.com/Gait3D/Gait3D-Benchmark/blob/main/opengait/modeling/models/xgait.py
"""

import logging
from typing import Optional

# Import from the new modular structure
from . import (
    OfficialXGaitInference, 
    create_official_xgait_inference,
    OfficialXGait,
    create_official_xgait_model
)

# Configure logging
logger = logging.getLogger(__name__)

# For backward compatibility, expose the main classes
__all__ = [
    'OfficialXGaitInference',
    'create_official_xgait_inference',
    'OfficialXGait',
    'create_official_xgait_model'
]

# Deprecation warning
logger.warning(
    "⚠️ DEPRECATED: official_xgait_model.py is deprecated. "
    "Please use 'from src.models.xgait import ...' instead."
)


if __name__ == "__main__":
    # Test the official XGait model
    print("🧪 Testing Official XGait Model (using new modular structure)")
    
    # Create test data
    import numpy as np
    test_silhouettes = [np.random.randint(0, 255, (64, 44), dtype=np.uint8) for _ in range(30)]
    test_parsing = [np.random.randint(0, 7, (64, 44), dtype=np.uint8) for _ in range(30)]
    
    # Create XGait inference
    xgait = create_official_xgait_inference(device="cpu")
    
    # Extract features
    features = xgait.extract_features(test_silhouettes, test_parsing)
    
    print(f"✅ Extracted features: shape {features.shape}")
    print(f"📊 Model loaded: {xgait.is_model_loaded()}")
    print("🎯 Using new modular XGait implementation")
