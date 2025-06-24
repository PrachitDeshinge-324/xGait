"""
Official XGait Model Entry Point
Uses the official Gait3D-Benchmark implementation for maximum performance
"""
from pathlib import Path
from typing import Optional
import logging

from .xgait_adapter import XGaitAdapter

logger = logging.getLogger(__name__)


def create_xgait_inference(model_path: Optional[str] = None, device: str = "cpu", num_classes: int = 3000):
    """
    Create and return an XGait inference engine using the official implementation
    
    Args:
        model_path: Path to XGait model weights (Gait3D-XGait-120000.pt recommended)
        device: Device to use for inference
        num_classes: Number of identity classes (3000 for Gait3D)
        
    Returns:
        XGaitAdapter instance (using official XGait implementation)
    """
    # Look for default model path in weights directory
    if model_path is None:
        current_dir = Path(__file__).parent.parent.parent
        potential_paths = [
            current_dir / "weights" / "Gait3D-XGait-120000.pt",
            current_dir / "weights" / "xgait_model.pth",
            current_dir / "weights" / "xgait.pt"
        ]
        
        for path in potential_paths:
            if path.exists():
                model_path = str(path)
                break
    
    # Use the adapter to provide backward compatibility with the official implementation
    return XGaitAdapter(model_path=model_path, device=device, num_classes=num_classes)


if __name__ == "__main__":
    # Test the official XGait model
    print("🧪 Testing Official XGait Model")
    
    # Create XGait inference using official implementation
    xgait = create_xgait_inference(device="cpu")
    
    print(f"✅ Official XGait initialized")
    print(f"📊 Model loaded: {xgait.model_loaded}")
    print(f"🎯 Implementation: {type(xgait).__name__}")
    print(f"🔧 Backend: {type(xgait.xgait_official).__name__}")
    
    # Test with dummy data
    import numpy as np
    test_silhouettes = [np.random.randint(0, 255, (64, 44), dtype=np.uint8) for _ in range(30)]
    test_parsing = [np.random.randint(0, 19, (64, 44), dtype=np.uint8) for _ in range(30)]
    
    # Extract features using official implementation
    features = xgait.extract_features([test_silhouettes], [test_parsing])
    
    print(f"✅ Extracted features: shape {features.shape}")
    print(f"🎯 Feature dimension: {features.shape[1] if len(features) > 0 else 'N/A'}")
    
    # Test gallery functionality
    if len(features) > 0:
        xgait.add_to_gallery("person_1", features[0])
        person_id, confidence = xgait.identify_person(features[0])
        print(f"🔍 Identification test: {person_id} (confidence: {confidence:.3f})")
        print(f"📚 Gallery summary: {xgait.get_gallery_summary()}")
    
    print(f"📈 Model utilization report:")
    report = xgait.get_model_utilization_report()
    for key, value in report.items():
        if key == 'recommendations':
            print(f"   💡 Recommendations: {value}")
        else:
            print(f"   {key}: {value}")
