"""
XGait Model Entry Point
Uses the official Gait3D-Benchmark implementation for maximum performance

This is the main entry point for XGait functionality in the codebase.
All XGait-related components are organized in the xgait/ folder.
"""
from pathlib import Path
from typing import Optional
import logging
import sys

# Add parent directory for imports when running directly
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.models.xgait.adapter import XGaitAdapter
else:
    from .xgait.adapter import XGaitAdapter

logger = logging.getLogger(__name__)


def create_xgait_inference(model_path: Optional[str] = None, device: str = None, num_classes: int = 3000):
    """
    Create and return an XGait inference engine using the official implementation
    
    This is the main entry point used by the main application and other parts of the codebase.
    
    Args:
        model_path: Path to XGait model weights (Gait3D-XGait-120000.pt recommended)
        device: Device to use for inference (uses XGait-specific device if None)
        num_classes: Number of identity classes (3000 for Gait3D)
        
    Returns:
        XGaitAdapter instance (using official XGait implementation)
    """
    if device is None:
        from ..config import get_xgait_device
        device = get_xgait_device()
    
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
