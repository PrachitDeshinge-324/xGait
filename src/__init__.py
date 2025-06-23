"""
Person Tracking System with Custom TransReID
"""
__version__ = "1.0.0"
__author__ = "Person Tracking Team"

from .config import SystemConfig, ModelConfig, TrackerConfig, VideoConfig
from .trackers.person_tracker import PersonTracker
from .models.reid_model import ReIDModel, create_reid_model
from .utils.visualization import TrackingVisualizer

__all__ = [
    "SystemConfig",
    "ModelConfig", 
    "TrackerConfig",
    "VideoConfig",
    "xgaitConfig",
    "PersonTracker",
    "ReIDModel",
    "create_reid_model",
    "TrackingVisualizer"
]
