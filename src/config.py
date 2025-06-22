"""
Configuration settings for the Person Tracking System
"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Model configuration settings"""
    yolo_model_path: str = "weights/yolo11m.pt"
    transreid_model_path: str = "weights/transreid_vitbase.pth"
    device: str = "mps"  # "cuda", "mps", or "cpu"

@dataclass
class TrackerConfig:
    """Tracker configuration settings"""
    # Detection parameters
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.6
    
    # ReID parameters
    similarity_threshold: float = 0.25
    max_missing_frames: int = 75
    
    # Tracking parameters
    track_history_length: int = 100
    stable_track_threshold: int = 20
    id_switch_distance_threshold: float = 80.0

@dataclass
class VideoConfig:
    """Video processing configuration"""
    input_path: str = "input/3c.mp4"
    output_path: Optional[str] = None
    display_window: bool = True
    save_output: bool = False

@dataclass
class SystemConfig:
    """Main system configuration"""
    model: ModelConfig = ModelConfig()
    tracker: TrackerConfig = TrackerConfig()
    video: VideoConfig = VideoConfig()
    
    # System settings
    verbose: bool = True
    debug_mode: bool = False
    
    @classmethod
    def load_default(cls) -> 'SystemConfig':
        """Load default configuration optimized for multi-person tracking"""
        return cls()
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        # Check if model files exist
        if not os.path.exists(self.model.yolo_model_path):
            raise FileNotFoundError(f"YOLO model not found: {self.model.yolo_model_path}")
        
        if not os.path.exists(self.video.input_path):
            raise FileNotFoundError(f"Input video not found: {self.video.input_path}")
        
        # Validate thresholds
        if not 0.0 <= self.tracker.confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        
        if not 0.0 <= self.tracker.similarity_threshold <= 1.0:
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
        
        return True
