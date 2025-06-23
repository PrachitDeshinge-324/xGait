"""
Configuration settings for the Person Tracking System
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch

def get_device() -> str:
    """Determine the device to use for PyTorch"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def get_device_config(device: str) -> Dict[str, Any]:
    """Get device-specific configuration"""
    if device == "cuda":
        return {
            "dtype": torch.float16,  # Use float16 for CUDA for memory efficiency
            "autocast": True,
            "compile": True,
            "memory_format": torch.channels_last,
            "batch_size_multiplier": 2.0
        }
    elif device == "mps":
        return {
            "dtype": torch.float32,  # MPS works best with float32
            "autocast": False,  # MPS doesn't support autocast yet
            "compile": False,  # MPS compilation can be unstable
            "memory_format": torch.contiguous_format,
            "batch_size_multiplier": 1.0
        }
    else:  # CPU
        return {
            "dtype": torch.float32,  # CPU standard precision
            "autocast": False,
            "compile": False,
            "memory_format": torch.contiguous_format,
            "batch_size_multiplier": 0.5
        }

@dataclass
class ModelConfig:
    """Model configuration settings"""
    yolo_model_path: str = "weights/yolo11m.pt"
    transreid_model_path: str = "weights/transreid_vitbase.pth"
    xgait_model_path: str = "weights/Gait3D-XGait-120000.pt"
    parsing_model_path: str = "weights/parsing_u2net.pth"
    silhoutte_model_path: str = "weights/u2net_fixed.pth"
    device: str = get_device()
    
    # Model-specific device overrides for compatibility
    xgait_device: str = "cpu"  # Force XGait to use CPU due to MPS compatibility issues
    
    def __post_init__(self):
        """Initialize device-specific settings"""
        self.device_config = get_device_config(self.device)
        self.dtype = self.device_config["dtype"]
        self.use_autocast = self.device_config["autocast"]
        self.use_compile = self.device_config["compile"]
        self.memory_format = self.device_config["memory_format"]
        
        # XGait-specific configuration for CPU fallback
        self.xgait_config = get_device_config(self.xgait_device)
    
    def get_model_device(self, model_name: str) -> str:
        """Get the appropriate device for a specific model"""
        if model_name.lower() == "xgait":
            return self.xgait_device
        return self.device
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get the appropriate device configuration for a specific model"""
        if model_name.lower() == "xgait":
            return self.xgait_config
        return self.device_config

@dataclass
class TrackerConfig:
    """Tracker configuration settings"""
    # Detection parameters
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.6
    
    # ReID parameters
    similarity_threshold: float = 0.25
    max_missing_frames: int = 75
    
    # XGait identification parameters
    identification_threshold: float = 0.6
    sequence_length: int = 10
    
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
    max_frames: Optional[int] = None  # Maximum number of frames to process (for testing)

@dataclass
class xgaitConfig:
    """XGait-specific configuration"""
    # Sequence buffer settings
    sequence_buffer_size = 100
    min_sequence_length = 10

    # Feature extraction settings
    xgait_extraction_interval = 5

    # Similarity threshold for identification
    similarity_threshold = 0.99
    device: str = "cuda"

@dataclass
class SystemConfig:
    """Main system configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    xgait: xgaitConfig = field(default_factory=xgaitConfig)  # Ensure xgaitConfig is included here

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
