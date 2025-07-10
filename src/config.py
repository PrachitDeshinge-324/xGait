"""
Configuration settings for the Person Tracking System

SIMILARITY THRESHOLD ANALYSIS:
Based on empirical analysis of XGait embeddings:
- Intra-person similarities: ~0.94 (mean), range 0.904-0.984
- Inter-person similarities: ~0.89 (mean), range 0.870-0.909
- Optimal threshold: 0.91 (100% precision, 94.1% recall)
- High confidence threshold: 0.93 (for very confident matches)

THRESHOLD HIERARCHY:
- tracker.similarity_threshold: 0.4 (for basic tracking/ReID)
- xgait.similarity_threshold: 0.91 (for XGait person identification)
- identity.similarity_threshold: 0.91 (for identity management)
- identity.high_confidence_threshold: 0.93 (for high confidence matches)
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch

def get_device() -> str:
    """Determine the best available device for PyTorch"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def get_global_device() -> str:
    """Get the global device for all models"""
    return get_device()

def get_xgait_device() -> str:
    """Get the appropriate device for XGait model (CPU when main device is MPS)"""
    main_device = get_global_device()
    if main_device == "mps":
        return "cpu"  # XGait has MPS compatibility issues
    else:
        return main_device  # Use best available device (CUDA or CPU)

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
    device: str = get_global_device()
    
    # Model-specific device overrides for compatibility
    xgait_device: str = get_xgait_device()  # Use global device logic for XGait
    
    def __post_init__(self):
        """Initialize device-specific settings"""
        self.device_config = get_device_config(self.device)
        self.dtype = self.device_config["dtype"]
        self.use_autocast = self.device_config["autocast"]
        self.use_compile = self.device_config["compile"]
        self.memory_format = self.device_config["memory_format"]
        
        # XGait-specific configuration 
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
    
    # ReID parameters - Further optimized to reduce ID fragmentation
    similarity_threshold: float = 0.4  # Reduced from 0.6 to 0.4 for more flexible matching
    max_missing_frames: int = 60       # Increased from 30 to 60 to keep tracks alive longer
    
    # XGait identification parameters
    identification_threshold: float = 0.91  # Updated to match XGait optimal threshold
    sequence_length: int = 10
    
    # Tracking parameters
    track_history_length: int = 100
    stable_track_threshold: int = 15   # Increased from 10 to 15 for more stable tracks
    id_switch_distance_threshold: float = 150.0  # Increased from 120 to 150 for even more tolerance

@dataclass
class VideoConfig:
    """Video processing configuration"""
    input_path: str = "input/3c.mp4"
    output_path: Optional[str] = None
    output_video_path: Optional[str] = None  # Path for saving annotated video
    display_window: bool = True
    show_progress_bar: bool = True  # Show progress bar in terminal
    save_output: bool = False
    save_annotated_video: bool = False  # Enable saving annotated video
    max_frames: Optional[int] = None  # Maximum number of frames to process (for testing)
    interactive_mode: bool = False  # Enable interactive mode for manual person identification
    # Video encoding settings
    output_fps: Optional[float] = None  # Auto-detect from input if None
    output_codec: str = "mp4v"  # Video codec for output
    output_quality: float = 0.8  # Quality factor (0.0-1.0)

@dataclass
class xgaitConfig:
    """XGait-specific configuration"""
    # Sequence buffer settings
    sequence_buffer_size = 100
    min_sequence_length = 10

    # Feature extraction settings
    xgait_extraction_interval = 5

    # Similarity threshold for identification - Optimized for XGait embeddings
    similarity_threshold = 0.91  # Updated to 0.91 based on XGait similarity analysis
    device: str = get_xgait_device()  # Use global XGait device logic

@dataclass
class IdentityConfig:
    """Identity system configuration settings"""
    # Gallery management
    gallery_path: str = "visualization_analysis/enhanced_gallery.json"
    backup_gallery_on_save: bool = True
    max_persons_in_gallery: int = 100
    
    # Enhanced gallery system settings
    use_enhanced_gallery: bool = True
    enhanced_gallery_path: str = "visualization_analysis/enhanced_gallery.json"
    max_embeddings_per_context: int = 5
    min_movement_confidence: float = 0.6
    movement_history_length: int = 10
    
    # Embedding quality and similarity thresholds
    min_quality_threshold: float = 0.3
    similarity_threshold: float = 0.91  # Updated to match XGait discrimination capability
    high_confidence_threshold: float = 0.93  # Updated to reflect high XGait similarity range
    
    # Embedding buffer management
    max_embeddings_per_person: int = 20
    min_embeddings_for_stable_prototype: int = 3
    
    # Prototype update strategies
    prototype_update_strategy: str = "weighted_average"  # "best_quality", "recent_average", "weighted_average"
    
    # Interactive naming settings
    auto_naming_enabled: bool = False  # Never auto-name, always prompt user
    skip_poor_quality_tracks: bool = True
    min_track_length_for_naming: int = 10
    
    # Visualization settings
    show_confidence_in_overlay: bool = True
    show_track_id_in_overlay: bool = True
    use_colored_boxes_for_known_persons: bool = True
    
    # Performance settings
    consolidation_interval: int = 500  # frames
    enable_periodic_cleanup: bool = True
    max_track_history_length: int = 1000

@dataclass
class SystemConfig:
    """Main system configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    xgait: xgaitConfig = field(default_factory=xgaitConfig)  # Ensure xgaitConfig is included here
    identity: IdentityConfig = field(default_factory=IdentityConfig)  # Include IdentityConfig

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
