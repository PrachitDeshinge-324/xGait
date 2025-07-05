"""
Video processing module for handling video input/output operations.
"""

import cv2
import sys
import os
from pathlib import Path
from typing import Tuple, Optional

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.visualization import VideoWriter


class VideoProcessor:
    """Handles video input/output operations"""
    
    def __init__(self, config):
        self.config = config
        self.video_writer = None
    
    def initialize_video(self, input_path: str) -> Tuple[cv2.VideoCapture, int, int, int, int]:
        """
        Initialize video capture and get properties.
        
        Args:
            input_path: Path to input video file
            
        Returns:
            Tuple of (cap, total_frames, fps, frame_width, frame_height)
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        return cap, total_frames, fps, frame_width, frame_height
    
    def initialize_video_writer(self, output_path: str, fps: int, frame_width: int, frame_height: int) -> Optional[VideoWriter]:
        """
        Initialize video writer for saving annotated video.
        
        Args:
            output_path: Path to output video file
            fps: Frame rate
            frame_width: Frame width
            frame_height: Frame height
            
        Returns:
            VideoWriter instance or None if initialization fails
        """
        try:
            output_fps = self.config.video.output_fps or fps
            codec = self.config.video.output_codec
            
            video_writer = VideoWriter(
                output_path=output_path,
                fps=output_fps,
                frame_size=(frame_width, frame_height),
                codec=codec,
                quality=self.config.video.output_quality
            )
            video_writer.open()
            return video_writer
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize video writer: {e}")
            return None
    
    def cleanup_video(self, cap: cv2.VideoCapture) -> None:
        """
        Clean up video capture resources.
        
        Args:
            cap: OpenCV VideoCapture instance
        """
        if cap:
            cap.release()
        
        if self.config.video.display_window:
            cv2.destroyAllWindows()
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
    
    def cleanup(self) -> None:
        """Clean up video writer resources"""
        if self.video_writer:
            try:
                self.video_writer.release()
                self.video_writer = None
            except:
                pass
