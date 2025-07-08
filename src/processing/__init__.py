"""
Processing module for person tracking application.
"""

from .video_processor import VideoProcessor
from .gait_processor import GaitProcessor
from .statistics_manager import StatisticsManager
from .enhanced_identity_manager import IdentityManager

__all__ = [
    'VideoProcessor',
    'GaitProcessor', 
    'StatisticsManager',
    'IdentityManager'
]
