"""
Statistics management module for tracking and performance metrics.
"""

import sys
import os
from collections import defaultdict
from typing import List, Tuple, Dict, Set

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


class StatisticsManager:
    """Manages tracking statistics and performance metrics"""
    
    def __init__(self, config):
        self.config = config
        
        # Tracking statistics
        self.track_history = defaultdict(list)
        self.id_switches = []
        self.stable_tracks: Set[int] = set()
        self.new_id_creations = []
        self.max_track_id_seen = 0
    
    def update_statistics(self, tracking_results: List[Tuple[int, any, float]], frame_count: int) -> None:
        """
        Update tracking statistics for the current frame.
        
        Args:
            tracking_results: List of (track_id, box, confidence) tuples
            frame_count: Current frame number
        """
        current_tracks = {}
        for track_id, box, conf in tracking_results:
            x1, y1, x2, y2 = box.astype(int)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Monitor new ID creation
            if track_id > self.max_track_id_seen:
                self.max_track_id_seen = track_id
                self.new_id_creations.append((track_id, frame_count, center_x, center_y))
                
                # Debug new ID creation
                if frame_count > 50 and self.config.debug_mode:
                    print(f"🆕 New ID {track_id} at frame {frame_count}, position ({center_x}, {center_y})")
            
            # Store current track info
            current_tracks[track_id] = (center_x, center_y, x1, y1, x2, y2, conf)
            
            # Update track history
            self.track_history[track_id].append((center_x, center_y, frame_count, conf))
            
            # Mark tracks as stable after threshold frames
            if len(self.track_history[track_id]) >= self.config.tracker.stable_track_threshold:
                self.stable_tracks.add(track_id)
            
            # Keep only recent history
            if len(self.track_history[track_id]) > self.config.tracker.track_history_length:
                self.track_history[track_id].pop(0)
    
    def get_performance_rating(self) -> str:
        """
        Get performance rating based on track ID creation.
        
        Returns:
            Performance rating string
        """
        if self.max_track_id_seen <= 8:
            return "EXCELLENT"
        elif self.max_track_id_seen <= 12:
            return "GOOD"
        elif self.max_track_id_seen <= 16:
            return "MODERATE"
        else:
            return "POOR"
    
    def print_final_statistics(self) -> None:
        """Print final tracking statistics"""
        print(f"\n📊 Detailed Statistics:")
        print(f"   • Total track IDs created: {self.max_track_id_seen}")
        print(f"   • New ID events: {len(self.new_id_creations)}")
        print(f"   • ID switches detected: {len(self.id_switches)}")
        print(f"   • Stable tracks: {len(self.stable_tracks)}")
        print(f"   • Overall performance: {self.get_performance_rating()}")
        
        # Configuration summary
        print(f"\n⚙️  Configuration Used:")
        print(f"   • Similarity threshold: {self.config.tracker.similarity_threshold}")
        print(f"   • Max missing frames: {self.config.tracker.max_missing_frames}")
        print(f"   • Confidence threshold: {self.config.tracker.confidence_threshold}")
    
    def get_statistics(self) -> Dict:
        """
        Get current statistics as a dictionary.
        
        Returns:
            Dictionary containing current statistics
        """
        return {
            'max_track_id_seen': self.max_track_id_seen,
            'new_id_creations': len(self.new_id_creations),
            'id_switches': len(self.id_switches),
            'stable_tracks': len(self.stable_tracks),
            'performance_rating': self.get_performance_rating()
        }
