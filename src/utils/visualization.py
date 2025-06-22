"""
Visualization utilities for person tracking
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Set
import colorsys

class TrackingVisualizer:
    """
    Handles visualization of tracking results
    """
    def __init__(self):
        self.colors = self._generate_colors(20)  # Pre-generate colors for tracks
        
    def _generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for track visualization"""
        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            bgr = tuple(int(c * 255) for c in reversed(rgb))
            colors.append(bgr)
        return colors
    
    def get_track_color(self, track_id: int, is_stable: bool = False) -> Tuple[int, int, int]:
        """
        Get color for a track ID
        
        Args:
            track_id: Track ID
            is_stable: Whether the track is stable
            
        Returns:
            BGR color tuple
        """
        base_colors = [
            (0, 255, 0),    # Bright Green - Stable
            (0, 150, 255),  # Orange - Stable
            (255, 0, 150),  # Pink - Stable
            (150, 255, 0),  # Lime - Stable
        ] if is_stable else [
            (0, 0, 255),    # Red - Unstable
            (0, 100, 255),  # Dark Orange - Unstable
        ]
        
        return base_colors[track_id % len(base_colors)]
    
    def draw_tracking_results(self, 
                            frame: np.ndarray,
                            tracking_results: List[Tuple[int, np.ndarray, float]],
                            track_history: Dict,
                            stable_tracks: Set[int],
                            frame_count: int,
                            max_track_id: int) -> np.ndarray:
        """
        Draw tracking results on frame
        
        Args:
            frame: Input frame
            tracking_results: List of (track_id, box, confidence) tuples
            track_history: Track history dictionary
            stable_tracks: Set of stable track IDs
            frame_count: Current frame number
            max_track_id: Maximum track ID seen
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Draw tracking results
        for track_id, box, conf in tracking_results:
            x1, y1, x2, y2 = box.astype(int)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Get color based on stability
            is_stable = track_id in stable_tracks
            color = self.get_track_color(track_id, is_stable)
            
            # Draw bounding box
            thickness = 5 if is_stable else 3
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # # Draw track history trail
            # if track_id in track_history and len(track_history[track_id]) > 1:
            #     points = [(x, y) for x, y, _, _ in track_history[track_id]]
            #     for k in range(1, len(points)):
            #         trail_thickness = 4 if is_stable else 2
            #         cv2.line(annotated_frame, points[k-1], points[k], color, trail_thickness)
            
            # Draw label
            stability_status = "STABLE" if is_stable else "UNSTABLE"
            track_age = len(track_history.get(track_id, []))
            label = f"ID:{track_id} ({conf:.2f})"
            
            # Label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 25),
                        (x1 + label_size[0] + 15, y1), color, -1)
            
            # Label text
            cv2.putText(annotated_frame, label, (x1 + 5, y1 - 12),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Stability indicator dot
            dot_color = (0, 255, 0) if is_stable else (0, 0, 255)
            cv2.circle(annotated_frame, (center_x, center_y), 8, dot_color, -1)
        
        # Add information overlay
        self._draw_info_overlay(annotated_frame, len(tracking_results), max_track_id, frame_count)
        
        return annotated_frame
    
    def _draw_info_overlay(self, 
                          frame: np.ndarray, 
                          active_tracks: int, 
                          max_track_id: int, 
                          frame_count: int) -> None:
        """Draw information overlay on frame"""
        
        info_lines = [
            # "CUSTOM TRANSREID TRACKER",
            f"Frame: {frame_count} | Active: {active_tracks} | Max ID: {max_track_id}",
            # f"Performance: {'EXCELLENT' if max_track_id <= 8 else 'GOOD' if max_track_id <= 12 else 'NEEDS TUNING'}",
            "Method: TransReID Model + Appearance Matching"
        ]
        
        for i, line in enumerate(info_lines):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            weight = 2
            cv2.putText(frame, line, (10, 30 + i * 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, weight)
        
        # Performance indicator
        # if max_track_id <= 8:
        #     status_text = f"✅ EXCELLENT - {max_track_id} IDs for 7 people!"
        #     status_color = (0, 255, 0)
        # elif max_track_id <= 12:
        #     status_text = f"⚠️ GOOD - {max_track_id} IDs (room for improvement)"
        #     status_color = (0, 255, 255)
        # else:
        #     status_text = f"🚨 NEEDS TUNING - {max_track_id} IDs"
        #     status_color = (0, 0, 255)
        
        # cv2.putText(frame, status_text, (10, 160),
        #           cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
    
    def print_summary(self, 
                     max_track_id: int, 
                     total_frames: int, 
                     target_people: int = 7) -> None:
        """
        Print tracking summary
        
        Args:
            max_track_id: Maximum track ID created
            total_frames: Total frames processed
            target_people: Expected number of people in scene
        """
        print("\n" + "=" * 70)
        print("CUSTOM TRANSREID TRACKER ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"Scene: {target_people} people total")
        print(f"Track IDs created: {max_track_id}")
        print(f"Frames processed: {total_frames}")
        
        accuracy = (target_people / max_track_id) * 100 if max_track_id > 0 else 0
        print(f"Accuracy: {accuracy:.1f}% ({target_people} people / {max_track_id} IDs)")
        
        if max_track_id <= target_people + 1:
            print("🎉 EXCELLENT: Near-perfect tracking!")
        elif max_track_id <= target_people + 3:
            print("✅ GOOD: Solid tracking performance!")
        elif max_track_id <= target_people * 2:
            print("⚠️ MODERATE: Room for improvement")
        else:
            print("🚨 POOR: Needs significant tuning")
        
        print(f"\n💡 Custom TransReID Benefits:")
        print(f"   • Appearance-based matching instead of motion-only")
        print(f"   • No reliance on YOLO's failing built-in ReID")
        print(f"   • Direct control over similarity thresholds")
        print(f"   • Robust performance in multi-person scenarios")
