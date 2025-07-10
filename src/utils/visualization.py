"""
Modern visualization utilities for person tracking
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Set
import colorsys

class TrackingVisualizer:
    """
    Modern, minimal visualization for tracking results
    """
    def __init__(self):
        # Modern color palette - vibrant but professional
        self.modern_colors = [
            (0, 122, 255),    # iOS Blue
            (52, 199, 89),    # iOS Green  
            (255, 69, 58),    # iOS Red
            (255, 159, 10),   # iOS Orange
            (175, 82, 222),   # iOS Purple
            (255, 45, 85),    # iOS Pink
            (100, 210, 255),  # iOS Light Blue
            (255, 214, 10),   # iOS Yellow
        ]
        
        # Typography settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        
    def get_track_color(self, track_id: int, confidence: float = 1.0) -> Tuple[int, int, int]:
        """Get modern color for track with confidence-based alpha"""
        base_color = self.modern_colors[track_id % len(self.modern_colors)]
        
        # Adjust brightness based on confidence
        if confidence < 0.7:
            # Dim color for low confidence
            return tuple(int(c * 0.6) for c in base_color)
        return base_color
    
    def draw_tracking_results(self, 
                            frame: np.ndarray,
                            tracking_results: List[Tuple[int, np.ndarray, float]],
                            track_history: Dict,
                            stable_tracks: Set[int],
                            frame_count: int,
                            max_track_id: int,
                            identification_results: Dict = None,
                            identification_confidence: Dict = None,
                            gallery_stats: Dict = None,
                            identification_stats: Dict = None,
                            current_fps: float = None,
                            avg_fps: float = None,
                            is_new_identity: Dict[int, bool] = None ,
                            gallery_loaded: bool = False
                            ) -> np.ndarray:
        """
        Draw modern, minimal tracking annotations
        
        Args:
            frame: Input frame
            tracking_results: List of (track_id, box, confidence) tuples
            identification_results: Person identification results
            identification_confidence: Identification confidence scores
            gallery_stats: Gallery statistics
            identification_stats: Identification statistics
            
        Returns:
            Annotated frame with modern design
        """
        annotated_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw person detections with modern style
        for track_id, box, conf in tracking_results:
            x1, y1, x2, y2 = box.astype(int)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            # Get track color and status
            is_stable = track_id in stable_tracks
            color = self.get_track_color(track_id, conf)

            # Determine if new or from gallery
            is_new = False
            if is_new_identity and track_id in is_new_identity:
                is_new = is_new_identity[track_id]

            # Change border color for new/gallery
            border_color = (0, 255, 0) if is_new else (0, 122, 255)  # Green for new, Blue for gallery

            # Modern bounding box with rounded corners effect
            self._draw_modern_bbox(annotated_frame, (x1, y1, x2, y2), border_color, is_stable)

            # Track ID badge - minimal design
            self._draw_track_badge(annotated_frame, track_id, (x1, y1), color)

            # Person identification label (if available)
            if identification_results and track_id in identification_results:
                person_id = identification_results[track_id]
                id_conf = identification_confidence.get(track_id, 0.0)
                if person_id != "Unknown":
                    # Color code the label based on similarity score
                    if id_conf >= 0.92:
                        label_color = (0, 255, 0)  # Green for high similarity (>= 0.92)
                    elif id_conf >= 0.90:
                        label_color = (0, 255, 255)  # Yellow for medium similarity (0.90-0.92)
                    else:
                        label_color = (0, 165, 255)  # Orange for low similarity (< 0.90)
                    
                    self._draw_person_label(annotated_frame, person_id, id_conf, (x1, y2), label_color)
                    # Add NEW/GALLERY label
                    label_text = "NEW" if is_new else "GALLERY"
                    gallery_color = (0, 255, 0) if is_new else (0, 122, 255)
                    cv2.putText(annotated_frame, label_text, (x1, y2 + 25),
                                self.font, 0.6, gallery_color, 2)

        # Modern status overlay - minimal and clean
        self._draw_modern_status(annotated_frame, len(tracking_results), frame_count, w, h,
                                gallery_stats, identification_stats, current_fps, avg_fps,gallery_loaded= gallery_loaded)
        
        return annotated_frame
    
    def _draw_modern_bbox(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                         color: Tuple[int, int, int], is_stable: bool):
        """Draw modern bounding box with subtle styling"""
        x1, y1, x2, y2 = bbox
        
        # Main bounding box - thin, clean lines
        thickness = 3 if is_stable else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Corner accents for modern look
        corner_length = 20
        corner_thickness = 4
        
        # Top-left corner
        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, corner_thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, corner_thickness)
        
        # Top-right corner  
        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, corner_thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, corner_thickness)
        
        # Bottom-left corner
        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, corner_thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, corner_thickness)
        
        # Bottom-right corner
        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, corner_thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, corner_thickness)
    
    def _draw_track_badge(self, frame: np.ndarray, track_id: int, 
                         position: Tuple[int, int], color: Tuple[int, int, int]):
        """Draw minimal track ID badge"""
        x, y = position
        
        # Badge background - subtle rounded rectangle effect
        badge_text = str(track_id)
        (text_w, text_h), baseline = cv2.getTextSize(badge_text, self.font, 0.5, 2)
        
        # Background rectangle with padding
        padding = 8
        badge_x1 = x - 2
        badge_y1 = y - text_h - padding - 5
        badge_x2 = x + text_w + padding
        badge_y2 = y - 5
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # White text on colored background
        text_x = x + padding // 2
        text_y = y - padding
        cv2.putText(frame, badge_text, (text_x, text_y), 
                   self.font, 0.5, (255, 255, 255), 2)
    
    def _draw_person_label(self, frame: np.ndarray, person_id: str, confidence: float,
                          position: Tuple[int, int], color: Tuple[int, int, int]):
        """Draw person identification label with similarity score"""
        x, y = position
        
        # Clean label text with similarity score
        if confidence > 0:
            label_text = f"{person_id} ({confidence:.2f})"
        else:
            label_text = f"{person_id}"
        (text_w, text_h), baseline = cv2.getTextSize(label_text, self.font, self.font_scale, self.font_thickness)
        
        # Background with rounded corners effect - color coded by similarity
        padding = 10
        label_x1 = x - 2
        label_y1 = y + 8
        label_x2 = x + text_w + padding
        label_y2 = y + text_h + padding + 8
        
        # Color background based on confidence level
        if confidence >= 0.92:
            bg_color = (0, 40, 0)  # Dark green for high similarity (>= 0.92)
        elif confidence >= 0.90:
            bg_color = (40, 40, 0)  # Dark yellow for medium similarity (0.90-0.92)
        elif confidence > 0.85:
            bg_color = (40, 20, 0)  # Dark orange for low similarity (0.85-0.90)
        else:
            bg_color = (40, 40, 40)  # Dark gray for very low similarity (< 0.85)
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (label_x1, label_y1), (label_x2, label_y2), bg_color, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Colored accent line
        cv2.rectangle(frame, (label_x1, label_y1), (label_x1 + 3, label_y2), color, -1)
        
        # White text
        text_x = x + padding // 2
        text_y = y + text_h + padding
        cv2.putText(frame, label_text, (text_x, text_y), 
                   self.font, self.font_scale, (255, 255, 255), self.font_thickness)
    
    def _draw_modern_status(self, frame: np.ndarray, active_tracks: int, 
                           frame_count: int, width: int, height: int,
                           gallery_stats: dict = None, identification_stats: dict = None,
                           current_fps: float = None, avg_fps: float = None,gallery_loaded: bool = False):
        """Draw minimal status overlay with optional identification info and FPS"""
        # Build status text
        status_lines = [f"Frame {frame_count} • {active_tracks} tracked"]
        if gallery_loaded:
            status_lines.append("Gallery loaded")
        else:
            status_lines.append("Gallery not loaded")
        if current_fps is not None and avg_fps is not None:
            status_lines.append(f"FPS: {current_fps:.1f} (avg: {avg_fps:.1f})")
        elif current_fps is not None:
            status_lines.append(f"FPS: {current_fps:.1f}")
        elif avg_fps is not None:
            status_lines.append(f"Avg FPS: {avg_fps:.1f}")
        # Add identification stats if available
        if identification_stats:
            identified = identification_stats.get('identified_tracks', 0)
            total = identification_stats.get('total_tracks', 0)
            gallery_count = identification_stats.get('gallery_persons', 0)
            if total > 0:
                status_lines.append(f"Gallery: {gallery_count} • ID: {identified}/{total}")
        
        # Add similarity legend
        status_lines.append("Similarity: Green≥0.92 Yellow≥0.90 Orange≥0.85")
        
        # Calculate dimensions for all lines
        max_text_w = 0
        total_text_h = 0
        line_heights = []
        
        for line in status_lines:
            (text_w, text_h), baseline = cv2.getTextSize(line, self.font, 0.5, 2)
            max_text_w = max(max_text_w, text_w)
            line_heights.append(text_h)
            total_text_h += text_h + 5  # 5px spacing between lines
        
        # Position in top-right with margin
        margin = 15
        text_x = width - max_text_w - margin - 10
        text_y = margin
        
        # Subtle background
        overlay = frame.copy()
        bg_padding = 8
        cv2.rectangle(overlay, 
                     (text_x - bg_padding, text_y - bg_padding), 
                     (text_x + max_text_w + bg_padding, text_y + total_text_h + bg_padding), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        # Draw text lines
        current_y = text_y + line_heights[0]
        for i, (line, line_height) in enumerate(zip(status_lines, line_heights)):
            color = (255, 255, 255) if i == 0 else (200, 200, 200)
            cv2.putText(frame, line, (text_x, current_y), 
                       self.font, 0.5, color, 2)
            if i < len(status_lines) - 1:
                current_y += line_height + 8
    
    def print_summary(self, max_track_id: int, total_frames: int, target_people: int = 7) -> None:
        """Print modern tracking summary"""
        print("\n" + "═" * 60)
        print("🎯 TRACKING ANALYSIS COMPLETE")
        print("═" * 60)
        print(f"📊 Tracks Created: {max_track_id}")
        print(f"🎬 Frames Processed: {total_frames}")
        print(f"👥 Target People: {target_people}")
        
        accuracy = (target_people / max_track_id) * 100 if max_track_id > 0 else 0
        print(f"✨ Efficiency: {accuracy:.1f}%")
        
        if max_track_id <= target_people + 1:
            print("🏆 EXCELLENT - Minimal ID fragmentation!")
        elif max_track_id <= target_people + 3:
            print("👍 GOOD - Acceptable tracking performance")
        else:
            print("⚠️  Needs optimization - Consider tuning thresholds")
        print("═" * 60)

    def draw_legacy_tracking_results(self, 
                            frame: np.ndarray,
                            tracking_results: List[Tuple[int, np.ndarray, float]],
                            track_history: Dict,
                            stable_tracks: Set[int],
                            frame_count: int,
                            max_track_id: int,
                            current_fps: float = None,
                            avg_fps: float = None) -> np.ndarray:
        """
        Draw tracking results on frame (legacy method for compatibility)
        
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
            color = self.get_track_color(track_id, conf)
            
            # Draw bounding box
            thickness = 5 if is_stable else 3
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
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
        self._draw_info_overlay(annotated_frame, len(tracking_results), max_track_id, frame_count, current_fps, avg_fps)
        
        return annotated_frame
    
    def _draw_info_overlay(self, 
                          frame: np.ndarray, 
                          active_tracks: int, 
                          max_track_id: int, 
                          frame_count: int,
                          current_fps: float = None,
                          avg_fps: float = None) -> None:
        """Draw information overlay on frame with FPS info"""
        
        info_lines = [
            f"Frame: {frame_count} | Active: {active_tracks} | Max ID: {max_track_id}",
            "Method: TransReID Model + Appearance Matching"
        ]
        if current_fps is not None and avg_fps is not None:
            info_lines.append(f"FPS: {current_fps:.1f} (avg: {avg_fps:.1f})")
        elif current_fps is not None:
            info_lines.append(f"FPS: {current_fps:.1f}")
        elif avg_fps is not None:
            info_lines.append(f"Avg FPS: {avg_fps:.1f}")
        for i, line in enumerate(info_lines):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            weight = 2
            cv2.putText(frame, line, (10, 30 + i * 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, weight)

    def print_legacy_summary(self, 
                     max_track_id: int, 
                     total_frames: int, 
                     target_people: int = 7) -> None:
        """
        Print tracking summary (legacy method)
        
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

class InferenceVisualizer:
    """Simple visualizer for inference pipeline"""
    
    def __init__(self):
        self.color_palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (255, 192, 203), (0, 128, 0), (128, 128, 0), (0, 0, 128)
        ]
    
    def get_person_color(self, person_id: int) -> Tuple[int, int, int]:
        """Get color for a person ID"""
        return self.color_palette[person_id % len(self.color_palette)]


def create_inference_visualizer() -> InferenceVisualizer:
    """Create an InferenceVisualizer instance"""
    return InferenceVisualizer()

class VideoWriter:
    """
    Modern video writer for saving annotated tracking results
    """
    def __init__(self, output_path: str, fps: float, frame_size: Tuple[int, int], 
                 codec: str = "mp4v", quality: float = 0.8):
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        self.quality = quality
        self.writer = None
        self.frame_count = 0
        
    def __enter__(self):
        """Context manager entry"""
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
        
    def open(self):
        """Initialize the video writer"""
        try:
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(
                self.output_path, 
                fourcc, 
                self.fps, 
                self.frame_size
            )
            if not self.writer.isOpened():
                raise RuntimeError(f"Failed to open video writer for {self.output_path}")
            print(f"📹 Video writer initialized: {self.output_path}")
            print(f"   • Resolution: {self.frame_size[0]}x{self.frame_size[1]}")
            print(f"   • FPS: {self.fps}")
            print(f"   • Codec: {self.codec}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize video writer: {e}")
    
    def write_frame(self, frame: np.ndarray):
        """Write a frame to the video"""
        if self.writer is None:
            raise RuntimeError("Video writer not initialized")
        
        # Ensure frame is the correct size
        if frame.shape[:2] != (self.frame_size[1], self.frame_size[0]):
            frame = cv2.resize(frame, self.frame_size)
        
        self.writer.write(frame)
        self.frame_count += 1
    
    def release(self):
        """Release the video writer"""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print(f"📹 Video saved: {self.output_path} ({self.frame_count} frames)")
    
    def is_opened(self) -> bool:
        """Check if video writer is opened"""
        return self.writer is not None and self.writer.isOpened()
