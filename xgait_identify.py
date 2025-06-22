#!/usr/bin/env python3
"""
XGait Person Identification Application
High-performance person identification using XGait, silhouette extraction, and human parsing
"""

import cv2
import sys
import traceback
import os
import argparse
import time
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import SystemConfig
from src.trackers.xgait_tracker import create_xgait_tracker
from src.utils.visualization import TrackingVisualizer

class XGaitIdentificationApp:
    """
    Main application for XGait-based person identification
    """
    def __init__(self, config: SystemConfig):
        self.config = config
        self.config.validate()
        
        # Initialize XGait tracker
        self.tracker = create_xgait_tracker(
            yolo_model_path=config.model.yolo_model_path,
            device=config.model.device,
            config=config.tracker,
            xgait_model_path=config.model.xgait_model_path,
            parsing_model_path=config.model.parsing_model_path,
            identification_threshold=config.tracker.identification_threshold
        )
        
        self.visualizer = TrackingVisualizer()
        
        # Statistics tracking
        self.track_history = defaultdict(list)
        self.person_identities = {}
        self.identity_changes = []
        self.max_track_id_seen = 0
        self.max_person_id_seen = 0
        
        print(f"🚀 XGait Identification App initialized")
        print(f"   Video: {config.video.input_path}")
        print(f"   YOLO model: {config.model.yolo_model_path}")
        print(f"   XGait model: {config.model.xgait_model_path}")
        print(f"   Parsing model: {config.model.parsing_model_path}")
        print(f"   Device: {config.model.device}")
        print(f"   Identification threshold: {config.tracker.identification_threshold}")
        
        # Print device information
        device_info = self.tracker.get_device_info()
        print(f"📱 Device Information:")
        print(f"   • Device: {device_info['device']}")
        print(f"   • Data type: {device_info['dtype']}")
        print(f"   • Autocast: {device_info['autocast']}")
        if 'gpu_name' in device_info:
            print(f"   • GPU: {device_info['gpu_name']}")
        if 'cuda_version' in device_info:
            print(f"   • CUDA version: {device_info['cuda_version']}")
        print()
    
    def process_video(self) -> None:
        """Process the input video and perform identification"""
        
        # Open video
        cap = cv2.VideoCapture(self.config.video.input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.config.video.input_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        if self.config.verbose:
            print(f"📹 Video properties: {total_frames} frames @ {fps} FPS")
            print("Press 'q' to quit, 'space' to pause, 's' to save gallery")
        
        frame_count = 0
        paused = False
        processing_fps = []
        
        while cap.isOpened():
            if not paused:
                success, frame = cap.read()
                if not success:
                    break
                
                frame_count += 1
                process_start = time.time()
                
                # Perform tracking and identification
                results = self.tracker.track_and_identify_persons(frame, frame_count)
                
                # Update statistics
                self._update_statistics(results, frame_count)
                
                # Calculate processing FPS
                process_time = time.time() - process_start
                current_fps = 1.0 / process_time if process_time > 0 else 0
                processing_fps.append(current_fps)
                
                # Create visualization
                if self.config.video.display_window:
                    annotated_frame = self._create_visualization(frame, results, frame_count, current_fps)
                    cv2.imshow("XGait Person Identification", annotated_frame)
                
                # Periodic memory cleanup
                if frame_count % 500 == 0:
                    self.tracker.clear_memory_cache()
                
                # Progress update with performance info
                if self.config.verbose and frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    avg_fps = np.mean(processing_fps[-100:]) if processing_fps else 0
                    
                    device_info = self.tracker.get_device_info()
                    memory_info = device_info.get('memory_usage', {})
                    
                    memory_str = ""
                    if 'allocated' in memory_info:
                        allocated_mb = memory_info['allocated'] / (1024**2)
                        memory_str = f" - GPU Memory: {allocated_mb:.1f}MB"
                    
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - "
                          f"FPS: {avg_fps:.1f} - Max Track: {self.max_track_id_seen} - "
                          f"Max Person: {self.max_person_id_seen}{memory_str}")
            
            # Handle key presses
            if self.config.video.display_window:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):  # Space to pause/unpause
                    paused = not paused
                    if paused:
                        print("⏸️  Paused - Press space to continue")
                    else:
                        print("▶️  Resumed")
                elif key == ord('s'):  # Save gallery
                    gallery_path = f"output/gallery_frame_{frame_count}.pt"
                    os.makedirs("output", exist_ok=True)
                    self.tracker.save_gallery(gallery_path)
                    print(f"💾 Gallery saved to {gallery_path}")
        
        # Cleanup
        cap.release()
        if self.config.video.display_window:
            cv2.destroyAllWindows()
        
        # Final cleanup
        self.tracker.clear_memory_cache()
        self.tracker.synchronize_device()
        
        # Print final results
        self._print_final_results(total_frames, processing_fps)
    
    def _create_visualization(self, frame: np.ndarray, results: List, frame_count: int, current_fps: float) -> np.ndarray:
        """Create visualization with tracking and identification info"""
        annotated_frame = frame.copy()
        
        # Draw tracking results
        for track_id, box, det_conf, person_id, id_conf in results:
            x1, y1, x2, y2 = box.astype(int)
            
            # Choose color based on person ID
            if person_id is not None:
                # Consistent color for each person ID
                color_idx = person_id % 10
                colors = [
                    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
                    (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 0), (128, 128, 0)
                ]
                color = colors[color_idx]
            else:
                color = (128, 128, 128)  # Gray for unidentified
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label_parts = [f"T{track_id}"]
            if person_id is not None:
                label_parts.append(f"P{person_id}")
                if id_conf > 0:
                    label_parts.append(f"{id_conf:.2f}")
            
            label = " | ".join(label_parts)
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw frame info
        info_text = [
            f"Frame: {frame_count}",
            f"FPS: {current_fps:.1f}",
            f"Tracks: {len(results)}",
            f"Max Track ID: {self.max_track_id_seen}",
            f"Max Person ID: {self.max_person_id_seen}"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(annotated_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 25
        
        return annotated_frame
    
    def _update_statistics(self, results: List, frame_count: int) -> None:
        """Update tracking and identification statistics"""
        
        for track_id, box, det_conf, person_id, id_conf in results:
            # Update max IDs seen
            self.max_track_id_seen = max(self.max_track_id_seen, track_id)
            if person_id is not None:
                self.max_person_id_seen = max(self.max_person_id_seen, person_id)
            
            # Track person identity changes
            if track_id in self.person_identities:
                if self.person_identities[track_id] != person_id:
                    self.identity_changes.append({
                        'frame': frame_count,
                        'track_id': track_id,
                        'old_person_id': self.person_identities[track_id],
                        'new_person_id': person_id
                    })
            
            self.person_identities[track_id] = person_id
            
            # Update track history
            x1, y1, x2, y2 = box
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            self.track_history[track_id].append((center_x, center_y, frame_count, person_id, id_conf))
            
            # Keep only recent history
            if len(self.track_history[track_id]) > self.config.tracker.track_history_length:
                self.track_history[track_id].pop(0)
    
    def _print_final_results(self, total_frames: int, processing_fps: List[float]) -> None:
        """Print comprehensive final results"""
        
        # Get performance statistics
        stats = self.tracker.get_statistics()
        performance = stats['performance']
        device_info = stats['device_info']
        
        print(f"\n📊 Final XGait Identification Results:")
        print(f"{'='*60}")
        
        # Basic statistics
        print(f"📈 Basic Statistics:")
        print(f"   • Total frames processed: {total_frames}")
        print(f"   • Max track ID: {self.max_track_id_seen}")
        print(f"   • Max person ID: {self.max_person_id_seen}")
        print(f"   • Identity changes: {len(self.identity_changes)}")
        
        # Performance statistics
        if processing_fps:
            avg_fps = np.mean(processing_fps)
            print(f"   • Average processing FPS: {avg_fps:.2f}")
            print(f"   • Min/Max FPS: {np.min(processing_fps):.2f}/{np.max(processing_fps):.2f}")
        
        # Detailed timing statistics
        print(f"\n⏱️  Timing Breakdown:")
        tracker_perf = performance.get('tracker', {})
        
        for stage, times in tracker_perf.items():
            if times.get('count', 0) > 0:
                print(f"   • {stage.capitalize()}: {times['mean']*1000:.2f}ms ± {times['std']*1000:.2f}ms")
        
        id_perf = performance.get('identification_processor', {})
        print(f"   Identification Pipeline:")
        for stage, times in id_perf.items():
            if times.get('count', 0) > 0:
                print(f"     - {stage.capitalize()}: {times['mean']*1000:.2f}ms ± {times['std']*1000:.2f}ms")
        
        # Identification statistics
        id_stats = performance.get('identification', {})
        print(f"\n🎯 Identification Statistics:")
        print(f"   • Active tracks: {id_stats.get('active_tracks', 0)}")
        print(f"   • Identified tracks: {id_stats.get('identified_tracks', 0)}")
        print(f"   • Average confidence: {id_stats.get('average_confidence', 0):.3f}")
        
        gallery_stats = id_stats.get('gallery_stats', {})
        print(f"   • Gallery identities: {gallery_stats.get('total_identities', 0)}")
        
        # Device information
        print(f"\n💻 Device Performance:")
        print(f"   • Device: {device_info.get('device', 'Unknown')}")
        print(f"   • Data type: {device_info.get('dtype', 'Unknown')}")
        if 'gpu_name' in device_info:
            print(f"   • GPU: {device_info['gpu_name']}")
        
        memory_info = device_info.get('memory_usage', {})
        if 'allocated' in memory_info:
            allocated_mb = memory_info['allocated'] / (1024**2)
            cached_mb = memory_info.get('cached', 0) / (1024**2)
            print(f"   • GPU memory: {allocated_mb:.1f}MB allocated, {cached_mb:.1f}MB cached")
        
        # Performance assessment
        if self.max_person_id_seen <= 8:
            performance_grade = "EXCELLENT"
        elif self.max_person_id_seen <= 12:
            performance_grade = "GOOD"
        elif self.max_person_id_seen <= 16:
            performance_grade = "MODERATE"
        else:
            performance_grade = "NEEDS IMPROVEMENT"
        
        print(f"\n🏆 Overall Performance: {performance_grade}")
        print(f"   Expected ~7 unique persons in video")
        print(f"   Detected {self.max_person_id_seen} unique person IDs")
        
        if len(self.identity_changes) > 0:
            print(f"\n⚠️  Identity Changes Detected: {len(self.identity_changes)}")
            for change in self.identity_changes[:5]:  # Show first 5
                print(f"     Frame {change['frame']}: Track {change['track_id']} "
                      f"changed from Person {change['old_person_id']} to {change['new_person_id']}")
            if len(self.identity_changes) > 5:
                print(f"     ... and {len(self.identity_changes) - 5} more")
        
        print(f"\n⚙️  Configuration Used:")
        print(f"   • Identification threshold: {self.config.tracker.identification_threshold}")
        print(f"   • Confidence threshold: {self.config.tracker.confidence_threshold}")
        print(f"   • Max missing frames: {self.config.tracker.max_missing_frames}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="XGait Person Identification")
    parser.add_argument("--video", type=str, default="input/3c.mp4", help="Input video path")
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "cuda", "mps"], help="Device to use")
    parser.add_argument("--xgait-model", type=str, default="weights/Gait3D-XGait-120000.pt", help="XGait model path")
    parser.add_argument("--parsing-model", type=str, default="weights/schp_resnet101.pth", help="Human parsing model path")
    parser.add_argument("--yolo-model", type=str, default="weights/yolo11m.pt", help="YOLO model path")
    parser.add_argument("--id-threshold", type=float, default=0.6, help="Identification threshold")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--no-display", action="store_true", help="Disable video display")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--load-gallery", type=str, help="Load existing person gallery")
    
    args = parser.parse_args()
    
    # Create configuration
    config = SystemConfig.load_default()
    config.video.input_path = args.video
    config.model.device = args.device
    config.model.xgait_model_path = args.xgait_model
    config.model.parsing_model_path = args.parsing_model
    config.model.yolo_model_path = args.yolo_model
    config.tracker.identification_threshold = args.id_threshold
    config.tracker.confidence_threshold = args.confidence
    config.video.display_window = not args.no_display
    config.debug_mode = args.debug
    config.verbose = args.verbose
    
    try:
        # Create and run identification app
        app = XGaitIdentificationApp(config)
        
        # Load existing gallery if specified
        if args.load_gallery:
            app.tracker.load_gallery(args.load_gallery)
            print(f"📂 Loaded person gallery from {args.load_gallery}")
        
        app.process_video()
        
    except KeyboardInterrupt:
        print("\n⏹️  Identification interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
