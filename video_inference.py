"""
XGait Video Inference Example
Demonstrates how to use the XGait inference pipeline with video input
"""
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
import argparse
import time
from typing import List, Dict, Tuple
from ultralytics import YOLO
from collections import defaultdict
import torch

from inference_pipeline import create_inference_pipeline
from utils.visualization import create_inference_visualizer


class SimplePersonTracker:
    """Simple person tracker using IoU-based tracking"""
    
    def __init__(self, iou_threshold: float = 0.3, max_missing_frames: int = 30):
        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames
        self.tracks = {}
        self.next_track_id = 1
        self.missing_frames = defaultdict(int)
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1, y1, x2, y2 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        xi1 = max(x1, x1_2)
        yi1 = max(y1, y1_2)
        xi2 = min(x2, x2_2)
        yi2 = min(y2, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union area
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def update(self, detections: List[List[float]]) -> Dict[int, List[float]]:
        """Update tracks with new detections"""
        if not detections:
            # Increment missing frames for all tracks
            for track_id in list(self.tracks.keys()):
                self.missing_frames[track_id] += 1
                if self.missing_frames[track_id] > self.max_missing_frames:
                    del self.tracks[track_id]
                    del self.missing_frames[track_id]
            return {}
        
        # Match detections to existing tracks
        matched_tracks = {}
        unmatched_detections = detections.copy()
        
        for track_id, track_box in self.tracks.items():
            best_iou = 0.0
            best_detection = None
            best_idx = -1
            
            for idx, detection in enumerate(unmatched_detections):
                iou = self.calculate_iou(track_box, detection)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_detection = detection
                    best_idx = idx
            
            if best_detection is not None:
                # Update track
                self.tracks[track_id] = best_detection
                matched_tracks[track_id] = best_detection
                self.missing_frames[track_id] = 0
                unmatched_detections.pop(best_idx)
            else:
                # Track not matched
                self.missing_frames[track_id] += 1
                if self.missing_frames[track_id] <= self.max_missing_frames:
                    matched_tracks[track_id] = track_box
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            track_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[track_id] = detection
            matched_tracks[track_id] = detection
            self.missing_frames[track_id] = 0
        
        # Remove tracks that have been missing too long
        tracks_to_remove = []
        for track_id in self.tracks:
            if self.missing_frames[track_id] > self.max_missing_frames:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            del self.missing_frames[track_id]
        
        return matched_tracks


class VideoInferenceRunner:
    """
    Complete video inference runner with person detection, tracking, and identification
    """
    
    def __init__(self,
                 video_path: str,
                 device: str = "mps",
                 yolo_model_path: str = "weights/yolo11m.pt",
                 xgait_model_path: str = "weights/Gait3D-XGait-120000.pt",
                 parsing_model_path: str = "weights/schp_resnet101.pth",
                 confidence_threshold: float = 0.5,
                 identification_threshold: float = 0.6,
                 display_output: bool = True,
                 save_results: bool = False,
                 output_path: str = "output_video.mp4"):
        
        self.video_path = video_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.identification_threshold = identification_threshold
        self.display_output = display_output
        self.save_results = save_results
        self.output_path = output_path
        
        # Initialize models
        print("🚀 Initializing models...")
        
        # YOLO for person detection
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_model.to(device)
        
        # Person tracker
        self.tracker = SimplePersonTracker()
        
        # XGait inference pipeline
        self.inference_pipeline = create_inference_pipeline(
            device=device,
            xgait_model_path=xgait_model_path,
            parsing_model_path=parsing_model_path,
            identification_threshold=identification_threshold,
            parallel_processing=True
        )
        
        # Visualization
        self.visualizer = create_inference_visualizer()
        
        print("✅ All models initialized successfully!")
    
    def extract_person_crop(self, frame: np.ndarray, box: List[float], 
                           margin: float = 0.1) -> np.ndarray:
        """Extract person crop from frame with margin"""
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        
        # Add margin
        box_h = y2 - y1
        box_w = x2 - x1
        margin_h = int(box_h * margin)
        margin_w = int(box_w * margin)
        
        # Expand box with margin
        x1 = max(0, x1 - margin_w)
        y1 = max(0, y1 - margin_h)
        x2 = min(w, x2 + margin_w)
        y2 = min(h, y2 + margin_h)
        
        # Extract crop
        crop = frame[y1:y2, x1:x2]
        
        # Resize to standard size for better processing
        if crop.shape[0] > 0 and crop.shape[1] > 0:
            crop = cv2.resize(crop, (64, 128))
        
        return crop
    
    def run_inference(self):
        """Run inference on the video"""
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📺 Video properties: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")
        
        # Initialize video writer if saving results
        writer = None
        if self.save_results:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_width, frame_height))
        
        # Processing statistics
        frame_count = 0
        processing_times = []
        detection_times = []
        identification_times = []
        
        # Person gallery for visualization
        person_colors = {}
        color_palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (255, 192, 203), (0, 128, 0), (128, 128, 0), (0, 0, 128)
        ]
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_start_time = time.time()
                
                # Person detection
                detect_start = time.time()
                results = self.yolo_model(frame, verbose=False)
                detections = []
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Filter for person class (class 0 in COCO)
                            if int(box.cls[0]) == 0 and float(box.conf[0]) > self.confidence_threshold:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                detections.append([x1, y1, x2, y2])
                
                detect_time = time.time() - detect_start
                detection_times.append(detect_time)
                
                # Track persons
                track_boxes = self.tracker.update(detections)
                
                # Extract crops and run identification
                identification_start = time.time()
                track_crops = {}
                for track_id, box in track_boxes.items():
                    crop = self.extract_person_crop(frame, box)
                    if crop.shape[0] > 0 and crop.shape[1] > 0:
                        track_crops[track_id] = [crop]
                
                # Run identification pipeline
                identification_results = self.inference_pipeline.process_frame_tracks(
                    frame, track_crops, frame_count
                )
                
                identification_time = time.time() - identification_start
                identification_times.append(identification_time)
                
                # Visualization
                vis_frame = frame.copy()
                
                for track_id, box in track_boxes.items():
                    person_id, confidence = identification_results.get(track_id, (None, 0.0))
                    
                    # Assign color to person
                    if person_id is not None:
                        if person_id not in person_colors:
                            color_idx = len(person_colors) % len(color_palette)
                            person_colors[person_id] = color_palette[color_idx]
                        color = person_colors[person_id]
                    else:
                        color = (128, 128, 128)  # Gray for unidentified
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw labels
                    if person_id is not None:
                        label = f"Person {person_id} ({confidence:.2f})"
                    else:
                        label = f"Track {track_id}"
                    
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(vis_frame, (x1, y1 - 25), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(vis_frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Add frame info
                frame_time = time.time() - frame_start_time
                processing_times.append(frame_time)
                
                info_text = [
                    f"Frame: {frame_count + 1}/{total_frames}",
                    f"FPS: {1.0 / frame_time:.1f}",
                    f"Tracks: {len(track_boxes)}",
                    f"Gallery: {len(self.inference_pipeline.xgait_model.person_gallery)}"
                ]
                
                for i, text in enumerate(info_text):
                    cv2.putText(vis_frame, text, (10, 30 + i * 25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(vis_frame, text, (10, 30 + i * 25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                
                # Display frame
                if self.display_output:
                    cv2.imshow('XGait Person Identification', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Save frame
                if writer is not None:
                    writer.write(vis_frame)
                
                frame_count += 1
                
                # Print progress
                if frame_count % 30 == 0:
                    avg_fps = 1.0 / np.mean(processing_times[-30:]) if processing_times else 0
                    print(f"📊 Processed {frame_count} frames, Avg FPS: {avg_fps:.1f}")
                
                # Cleanup old tracks periodically
                if frame_count % 100 == 0:
                    self.inference_pipeline.cleanup_old_tracks(frame_count)
        
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if writer is not None:
                writer.release()
            if self.display_output:
                cv2.destroyAllWindows()
        
        # Print final statistics
        self._print_final_statistics(frame_count, processing_times, 
                                   detection_times, identification_times)
    
    def _print_final_statistics(self, frame_count: int, processing_times: List[float],
                              detection_times: List[float], identification_times: List[float]):
        """Print final processing statistics"""
        print("\n📈 Final Statistics:")
        print(f"   Total frames processed: {frame_count}")
        
        if processing_times:
            avg_fps = 1.0 / np.mean(processing_times)
            print(f"   Average FPS: {avg_fps:.2f}")
            print(f"   Total processing time: {sum(processing_times):.2f}s")
        
        if detection_times:
            print(f"   Average detection time: {np.mean(detection_times)*1000:.2f}ms")
        
        if identification_times:
            print(f"   Average identification time: {np.mean(identification_times)*1000:.2f}ms")
        
        # Pipeline statistics
        pipeline_stats = self.inference_pipeline.get_performance_stats()
        print("\n🔧 Pipeline Component Times:")
        for component, metrics in pipeline_stats.items():
            if component != 'gallery' and 'avg_time' in metrics:
                print(f"   {component}: {metrics['avg_time']*1000:.2f}ms avg")
        
        # Gallery statistics
        if 'gallery' in pipeline_stats:
            gallery = pipeline_stats['gallery']
            print(f"\n👥 Person Gallery: {gallery['total_identities']} identities")
        
        if self.save_results:
            print(f"\n💾 Results saved to: {self.output_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="XGait Video Inference")
    parser.add_argument("--video", type=str, default="input/3c.mp4", 
                       help="Input video path")
    parser.add_argument("--device", type=str, default="mps", 
                       choices=["cpu", "cuda", "mps"], help="Device to use")
    parser.add_argument("--yolo-model", type=str, default="weights/yolo11m.pt", 
                       help="YOLO model path")
    parser.add_argument("--xgait-model", type=str, default="weights/Gait3D-XGait-120000.pt", 
                       help="XGait model path")
    parser.add_argument("--parsing-model", type=str, default="weights/schp_resnet101.pth", 
                       help="Human parsing model path")
    parser.add_argument("--confidence", type=float, default=0.5, 
                       help="Detection confidence threshold")
    parser.add_argument("--id-threshold", type=float, default=0.6, 
                       help="Identification threshold")
    parser.add_argument("--no-display", action="store_true", 
                       help="Disable video display")
    parser.add_argument("--save", action="store_true", 
                       help="Save output video")
    parser.add_argument("--output", type=str, default="output_inference.mp4", 
                       help="Output video path")
    
    args = parser.parse_args()
    
    # Run inference
    runner = VideoInferenceRunner(
        video_path=args.video,
        device=args.device,
        yolo_model_path=args.yolo_model,
        xgait_model_path=args.xgait_model,
        parsing_model_path=args.parsing_model,
        confidence_threshold=args.confidence,
        identification_threshold=args.id_threshold,
        display_output=not args.no_display,
        save_results=args.save,
        output_path=args.output
    )
    
    runner.run_inference()


if __name__ == "__main__":
    main()
