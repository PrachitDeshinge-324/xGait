"""
XGait-enhanced Person Tracker
Integrates person tracking with XGait-based identification
"""
import sys
import os
from pathlib import Path

# Add parent directory to path for utils and config imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Set, Optional
import math
import time

from models.identification_processor import create_identification_processor
from config import TrackerConfig, get_device_config
from utils.device_utils import DeviceManager

class XGaitTracker:
    """
    Enhanced person tracker using XGait-based identification
    Combines multi-object tracking with gait-based person identification
    """
    def __init__(self, 
                 yolo_model_path: str = "weights/yolo11m.pt",
                 device: str = "mps",
                 config: TrackerConfig = None,
                 xgait_model_path: str = "weights/Gait3D-XGait-120000.pt",
                 parsing_model_path: str = "weights/schp_resnet101.pth",
                 identification_threshold: float = 0.6):
        
        self.device = device
        self.config = config or TrackerConfig()
        self.identification_threshold = identification_threshold
        
        # Get device-specific configuration
        self.device_config = get_device_config(device)
        self.device_manager = DeviceManager(device, self.device_config["dtype"])
        
        # Load YOLO for person detection with device optimization
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_model.to(device)
        
        # Apply device-specific optimizations to YOLO
        if device == "cuda" and self.device_config["dtype"] == torch.float16:
            try:
                self.yolo_model.model.half()
            except Exception as e:
                print(f"Warning: Could not apply half precision to YOLO: {e}")
        
        # Initialize XGait identification processor
        self.identification_processor = create_identification_processor(
            device=device,
            xgait_model_path=xgait_model_path,
            parsing_model_path=parsing_model_path,
            identification_threshold=identification_threshold,
            parallel_processing=True
        )
        
        # Tracking state
        self.track_boxes: Dict[int, np.ndarray] = {}
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.config.track_history_length))
        self.track_crops: Dict[int, List[np.ndarray]] = defaultdict(list)
        self.missing_frames: Dict[int, int] = defaultdict(int)
        
        # ID management
        self.next_track_id = 1
        self.stable_tracks: Set[int] = set()
        
        # Performance tracking
        self.processing_times = {
            'detection': [],
            'tracking': [],
            'identification': [],
            'total': []
        }
        
        print(f"✅ XGaitTracker initialized")
        print(f"   Device: {device}")
        print(f"   Detection model: {yolo_model_path}")
        print(f"   XGait model: {xgait_model_path}")
        print(f"   Parsing model: {parsing_model_path}")
        print(f"   Identification threshold: {identification_threshold}")
    
    def detect_persons(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect persons in frame using YOLO
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (boxes, confidences) for detected persons
        """
        start_time = time.time()
        
        # Run YOLO detection
        results = self.yolo_model(frame, verbose=False)
        
        boxes = []
        confidences = []
        
        for result in results:
            for detection in result.boxes.data:
                x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
                
                # Filter for person class (class 0 in COCO)
                if int(cls) == 0 and conf >= self.config.confidence_threshold:
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(conf)
        
        detection_time = time.time() - start_time
        self.processing_times['detection'].append(detection_time)
        
        return np.array(boxes), np.array(confidences)
    
    def extract_person_crops(self, frame: np.ndarray, boxes: np.ndarray) -> List[np.ndarray]:
        """
        Extract person crops from frame using bounding boxes
        
        Args:
            frame: Input video frame
            boxes: Bounding boxes array (N, 4) in xyxy format
            
        Returns:
            List of person crop images
        """
        crops = []
        h, w = frame.shape[:2]
        
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            
            # Add padding and ensure valid coordinates
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            crop = frame[y1:y2, x1:x2]
            
            # Ensure minimum crop size
            if crop.shape[0] > 32 and crop.shape[1] > 16:
                crops.append(crop.copy())  # Make a copy to avoid memory issues
            else:
                # Create placeholder crop if too small
                crops.append(np.zeros((64, 32, 3), dtype=np.uint8))
        
        return crops
    
    def compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two bounding boxes"""
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])
        
        if x2_min <= x1_max or y2_min <= y1_max:
            return 0.0
        
        intersection = (x2_min - x1_max) * (y2_min - y1_max)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def match_detections_to_tracks(self, current_boxes: np.ndarray) -> List[int]:
        """
        Match current detections to existing tracks using IoU
        
        Args:
            current_boxes: Bounding boxes for current detections
            
        Returns:
            List of track IDs for each detection
        """
        assignments = []
        
        if len(self.track_boxes) == 0:
            # No existing tracks, assign new IDs
            for _ in range(len(current_boxes)):
                assignments.append(self.next_track_id)
                self.next_track_id += 1
            return assignments
        
        # Get existing track boxes and IDs
        track_ids = list(self.track_boxes.keys())
        track_boxes_list = [self.track_boxes[tid] for tid in track_ids]
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(current_boxes), len(track_boxes_list)))
        for i, current_box in enumerate(current_boxes):
            for j, track_box in enumerate(track_boxes_list):
                iou_matrix[i, j] = self.compute_iou(current_box, track_box)
        
        # Greedy assignment based on IoU
        used_tracks = set()
        iou_threshold = 0.3
        
        for i in range(len(current_boxes)):
            best_match_idx = -1
            best_iou = -1
            
            for j, track_id in enumerate(track_ids):
                if j not in used_tracks and iou_matrix[i, j] > iou_threshold:
                    if iou_matrix[i, j] > best_iou:
                        best_iou = iou_matrix[i, j]
                        best_match_idx = j
            
            if best_match_idx >= 0:
                # Matched to existing track
                assignments.append(track_ids[best_match_idx])
                used_tracks.add(best_match_idx)
            else:
                # Create new track
                assignments.append(self.next_track_id)
                self.next_track_id += 1
        
        return assignments
    
    def update_tracks(self, 
                     track_ids: List[int], 
                     boxes: np.ndarray, 
                     crops: List[np.ndarray],
                     frame_number: int) -> None:
        """
        Update track information
        
        Args:
            track_ids: Track IDs for each detection
            boxes: Bounding boxes for each detection
            crops: Person crops for each detection
            frame_number: Current frame number
        """
        active_tracks = set()
        
        for track_id, box, crop in zip(track_ids, boxes, crops):
            # Update track box and crop
            self.track_boxes[track_id] = box
            self.track_crops[track_id].append(crop)
            
            # Keep only recent crops (for memory efficiency)
            max_crops = 20
            if len(self.track_crops[track_id]) > max_crops:
                self.track_crops[track_id] = self.track_crops[track_id][-max_crops:]
            
            # Update history
            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)
            self.track_history[track_id].append((center_x, center_y, frame_number))
            
            # Reset missing counter and mark as active
            self.missing_frames[track_id] = 0
            active_tracks.add(track_id)
            
            # Mark as stable if enough history
            if len(self.track_history[track_id]) >= self.config.stable_track_threshold:
                self.stable_tracks.add(track_id)
        
        # Increment missing counter for inactive tracks
        for track_id in list(self.track_boxes.keys()):
            if track_id not in active_tracks:
                self.missing_frames[track_id] += 1
                
                # Remove tracks that have been missing too long
                if self.missing_frames[track_id] > self.config.max_missing_frames:
                    self.remove_track(track_id)
    
    def remove_track(self, track_id: int) -> None:
        """Remove a track from all tracking structures"""
        if track_id in self.track_boxes:
            del self.track_boxes[track_id]
        if track_id in self.track_crops:
            del self.track_crops[track_id]
        if track_id in self.missing_frames:
            del self.missing_frames[track_id]
        if track_id in self.track_history:
            del self.track_history[track_id]
        if track_id in self.stable_tracks:
            self.stable_tracks.remove(track_id)
    
    def track_and_identify_persons(self, frame: np.ndarray, frame_number: int) -> List[Tuple[int, np.ndarray, float, Optional[int], float]]:
        """
        Track persons and perform gait-based identification
        
        Args:
            frame: Input video frame
            frame_number: Current frame number
            
        Returns:
            List of (track_id, box, detection_conf, person_id, identification_conf) tuples
        """
        start_time = time.time()
        
        # Step 1: Detect persons
        boxes, confidences = self.detect_persons(frame)
        
        if len(boxes) == 0:
            # Cleanup old tracks and return empty
            self.identification_processor.cleanup_old_tracks(frame_number)
            return []
        
        # Step 2: Extract crops
        crops = self.extract_person_crops(frame, boxes)
        
        # Step 3: Track assignment
        track_start = time.time()
        track_ids = self.match_detections_to_tracks(boxes)
        self.update_tracks(track_ids, boxes, crops, frame_number)
        tracking_time = time.time() - track_start
        self.processing_times['tracking'].append(tracking_time)
        
        # Step 4: Identification for stable tracks
        id_start = time.time()
        track_data = {}
        for track_id in track_ids:
            if track_id in self.stable_tracks and track_id in self.track_crops:
                # Use recent crops for identification
                recent_crops = self.track_crops[track_id][-10:]  # Last 10 crops
                if recent_crops:
                    track_data[track_id] = recent_crops
        
        # Process identification in batch
        identification_results = {}
        if track_data:
            identification_results = self.identification_processor.process_multiple_tracks(
                track_data, frame_number
            )
        
        identification_time = time.time() - id_start
        self.processing_times['identification'].append(identification_time)
        
        # Step 5: Prepare results
        results = []
        for track_id, box, conf in zip(track_ids, boxes, confidences):
            person_id, id_confidence = identification_results.get(track_id, (None, 0.0))
            results.append((track_id, box, conf, person_id, id_confidence))
        
        # Cleanup old tracks periodically
        if frame_number % 50 == 0:
            self.identification_processor.cleanup_old_tracks(frame_number)
        
        total_time = time.time() - start_time
        self.processing_times['total'].append(total_time)
        
        return results
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        tracker_stats = {}
        for stage, times in self.processing_times.items():
            if times:
                tracker_stats[stage] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'count': len(times)
                }
            else:
                tracker_stats[stage] = {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0}
        
        # Get identification processor stats
        id_stats = self.identification_processor.get_performance_stats()
        identification_stats = self.identification_processor.get_identification_stats()
        
        return {
            'tracker': tracker_stats,
            'identification_processor': id_stats,
            'identification': identification_stats,
            'active_tracks': len(self.track_boxes),
            'stable_tracks': len(self.stable_tracks)
        }
    
    def get_device_info(self) -> Dict:
        """Get device information"""
        info = {
            'device': self.device,
            'dtype': str(self.device_config['dtype']),
            'autocast': self.device_config['autocast'],
            'compile': False  # Not using torch.compile in this implementation
        }
        
        if self.device == "cuda" and torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['cuda_version'] = torch.version.cuda
            info['memory_usage'] = {
                'allocated': torch.cuda.memory_allocated(0),
                'cached': torch.cuda.memory_reserved(0)
            }
        elif self.device == "mps" and torch.backends.mps.is_available():
            info['memory_usage'] = {
                'system_memory': 0  # MPS doesn't provide detailed memory info
            }
        
        return info
    
    def clear_memory_cache(self):
        """Clear memory cache"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()
    
    def synchronize_device(self):
        """Synchronize device"""
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()
    
    def get_statistics(self) -> Dict:
        """Get tracking and identification statistics"""
        return {
            'performance': self.get_performance_stats(),
            'device_info': self.get_device_info()
        }
    
    def save_gallery(self, filepath: str):
        """Save person gallery to file"""
        self.identification_processor.save_gallery(filepath)
    
    def load_gallery(self, filepath: str):
        """Load person gallery from file"""
        self.identification_processor.load_gallery(filepath)

def create_xgait_tracker(device: str = "mps", **kwargs) -> XGaitTracker:
    """Create an XGaitTracker instance"""
    return XGaitTracker(device=device, **kwargs)
