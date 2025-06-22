"""
Custom Person Tracker using TransReID for appearance-based re-identification
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
from typing import List, Tuple, Dict, Set
import math

from models.reid_model import create_reid_model
from config import TrackerConfig, get_device_config
from utils.device_utils import DeviceManager

class PersonTracker:
    """
    Custom person tracker using appearance-based re-identification with device optimization
    """
    def __init__(self, 
                 yolo_model_path: str = "weights/yolo11m.pt", 
                 device: str = "mps",
                 config: TrackerConfig = None):
        
        self.device = device
        self.config = config or TrackerConfig()
        
        # Get device-specific configuration
        self.device_config = get_device_config(device)
        self.device_manager = DeviceManager(device, self.device_config["dtype"])
        
        # Load YOLO for person detection with device optimization
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_model.to(device)
        
        # Apply device-specific optimizations to YOLO
        if device == "cuda" and self.device_config["dtype"] == torch.float16:
            # Use half precision for CUDA
            try:
                self.yolo_model.model.half()
            except Exception as e:
                print(f"Warning: Could not apply half precision to YOLO: {e}")
        
        # Load ReID model for appearance matching
        self.reid_model = create_reid_model(device)
        
        # Tracking state
        self.track_features: Dict[int, torch.Tensor] = {}
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.config.track_history_length))
        self.track_crops: Dict[int, np.ndarray] = {}
        self.missing_frames: Dict[int, int] = defaultdict(int)
        
        # ID management
        self.next_id = 1
        self.stable_tracks: Set[int] = set()
        
        print(f"✅ PersonTracker initialized")
        print(f"   Device: {device}")
        print(f"   Dtype: {self.device_config['dtype']}")
        print(f"   Autocast: {self.device_config['autocast']}")
        print(f"   Similarity threshold: {self.config.similarity_threshold}")
        print(f"   Max missing frames: {self.config.max_missing_frames}")
    
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
                crops.append(crop)
            else:
                # Create placeholder crop if too small
                crops.append(np.zeros((64, 32, 3), dtype=np.uint8))
        
        return crops
    
    def match_detections_to_tracks(self, 
                                  current_features: torch.Tensor, 
                                  current_boxes: np.ndarray) -> List[int]:
        """
        Match current detections to existing tracks using appearance similarity
        
        Args:
            current_features: Feature vectors for current detections
            current_boxes: Bounding boxes for current detections
            
        Returns:
            List of track IDs for each detection
        """
        assignments = []
        
        if len(self.track_features) == 0:
            # No existing tracks, assign new IDs
            for _ in range(len(current_features)):
                assignments.append(self.next_id)
                self.next_id += 1
            return assignments
        
        # Get features of existing tracks
        track_ids = list(self.track_features.keys())
        track_features_list = [self.track_features[tid] for tid in track_ids]
        
        if len(track_features_list) > 0 and current_features.numel() > 0:
            # Stack track features
            track_features_tensor = torch.stack(track_features_list)
            
            # Compute similarity matrix
            similarity_matrix = self.reid_model.compute_similarity(
                current_features, track_features_tensor
            )
            
            # Greedy assignment (Hungarian algorithm would be better but this is simpler)
            similarity_np = similarity_matrix.cpu().numpy()
            used_tracks = set()
            
            for i in range(len(current_features)):
                best_match_idx = -1
                best_similarity = -1
                
                for j, track_id in enumerate(track_ids):
                    if j not in used_tracks and similarity_np[i, j] > self.config.similarity_threshold:
                        if similarity_np[i, j] > best_similarity:
                            best_similarity = similarity_np[i, j]
                            best_match_idx = j
                
                if best_match_idx >= 0:
                    # Matched to existing track
                    assignments.append(track_ids[best_match_idx])
                    used_tracks.add(best_match_idx)
                else:
                    # Create new track
                    assignments.append(self.next_id)
                    self.next_id += 1
        else:
            # Fallback: create new IDs
            for _ in range(len(current_features)):
                assignments.append(self.next_id)
                self.next_id += 1
        
        return assignments
    
    def update_tracks(self, 
                     track_ids: List[int], 
                     features: torch.Tensor, 
                     boxes: np.ndarray, 
                     crops: List[np.ndarray],
                     frame_number: int) -> None:
        """
        Update track features and history
        
        Args:
            track_ids: Track IDs for each detection
            features: Feature vectors for each detection
            boxes: Bounding boxes for each detection
            crops: Person crops for each detection
            frame_number: Current frame number
        """
        active_tracks = set()
        
        for track_id, feature, box, crop in zip(track_ids, features, boxes, crops):
            # Update track feature (exponential moving average)
            alpha = 0.7
            if track_id in self.track_features:
                self.track_features[track_id] = (
                    alpha * feature + (1 - alpha) * self.track_features[track_id]
                )
            else:
                self.track_features[track_id] = feature
            
            # Update history
            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)
            self.track_history[track_id].append((center_x, center_y, frame_number))
            
            # Store crop
            self.track_crops[track_id] = crop
            
            # Reset missing counter and mark as active
            self.missing_frames[track_id] = 0
            active_tracks.add(track_id)
            
            # Mark as stable if enough history
            if len(self.track_history[track_id]) >= self.config.stable_track_threshold:
                self.stable_tracks.add(track_id)
        
        # Increment missing counter for inactive tracks
        for track_id in list(self.track_features.keys()):
            if track_id not in active_tracks:
                self.missing_frames[track_id] += 1
                
                # Remove tracks that have been missing too long
                if self.missing_frames[track_id] > self.config.max_missing_frames:
                    self.remove_track(track_id)
    
    def remove_track(self, track_id: int) -> None:
        """Remove a track from all tracking structures"""
        if track_id in self.track_features:
            del self.track_features[track_id]
        if track_id in self.missing_frames:
            del self.missing_frames[track_id]
        if track_id in self.track_crops:
            del self.track_crops[track_id]
        if track_id in self.track_history:
            del self.track_history[track_id]
        if track_id in self.stable_tracks:
            self.stable_tracks.remove(track_id)
    
    def detect_id_switches(self, track_id: int) -> Tuple[bool, float]:
        """
        Detect potential ID switches based on sudden position jumps
        
        Args:
            track_id: Track ID to check
            
        Returns:
            (is_switch, jump_distance)
        """
        if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
            return False, 0.0
        
        history = list(self.track_history[track_id])
        if len(history) < 2:
            return False, 0.0
        
        # Get last two positions
        prev_pos = history[-2][:2]
        curr_pos = history[-1][:2]
        
        # Calculate distance
        distance = math.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
        
        # Check if distance exceeds threshold
        is_switch = distance > self.config.id_switch_distance_threshold
        
        return is_switch, distance
    
    def track_persons(self, frame: np.ndarray, frame_number: int = 0) -> List[Tuple[int, np.ndarray, float]]:
        """
        Main tracking function with device optimization
        
        Args:
            frame: Input video frame
            frame_number: Current frame number
            
        Returns:
            List of (track_id, bounding_box, confidence) tuples
        """
        # YOLO detection with device-specific optimization
        with self.device_manager.autocast_context():
            results = self.yolo_model(
                frame,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                classes=[0],  # Person class only
                verbose=False,
                half=self.device_config["dtype"] == torch.float16 and self.device == "cuda"
            )
        
        if results[0].boxes is None:
            return []
        
        # Extract detection data
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        if len(boxes) == 0:
            return []
        
        # Extract person crops and features
        crops = self.extract_person_crops(frame, boxes)
        features = self.reid_model.extract_features(crops)
        
        if features.numel() == 0:
            return []
        
        # Match detections to tracks
        track_ids = self.match_detections_to_tracks(features, boxes)
        
        # Update track state
        self.update_tracks(track_ids, features, boxes, crops, frame_number)
        
        # Return tracking results
        results_list = []
        for track_id, box, conf in zip(track_ids, boxes, confidences):
            results_list.append((track_id, box, conf))
        
        return results_list
    
    def get_device_info(self) -> Dict:
        """Get device information and memory usage"""
        info = {
            'device': self.device,
            'dtype': str(self.device_config['dtype']),
            'autocast': self.device_config['autocast'],
            'compile': self.device_config['compile'],
            'memory_usage': self.device_manager.get_memory_info()
        }
        
        if self.device == "cuda":
            info['gpu_name'] = torch.cuda.get_device_name()
            info['cuda_version'] = torch.version.cuda
        
        return info
    
    def clear_memory_cache(self):
        """Clear device memory cache"""
        self.device_manager.clear_cache()
        
    def synchronize_device(self):
        """Synchronize device operations"""
        self.device_manager.synchronize()

    def get_statistics(self) -> Dict:
        """Get tracking statistics including device info"""
        stats = {
            "active_tracks": len(self.track_features),
            "stable_tracks": len(self.stable_tracks),
            "max_track_id": self.next_id - 1,
            "total_tracks_created": self.next_id - 1,
            "device_info": self.get_device_info()
        }
        return stats
