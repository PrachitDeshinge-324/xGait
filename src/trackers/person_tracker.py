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
                 device: str = None,
                 config: TrackerConfig = None):
        
        from config import get_global_device
        self.device = device if device is not None else get_global_device()
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
        
        # Track memory gallery for re-identification of disappeared tracks
        self.track_gallery = {}  # Store features of disappeared tracks
        self.gallery_last_seen = {}  # Frame when each gallery track was last seen
        self.gallery_expiration = 1000  # How long to keep disappeared tracks in gallery
        
        # Enhanced feature templates
        self.track_feature_templates = defaultdict(list)  # Multiple templates per track
        self.max_templates = 3  # Maximum number of feature templates per track
        
        # ID consistency mechanisms
        self.id_switch_cooldown = {}  # Prevent rapid ID switches
        self.cooldown_period = 30  # Frames to wait before allowing an ID switch
    
        print(f"✅ PersonTracker initialized with enhanced identity consistency")
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
                                  current_boxes: np.ndarray,
                                  frame_number: int) -> List[int]:
        """
        Enhanced matching with memory gallery and consistent identity assignment
        """
        assignments = []
        
        if len(self.track_features) == 0 and len(self.track_gallery) == 0:
            # No existing tracks or gallery, assign new IDs
            for _ in range(len(current_features)):
                assignments.append(self.next_id)
                self.next_id += 1
            return assignments
        
        # Get features of existing active tracks
        active_track_ids = list(self.track_features.keys())
        active_track_features = [self.track_features[tid] for tid in active_track_ids]
        
        # Get features from track gallery (disappeared tracks)
        gallery_track_ids = list(self.track_gallery.keys())
        gallery_track_features = [self.track_gallery[tid] for tid in gallery_track_ids]
        
        # Combine active and gallery tracks
        all_track_ids = active_track_ids + gallery_track_ids
        
        if active_track_features and current_features.numel() > 0:
            # Stack active track features
            active_features_tensor = torch.stack(active_track_features) if active_track_features else torch.empty(0, current_features.shape[1], device=self.device)
            
            # Calculate similarity with active tracks
            active_similarity_matrix = self.reid_model.compute_similarity(
                current_features, active_features_tensor
            )
            active_similarity_np = active_similarity_matrix.cpu().numpy()
            
            # Calculate similarity with gallery tracks
            gallery_similarity_np = np.zeros((len(current_features), len(gallery_track_ids)))
            if gallery_track_features and current_features.numel() > 0:
                gallery_features_tensor = torch.stack(gallery_track_features)
                gallery_similarity_matrix = self.reid_model.compute_similarity(
                    current_features, gallery_features_tensor
                )
                gallery_similarity_np = gallery_similarity_matrix.cpu().numpy()
            
            # Combine similarities (active tracks first, then gallery)
            similarity_np = np.hstack([active_similarity_np, gallery_similarity_np]) if gallery_similarity_np.size > 0 else active_similarity_np
            
            # Calculate spatial distances for additional constraint (only for active tracks)
            spatial_weights = np.ones_like(similarity_np)
            
            # Apply spatial weighting only to active tracks
            for i, current_box in enumerate(current_boxes):
                current_center = np.array([(current_box[0] + current_box[2]) / 2, 
                                         (current_box[1] + current_box[3]) / 2])
                
                # For active tracks - apply spatial weighting
                for j, track_id in enumerate(active_track_ids):
                    if track_id in self.track_history and len(self.track_history[track_id]) > 0:
                        last_pos = np.array(self.track_history[track_id][-1][:2])
                        spatial_distance = np.linalg.norm(current_center - last_pos)
                        
                        # Apply adaptive spatial penalty based on track stability
                        if track_id in self.stable_tracks:
                            max_distance = 250.0  # More tolerance for stable tracks
                        else:
                            max_distance = 200.0
                            
                        if spatial_distance > max_distance:
                            spatial_weights[i, j] = 0.1  # Heavy penalty for distant matches
                        else:
                            # Slight penalty based on distance
                            spatial_weights[i, j] = 1.0 - (spatial_distance / max_distance) * 0.3
                
                # For gallery tracks - no spatial weighting, but apply time penalty
                for j, track_id in enumerate(gallery_track_ids):
                    frames_gone = frame_number - self.gallery_last_seen[track_id]
                    # Gradually reduce score for tracks that have been gone longer
                    time_factor = max(0.4, 1.0 - (frames_gone / 200.0))  # Minimum factor of 0.4
                    spatial_weights[i, j + len(active_track_ids)] = time_factor
            
            # Apply weights to similarity scores
            weighted_similarity = similarity_np * spatial_weights
            
            # Enhanced assignment with ID consistency
            used_tracks = set()
            
            # First pass: use Hungarian algorithm for globally optimal assignment
            from scipy.optimize import linear_sum_assignment
            
            # Convert similarities to costs (1 - similarity)
            cost_matrix = 1.0 - weighted_similarity
            
            # Only consider matches above threshold
            mask = weighted_similarity < self.config.similarity_threshold
            cost_matrix[mask] = 1000.0  # High cost for below-threshold matches
            
            # Find optimal assignment
            detection_indices, track_indices = linear_sum_assignment(cost_matrix)
            
            # Apply assignments if they are valid matches
            for det_idx, track_idx in zip(detection_indices, track_indices):
                if track_idx < len(active_track_ids) and weighted_similarity[det_idx, track_idx] >= self.config.similarity_threshold:
                    # Match to active track
                    assignments.append(active_track_ids[track_idx])
                    used_tracks.add(track_idx)
                elif track_idx >= len(active_track_ids) and weighted_similarity[det_idx, track_idx] >= self.config.similarity_threshold * 1.2:
                    # Higher threshold for gallery matches (20% higher)
                    gallery_idx = track_idx - len(active_track_ids)
                    track_id = gallery_track_ids[gallery_idx]
                    
                    # Check cooldown period for gallery re-identification
                    if track_id in self.id_switch_cooldown and \
                       frame_number - self.id_switch_cooldown[track_id] < self.cooldown_period:
                        # Still in cooldown, create new ID
                        assignments.append(self.next_id)
                        self.next_id += 1
                    else:
                        # Re-identify from gallery
                        assignments.append(track_id)
                        # Remove from gallery as it's active again
                        del self.track_gallery[track_id]
                        del self.gallery_last_seen[track_id]
                else:
                    # No valid match, create new ID
                    assignments.append(self.next_id)
                    self.next_id += 1
            
            # Handle unmatched detections
            for i in range(len(current_features)):
                if i not in detection_indices:
                    # Try spatial-only matching for remaining detections
                    current_center = np.array([(current_boxes[i][0] + current_boxes[i][2]) / 2, 
                                             (current_boxes[i][1] + current_boxes[i][3]) / 2])
                    
                    min_distance = float('inf')
                    best_track_idx = -1
                    
                    for j, track_id in enumerate(active_track_ids):
                        if j not in used_tracks and track_id in self.track_history and len(self.track_history[track_id]) > 0:
                            last_pos = np.array(self.track_history[track_id][-1][:2])
                            distance = np.linalg.norm(current_center - last_pos)
                            
                            if distance < min_distance and distance < 80.0:  # Very close threshold
                                min_distance = distance
                                best_track_idx = j
                    
                    if best_track_idx >= 0 and active_similarity_np[i, best_track_idx] > 0.25:
                        # Match to spatially close track if similarity is reasonable
                        assignments.append(active_track_ids[best_track_idx])
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
        Enhanced track update with improved feature management
        """
        active_tracks = set()
        
        for track_id, feature, box, crop in zip(track_ids, features, boxes, crops):
            # Update track feature with adaptive exponential moving average
            if track_id in self.track_features:
                # Adaptive update rate based on track stability
                if track_id in self.stable_tracks:
                    alpha = 0.85  # More conservative update for stable tracks
                else:
                    alpha = 0.65  # Faster adaptation for new tracks
            
                # Update main feature representation    
                self.track_features[track_id] = (
                    alpha * self.track_features[track_id] + (1 - alpha) * feature
                )
                
                # Manage multiple feature templates for different appearances
                should_add_template = True
                
                # Check if this feature is significantly different from existing templates
                for template in self.track_feature_templates[track_id]:
                    similarity = self.reid_model.compute_similarity(
                        feature.unsqueeze(0), 
                        template.unsqueeze(0)
                    ).item()
                    
                    if similarity > 0.85:  # Very similar to existing template
                        should_add_template = False
                        break
            
                # Add new template if different enough
                if should_add_template:
                    self.track_feature_templates[track_id].append(feature)
                    
                    # Maintain maximum number of templates
                    while len(self.track_feature_templates[track_id]) > self.max_templates:
                        self.track_feature_templates[track_id].pop(0)  # Remove oldest
            else:
                # New track
                self.track_features[track_id] = feature
                self.track_feature_templates[track_id].append(feature)  # Add first template
        
            # Update position history
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
    
        # Process inactive tracks
        for track_id in list(self.track_features.keys()):
            if track_id not in active_tracks:
                self.missing_frames[track_id] += 1
                
                # Move to gallery instead of immediately removing
                if self.missing_frames[track_id] > self.config.max_missing_frames:
                    # Store in gallery before removing
                    if track_id in self.track_features:
                        self.track_gallery[track_id] = self.track_features[track_id].clone()
                        self.gallery_last_seen[track_id] = frame_number
                    
                    # Remove from active tracks
                    self.remove_track(track_id)
    
        # Clean expired gallery entries
        for track_id in list(self.gallery_last_seen.keys()):
            if frame_number - self.gallery_last_seen[track_id] > self.gallery_expiration:
                if track_id in self.track_gallery:
                    del self.track_gallery[track_id]
                del self.gallery_last_seen[track_id]
    
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
        Main tracking function with enhanced identity consistency
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
        
        # Match detections to tracks with enhanced consistency
        track_ids = self.match_detections_to_tracks(features, boxes, frame_number)
        
        # Update track state with improved feature management
        self.update_tracks(track_ids, features, boxes, crops, frame_number)
        
        # Apply sequence-based ID correction
        corrected_track_ids = self._verify_and_correct_id_switches(track_ids, boxes, features, frame_number)
        
        # Return tracking results
        results_list = []
        for track_id, box, conf in zip(corrected_track_ids, boxes, confidences):
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
    
    def _verify_and_correct_id_switches(self, track_ids: List[int], boxes: np.ndarray, 
                              features: torch.Tensor, frame_number: int) -> List[int]:
        """
        Verify ID consistency with sequence history
        """
        corrected_ids = track_ids.copy()
        
        # First pass: detect suspicious motion
        suspicious_tracks = []
        for i, track_id in enumerate(track_ids):
            if track_id in self.track_history and len(self.track_history[track_id]) >= 2:
                is_switch, distance = self.detect_id_switches(track_id)
                if is_switch:
                    suspicious_tracks.append((i, track_id, boxes[i], features[i]))
        
        # Second pass: analyze sequence history for each suspicious track
        for idx, track_id, box, feature in suspicious_tracks:
            # Get sequence history for this track
            sequence_length = min(10, len(self.track_feature_templates[track_id]))
            if sequence_length < 3:
                continue  # Not enough history
                
            # Find most consistent ID using template history
            best_match_id = None
            best_consistency_score = 0
            
            for other_id in self.track_features.keys():
                if other_id == track_id or other_id not in self.track_feature_templates:
                    continue
                    
                # Skip tracks that are too far away
                if other_id in self.track_history and len(self.track_history[other_id]) > 0:
                    other_pos = np.array(self.track_history[other_id][-1][:2])
                    curr_pos = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                    distance = np.linalg.norm(curr_pos - other_pos)
                    if distance > 150:
                        continue
            
                # Calculate sequence consistency score
                match_count = 0
                avg_similarity = 0
                
                for template in self.track_feature_templates[other_id][-sequence_length:]:
                    sim = self.reid_model.compute_similarity(
                        feature.unsqueeze(0), template.unsqueeze(0)
                    ).item()
                    
                    if sim > 0.5:
                        match_count += 1
                        avg_similarity += sim
            
                if match_count > 0:
                    avg_similarity /= match_count
                    consistency_score = avg_similarity * (match_count / sequence_length)
                    
                    if consistency_score > best_consistency_score and consistency_score > 0.4:
                        best_consistency_score = consistency_score
                        best_match_id = other_id
            
            # Apply correction if we found a better match for this suspicious track
            if best_match_id is not None and best_consistency_score > 0.5:
                print(f"Sequence-based ID correction: {track_id} -> {best_match_id} (score: {best_consistency_score:.3f})")
                corrected_ids[idx] = best_match_id
                self.id_switch_cooldown[track_id] = frame_number
    
        return corrected_ids

    def extract_features_with_templates(self, detection_features: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Extract the most similar features from templates for each track
        
        Args:
            detection_features: Features of current detections
            
        Returns:
            Dictionary of {track_id: best_feature}
        """
        best_features = {}
        
        for track_id in self.track_features.keys():
            # Get all templates for this track
            templates = self.track_feature_templates[track_id]
            
            if not templates:
                best_features[track_id] = self.track_features[track_id]
                continue
            
            # Stack all templates
            templates_tensor = torch.stack(templates)
            
            # Find most similar template to current detection
            similarity = self.reid_model.compute_similarity(detection_features, templates_tensor)
            
            # Use the highest similarity template or default track feature
            max_vals, _ = torch.max(similarity, dim=1)
            if max_vals.numel() > 0 and torch.max(max_vals) > 0.7:
                best_idx = torch.argmax(max_vals)
                best_features[track_id] = templates[best_idx % len(templates)]
            else:
                best_features[track_id] = self.track_features[track_id]
        
        return best_features
