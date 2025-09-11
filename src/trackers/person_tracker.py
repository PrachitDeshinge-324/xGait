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
from typing import List, Tuple, Dict, Set, Optional
import math

from models.reid_model import create_reid_model
from config import TrackerConfig, get_device_config
from utils.device_utils import DeviceManager

class PersonTracker:
    """
    Custom person tracker using appearance-based re-identification with device optimization
    """
    def __init__(self, 
                 yolo_model_path: str = "weights/yolo11s-seg.pt",  # Default to segmentation model 
                 device: str = None,
                 config: TrackerConfig = None):
        
        from config import get_global_device
        self.device = device if device is not None else get_global_device()
        # Enhanced configuration for identity-aware tracking
        self.config = config or TrackerConfig()
        self.similarity_threshold = 0.65  # Lowered for better track association
        self.high_confidence_threshold = 0.80  # High confidence matches
        self.spatial_distance_threshold = 150  # Increased for better re-identification
        self.temporal_window = 60  # Frames to maintain track memory
        self.identity_consistency_weight = 0.3  # Weight for identity consistency in matching
        
        # Get device-specific configuration
        self.device_config = get_device_config(device)
        self.device_manager = DeviceManager(device, self.device_config["dtype"])
        
        # Load YOLO segmentation model for person detection and segmentation
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
        self.track_masks: Dict[int, np.ndarray] = {}  # Store segmentation masks for tracks
        self.missing_frames: Dict[int, int] = defaultdict(int)
        
        # ID management
        self.next_id = 1
        self.stable_tracks: Set[int] = set()
        
        # Track memory gallery for re-identification of disappeared tracks
        self.track_gallery = {}  # Store features of disappeared tracks
        self.gallery_last_seen = {}  # Frame when each gallery track was last seen
        self.gallery_expiration = 1000  # How long to keep disappeared tracks in gallery
        
        # Enhanced feature templates with identity consistency
        self.track_feature_templates = defaultdict(list)  # Multiple templates per track
        self.max_templates = 5  # Increased for better representation
        
        # Identity consistency tracking
        self.track_identity_votes = defaultdict(lambda: defaultdict(int))  # track_id -> {identity: vote_count}
        self.track_identity_confidence = defaultdict(float)  # track_id -> confidence_in_current_identity
        self.identity_spatial_memory = {}  # identity -> recent_positions for spatial consistency
        
        # Identity-aware tracking attributes for LOGIC-013 fix
        self.track_stability_score = {}  # track_id -> stability score (0-1)
        self.recent_assignments = {}  # track_id -> last_frame_assigned
        
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
        Extract person crops from frame using bounding boxes with improved edge handling
        
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
            
            # Calculate original box dimensions for aspect ratio preservation
            orig_width = x2 - x1
            orig_height = y2 - y1
            
            # Target aspect ratio for person crops (width:height = 1:2 approximately)
            target_aspect_ratio = 0.5  # width/height
            
            # Calculate current aspect ratio
            current_aspect_ratio = orig_width / max(orig_height, 1)
            
            # Adjust box to better aspect ratio if needed
            if current_aspect_ratio > target_aspect_ratio * 1.3:  # Too wide
                # Make it taller
                new_height = int(orig_width / target_aspect_ratio)
                height_increase = new_height - orig_height
                y1 = max(0, y1 - height_increase // 2)
                y2 = min(h, y2 + height_increase // 2)
            elif current_aspect_ratio < target_aspect_ratio * 0.7:  # Too tall/narrow
                # Make it wider
                new_width = int(orig_height * target_aspect_ratio)
                width_increase = new_width - orig_width
                x1 = max(0, x1 - width_increase // 2)
                x2 = min(w, x2 + width_increase // 2)
            
            # Recalculate dimensions after aspect ratio adjustment
            adjusted_width = x2 - x1
            adjusted_height = y2 - y1
            
            # Expand bounding box with adaptive padding based on adjusted box size
            padding_x = max(20, int(adjusted_width * 0.15))  # 15% padding or minimum 20px
            padding_y = max(25, int(adjusted_height * 0.08))  # 8% padding or minimum 25px
            
            # Expand the box
            exp_x1 = x1 - padding_x
            exp_y1 = y1 - padding_y
            exp_x2 = x2 + padding_x
            exp_y2 = y2 + padding_y
            
            # Calculate how much we need to extend beyond frame boundaries
            left_extend = max(0, -exp_x1)
            top_extend = max(0, -exp_y1)
            right_extend = max(0, exp_x2 - w)
            bottom_extend = max(0, exp_y2 - h)
            
            # Clamp to frame boundaries
            crop_x1 = max(0, exp_x1)
            crop_y1 = max(0, exp_y1)
            crop_x2 = min(w, exp_x2)
            crop_y2 = min(h, exp_y2)
            
            # Extract the available crop
            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # If we need to extend beyond frame boundaries, pad the crop
            if left_extend > 0 or top_extend > 0 or right_extend > 0 or bottom_extend > 0:
                # Create padded crop with black borders
                padded_height = crop.shape[0] + top_extend + bottom_extend
                padded_width = crop.shape[1] + left_extend + right_extend
                padded_crop = np.zeros((padded_height, padded_width, 3), dtype=np.uint8)
                
                # Place the original crop in the center
                padded_crop[top_extend:top_extend + crop.shape[0], 
                           left_extend:left_extend + crop.shape[1]] = crop
                
                crop = padded_crop
            
            # Ensure minimum crop size and reasonable dimensions
            if crop.shape[0] > 60 and crop.shape[1] > 30:
                crops.append(crop)
            else:
                # Create placeholder crop with proper aspect ratio
                placeholder_height = max(120, int(adjusted_height * 1.2))
                placeholder_width = max(60, int(placeholder_height * target_aspect_ratio))
                crops.append(np.zeros((placeholder_height, placeholder_width, 3), dtype=np.uint8))
        
        return crops
    
    def extract_crop_masks(self, masks: List[np.ndarray], boxes: np.ndarray) -> List[np.ndarray]:
        """
        Extract crop-level masks from full frame masks using bounding boxes with improved edge handling
        
        Args:
            masks: List of full frame segmentation masks
            boxes: Bounding boxes array (N, 4) in xyxy format
            
        Returns:
            List of cropped mask images
        """
        crop_masks = []
        
        for i, (mask, box) in enumerate(zip(masks, boxes)):
            x1, y1, x2, y2 = box.astype(int)
            h, w = mask.shape[:2]
            
            # Calculate original box dimensions
            orig_width = x2 - x1
            orig_height = y2 - y1
            
            # Target aspect ratio for person crops (width:height = 1:2 approximately)
            target_aspect_ratio = 0.5  # width/height
            
            # Calculate current aspect ratio
            current_aspect_ratio = orig_width / max(orig_height, 1)
            
            # Adjust box to better aspect ratio if needed
            if current_aspect_ratio > target_aspect_ratio * 1.3:  # Too wide
                # Make it taller
                new_height = int(orig_width / target_aspect_ratio)
                height_increase = new_height - orig_height
                y1 = max(0, y1 - height_increase // 2)
                y2 = min(h, y2 + height_increase // 2)
            elif current_aspect_ratio < target_aspect_ratio * 0.7:  # Too tall/narrow
                # Make it wider
                new_width = int(orig_height * target_aspect_ratio)
                width_increase = new_width - orig_width
                x1 = max(0, x1 - width_increase // 2)
                x2 = min(w, x2 + width_increase // 2)
            
            # Recalculate dimensions after aspect ratio adjustment
            adjusted_width = x2 - x1
            adjusted_height = y2 - y1
            
            # Expand bounding box with adaptive padding
            padding_x = max(20, int(adjusted_width * 0.15))
            padding_y = max(25, int(adjusted_height * 0.08))
            
            # Expand the box
            exp_x1 = x1 - padding_x
            exp_y1 = y1 - padding_y
            exp_x2 = x2 + padding_x
            exp_y2 = y2 + padding_y
            
            # Calculate extensions needed
            left_extend = max(0, -exp_x1)
            top_extend = max(0, -exp_y1)
            right_extend = max(0, exp_x2 - w)
            bottom_extend = max(0, exp_y2 - h)
            
            # Clamp to frame boundaries
            crop_x1 = max(0, exp_x1)
            crop_y1 = max(0, exp_y1)
            crop_x2 = min(w, exp_x2)
            crop_y2 = min(h, exp_y2)
            
            # Extract the available mask crop
            mask_crop = mask[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # If we need to extend beyond frame boundaries, pad the mask
            if left_extend > 0 or top_extend > 0 or right_extend > 0 or bottom_extend > 0:
                # Create padded mask (background is 0 for masks)
                padded_height = mask_crop.shape[0] + top_extend + bottom_extend
                padded_width = mask_crop.shape[1] + left_extend + right_extend
                padded_mask = np.zeros((padded_height, padded_width), dtype=np.uint8)
                
                # Place the original mask crop in the center
                padded_mask[top_extend:top_extend + mask_crop.shape[0], 
                           left_extend:left_extend + mask_crop.shape[1]] = mask_crop
                
                mask_crop = padded_mask
            
            # Ensure minimum crop size
            if mask_crop.shape[0] > 60 and mask_crop.shape[1] > 30:
                crop_masks.append(mask_crop)
            else:
                # Create placeholder mask with proper aspect ratio
                placeholder_height = max(120, int(adjusted_height * 1.2))
                placeholder_width = max(60, int(placeholder_height * target_aspect_ratio))
                crop_masks.append(np.zeros((placeholder_height, placeholder_width), dtype=np.uint8))
        
        return crop_masks
    
    def match_detections_to_tracks(self, 
                                  current_features: torch.Tensor, 
                                  current_boxes: np.ndarray,
                                  frame_number: int) -> List[int]:
        """
        Enhanced matching with memory gallery and consistent identity assignment
        """
        assignments = []
        
        try:
            # Safety checks
            if current_features is None or len(current_features) == 0:
                return assignments
                
            if current_boxes is None or len(current_boxes) == 0:
                return assignments
                
            # Ensure consistent input sizes
            if len(current_features) != len(current_boxes):
                print(f"WARNING: Feature/box size mismatch: {len(current_features)} vs {len(current_boxes)}")
                min_size = min(len(current_features), len(current_boxes))
                current_features = current_features[:min_size]
                current_boxes = current_boxes[:min_size]
            
            num_detections = len(current_features)
            
            if len(self.track_features) == 0 and len(self.track_gallery) == 0:
                # No existing tracks or gallery, assign new IDs
                for _ in range(num_detections):
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
                # Proper feature-based assignment using similarity matching
                assignments = []
                used_track_ids = set()
                
                # Calculate similarity matrix between current detections and active tracks
                active_features_tensor = torch.stack(active_track_features)
                similarity_matrix = self.reid_model.compute_similarity(current_features, active_features_tensor)
                
                # For each detection, find the best matching track
                for i in range(num_detections):
                    best_track_id = None
                    best_similarity = 0.0
                    
                    # Check similarity with all active tracks
                    for j, track_id in enumerate(active_track_ids):
                        if track_id in used_track_ids:
                            continue  # Track already assigned
                            
                        similarity = similarity_matrix[i, j].item()
                        
                        # Add spatial proximity check to improve matching
                        if len(current_boxes) > i and track_id in self.track_history and len(self.track_history[track_id]) > 0:
                            curr_center = np.array([(current_boxes[i][0] + current_boxes[i][2]) / 2, 
                                                  (current_boxes[i][1] + current_boxes[i][3]) / 2])
                            last_pos = np.array(self.track_history[track_id][-1][:2])
                            spatial_distance = np.linalg.norm(curr_center - last_pos)
                            
                            # Boost similarity if spatially close, penalize if far
                            if spatial_distance < self.spatial_distance_threshold:
                                spatial_weight = max(0.1, 1.0 - (spatial_distance / self.spatial_distance_threshold))
                                similarity = similarity * 0.7 + spatial_weight * 0.3
                            else:
                                similarity *= 0.5  # Penalize distant matches
                        
                        if similarity > best_similarity and similarity > self.similarity_threshold:
                            best_similarity = similarity
                            best_track_id = track_id
                    
                    if best_track_id is not None:
                        assignments.append(best_track_id)
                        used_track_ids.add(best_track_id)
                    else:
                        # No good match found, create new ID
                        assignments.append(self.next_id)
                        self.next_id += 1
            else:
                # Fallback: create new IDs
                for _ in range(num_detections):
                    assignments.append(self.next_id)
                    self.next_id += 1
            
            # Safety check: ensure assignments list matches current_features length
            while len(assignments) < num_detections:
                assignments.append(self.next_id)
                self.next_id += 1
            
            # Trim if somehow assignments is longer than current_features
            assignments = assignments[:num_detections]
    
        except Exception as e:
            print(f"ERROR in match_detections_to_tracks: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: create new IDs for all detections
            assignments = []
            for _ in range(len(current_features)):
                assignments.append(self.next_id)
                self.next_id += 1
    
        return assignments
    
    def update_tracks(self, 
                 track_ids: List[int], 
                 features: torch.Tensor, 
                 boxes: np.ndarray, 
                 crops: List[np.ndarray],
                 frame_number: int,
                 masks: List[np.ndarray] = None) -> None:
        """
        Enhanced track update with improved feature management
        """
        active_tracks = set()
        
        for i, (track_id, feature, box, crop) in enumerate(zip(track_ids, features, boxes, crops)):
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
            
            # Store segmentation mask if available
            if masks is not None and i < len(masks):
                self.track_masks[track_id] = masks[i]
            
            # Reset missing counter and mark as active
            self.missing_frames[track_id] = 0
            active_tracks.add(track_id)
            
            # Mark as stable if enough history
            if len(self.track_history[track_id]) >= self.config.stable_track_threshold:
                self.stable_tracks.add(track_id)
    
        # Process inactive tracks with stability decay
        for track_id in list(self.track_features.keys()):
            if track_id not in active_tracks:
                self.missing_frames[track_id] += 1
                
                # Decay stability score for inactive tracks
                if track_id in self.track_stability_score:
                    self.track_stability_score[track_id] = max(0.0, self.track_stability_score[track_id] - 0.02)
                
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
        if track_id in self.track_masks:
            del self.track_masks[track_id]
        if track_id in self.track_history:
            del self.track_history[track_id]
        if track_id in self.stable_tracks:
            self.stable_tracks.remove(track_id)
        
        # Clean up identity-aware tracking attributes
        if track_id in self.track_stability_score:
            del self.track_stability_score[track_id]
        if track_id in self.recent_assignments:
            del self.recent_assignments[track_id]
    
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
    
    def track_persons(self, frame: np.ndarray, frame_number: int = 0) -> List[Tuple[int, np.ndarray, float, Optional[np.ndarray]]]:
        """
        Main tracking function with enhanced identity consistency and segmentation masks
        
        Returns:
            List of tuples: (track_id, bbox, confidence, segmentation_mask)
        """
        # YOLO detection and segmentation with device-specific optimization
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
        
        # Extract segmentation masks if available
        masks = []
        if hasattr(results[0], 'masks') and results[0].masks is not None:
            masks_data = results[0].masks.data.cpu().numpy()
            for mask_data in masks_data:
                # Resize mask to frame dimensions
                h, w = frame.shape[:2]
                mask = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_NEAREST)
                mask = (mask > 0.5).astype(np.uint8) * 255  # Convert to binary mask
                masks.append(mask)
        else:
            # Create rectangular masks from bounding boxes as fallback
            h, w = frame.shape[:2]
            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 255
                masks.append(mask)
        
        if len(boxes) == 0:
            return []
        
        # Extract person crops and features
        crops = self.extract_person_crops(frame, boxes)
        crop_masks = self.extract_crop_masks(masks, boxes)
        features = self.reid_model.extract_features(crops)
        
        if features.numel() == 0:
            return []
        
        # Match detections to tracks with enhanced consistency
        track_ids = self.match_detections_to_tracks(features, boxes, frame_number)
        
        # Update track state with improved feature management
        self.update_tracks(track_ids, features, boxes, crops, frame_number, crop_masks)
        
        # Apply sequence-based ID correction
        corrected_track_ids = self._verify_and_correct_id_switches(track_ids, boxes, features, frame_number)
        
        # Return tracking results with masks
        results_list = []
        # Safety check: ensure all lists have the same length
        min_length = min(len(corrected_track_ids), len(boxes), len(confidences))
        for i in range(min_length):
            track_id = corrected_track_ids[i]
            box = boxes[i]
            conf = confidences[i]
            mask = masks[i] if i < len(masks) else None
            results_list.append((track_id, box, conf, mask))
        
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
