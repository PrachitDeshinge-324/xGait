"""
Gait processing module for XGait feature extraction and analysis.
"""

import sys
import os
import cv2
import numpy as np
import time
import queue
import threading
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config import xgaitConfig
from src.models.silhouette_model import SilhouetteExtractor
from src.models.parsing_model import HumanParsingModel
from src.models.xgait_model import create_xgait_inference


class GaitProcessor:
    """Handles gait parsing and XGait feature extraction"""
    
    def __init__(self, config, identity_manager):
        self.config = config
        self.identity_manager = identity_manager
        
        # Initialize models
        self.silhouette_extractor = SilhouetteExtractor(device=config.model.device)
        self.parsing_model = HumanParsingModel(
            model_path='weights/parsing_u2net.pth', 
            device=config.model.device
        )
        self.xgait_model = create_xgait_inference(
            model_path='weights/Gait3D-XGait-120000.pt', 
            device=config.model.get_model_device("xgait")
        )
        
        # Create directories
        self.debug_output_dir = Path("debug_gait_parsing")
        self.debug_output_dir.mkdir(exist_ok=True)
        self.visualization_output_dir = Path("visualization_analysis")
        self.visualization_output_dir.mkdir(exist_ok=True)
        
        # Threading for parallel processing
        self.parsing_executor = ThreadPoolExecutor(max_workers=2)
        self.parsing_queue = queue.Queue(maxsize=50)
        self.visualization_queue = queue.Queue(maxsize=20)
        
        # Data storage
        self.track_parsing_results = defaultdict(list)
        self.track_silhouettes = defaultdict(list)
        self.track_parsing_masks = defaultdict(list)
        self.track_crops = defaultdict(list)  # Store crop sequences for quality calculation
        self.track_gait_features = defaultdict(list)
        self.track_last_xgait_extraction = defaultdict(int)
        
        # Buffer management
        self.sequence_buffer_size = xgaitConfig.sequence_buffer_size
        self.min_sequence_length = xgaitConfig.min_sequence_length
        self.xgait_extraction_interval = xgaitConfig.xgait_extraction_interval
        
        print(f"   XGait model weights loaded: {self.xgait_model.is_model_loaded()}")
    
    def clear_data(self) -> None:
        """Clear all processing data"""
        self.track_gait_features.clear()
        self.track_silhouettes.clear()
        self.track_parsing_masks.clear()
        self.track_crops.clear()
        self.track_parsing_results.clear()
        self.track_last_xgait_extraction.clear()
    
    def process_frame(self, frame: np.ndarray, tracking_results: List[Tuple[int, any, float]], frame_count: int) -> None:
        """
        Process gait parsing for all tracks in the current frame.
        
        Args:
            frame: Input frame
            tracking_results: List of (track_id, box, confidence) tuples
            frame_count: Current frame number
        """
        # Submit parsing tasks for every track on every frame
        for track_id, box, conf in tracking_results:
            x1, y1, x2, y2 = box.astype(int)
            
            # Add padding to the bounding box
            padding = 15
            h, w = frame.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # Extract crop
            crop = frame[y1:y2, x1:x2]
            
            # Update identity manager with context data for enhanced gallery
            if hasattr(self.identity_manager, 'update_track_context'):
                self.identity_manager.update_track_context(track_id, crop, (x1, y1, x2, y2))
            
            # Process crops that have minimum size
            if crop.shape[0] > 40 and crop.shape[1] > 20:
                # Submit to thread pool (non-blocking)
                if not self.parsing_queue.full():
                    future = self.parsing_executor.submit(
                        self._process_single_track_parsing, 
                        track_id, crop.copy(), frame_count
                    )
                    try:
                        self.parsing_queue.put_nowait((track_id, future))
                    except queue.Full:
                        pass
        
        # Process completed parsing results
        self._collect_parsing_results(frame_count)
        
        # Save debug visualizations every 10 frames
        if frame_count % 10 == 0:
            self._save_debug_visualization(frame, tracking_results, frame_count)
    
    def _process_single_track_parsing(self, track_id: int, crop: np.ndarray, frame_count: int) -> Dict:
        """
        Process gait parsing pipeline for a single track.
        
        Args:
            track_id: Track identifier
            crop: Cropped person image
            frame_count: Current frame number
            
        Returns:
            Dictionary containing processing results
        """
        try:
            start_time = time.time()
            
            # Resize crop to standard size
            crop_resized = cv2.resize(crop, (128, 256))
            
            # Step 1: Extract silhouette
            silhouettes = self.silhouette_extractor.extract_silhouettes([crop_resized])
            silhouette = silhouettes[0] if silhouettes else np.zeros((256, 128), dtype=np.uint8)
            
            # Step 2: Extract human parsing
            parsing_results = self.parsing_model.extract_parsing([crop_resized])
            parsing_mask = parsing_results[0] if parsing_results else np.zeros((256, 128), dtype=np.uint8)
            
            # Step 3: Store in sequence buffers
            self.track_silhouettes[track_id].append(silhouette)
            self.track_parsing_masks[track_id].append(parsing_mask)
            self.track_crops[track_id].append(crop_resized)  # Store the crop for quality calculation
            
            # Keep only recent frames
            if len(self.track_silhouettes[track_id]) > self.sequence_buffer_size:
                self.track_silhouettes[track_id].pop(0)
                self.track_parsing_masks[track_id].pop(0)
                self.track_crops[track_id].pop(0)
            
            # Step 4: Extract XGait features if conditions are met
            feature_vector = np.zeros(256)
            xgait_extracted = False
            
            if (len(self.track_silhouettes[track_id]) >= self.min_sequence_length and 
                frame_count - self.track_last_xgait_extraction[track_id] >= self.xgait_extraction_interval):
                try:
                    crop_sequence = self.track_crops[track_id].copy()
                    parsing_sequence = self.track_parsing_masks[track_id].copy()
                    
                    # Extract features using silhouettes for XGait (still needed for feature extraction)
                    silhouette_sequence = self.track_silhouettes[track_id].copy()
                    
                    feature_vector = self.xgait_model.extract_features_from_sequence(
                        silhouettes=silhouette_sequence,
                        parsing_masks=parsing_sequence
                    )
                    
                    # Use parsing-based quality calculation
                    sequence_quality = self._compute_sequence_quality(crop_sequence, parsing_sequence)
                    
                    # Update identity manager with embeddings
                    self.identity_manager.update_track_embeddings(track_id, feature_vector, sequence_quality)
                    
                    self.track_last_xgait_extraction[track_id] = frame_count
                    xgait_extracted = True
                    
                    # Store feature vector for visualization
                    self.track_gait_features[track_id].append(feature_vector)
                    
                    # Keep only recent features
                    if len(self.track_gait_features[track_id]) > 10:
                        self.track_gait_features[track_id].pop(0)
                        
                except Exception as e:
                    if self.config.verbose:
                        print(f"⚠️  XGait extraction error for track {track_id}: {e}")
                    feature_vector = np.zeros(256)
            
            processing_time = time.time() - start_time
            
            return {
                'track_id': track_id,
                'frame_count': frame_count,
                'crop': crop,
                'crop_resized': crop_resized,
                'silhouette': silhouette,
                'parsing_mask': parsing_mask,
                'feature_vector': feature_vector,
                'xgait_extracted': xgait_extracted,
                'sequence_length': len(self.track_silhouettes[track_id]),
                'processing_time': processing_time,
                'success': True
            }
            
        except Exception as e:
            return {
                'track_id': track_id,
                'frame_count': frame_count,
                'success': False,
                'error': str(e)
            }
    
    def _compute_sequence_quality(self, crop_sequence: List[np.ndarray], parsing_sequence: List[np.ndarray]) -> float:
        """
        Compute quality score for a sequence using both crop and parsing information.
        
        Args:
            crop_sequence: List of crop images
            parsing_sequence: List of human parsing masks
            
        Returns:
            Quality score between 0 and 1
        """
        if not crop_sequence or not parsing_sequence or len(crop_sequence) != len(parsing_sequence):
            return 0.0
            
        if len(crop_sequence) < 2:
            return 0.0
        
        qualities = []
        
        # 1. Parsing completeness quality
        parsing_completeness = self._compute_parsing_completeness_quality(parsing_sequence)
        qualities.append(parsing_completeness)
        
        # 2. Crop image quality
        crop_quality = self._compute_crop_quality(crop_sequence)
        qualities.append(crop_quality)
        
        # 3. Parsing temporal consistency
        parsing_consistency = self._compute_parsing_consistency_quality(parsing_sequence)
        qualities.append(parsing_consistency)
        
        # 4. Parsing confidence quality (model certainty)
        parsing_confidence = self._compute_parsing_confidence_quality(parsing_sequence)
        qualities.append(parsing_confidence)
        
        # 5. Sequence length bonus
        length_bonus = min(len(crop_sequence) / 30.0, 1.0)
        qualities.append(length_bonus)
        
        # Combine qualities with weights focused on crop and parsing quality
        if qualities:
            # Parsing completeness (35%), Crop quality (35%), Parsing consistency (15%), 
            # Parsing confidence (10%), Length bonus (5%)
            weights = [0.35, 0.35, 0.15, 0.10, 0.05][:len(qualities)]
            final_quality = sum(q * w for q, w in zip(qualities, weights)) / sum(weights)
            return min(max(final_quality, 0.0), 1.0)
        
        return 0.0
    
    def _compute_parsing_completeness_quality(self, parsing_sequence: List[np.ndarray]) -> float:
        """
        Compute quality based on how complete the human parsing is.
        
        Args:
            parsing_sequence: List of parsing masks
            
        Returns:
            Completeness quality score between 0 and 1
        """
        if not parsing_sequence:
            return 0.0
            
        completeness_scores = []
        
        # Expected body parts: head(1), body(2), r_arm(3), l_arm(4), r_leg(5), l_leg(6)
        expected_parts = [1, 2, 3, 4, 5, 6]
        essential_parts = [1, 2]  # head, body are most important
        limb_parts = [3, 4, 5, 6]  # arms and legs
        
        for parsing_mask in parsing_sequence:
            if parsing_mask.size == 0:
                completeness_scores.append(0.0)
                continue
                
            # Count detected body parts with area thresholds
            detected_parts = []
            part_areas = {}
            total_person_area = np.sum(parsing_mask > 0)
            
            for part_id in expected_parts:
                if part_id in np.unique(parsing_mask):
                    part_mask = (parsing_mask == part_id)
                    part_area = np.sum(part_mask)
                    relative_area = part_area / max(total_person_area, 1)
                    
                    # Different thresholds for different parts
                    if part_id in essential_parts:
                        min_area_threshold = 0.02  # 2% for head/body
                    else:
                        min_area_threshold = 0.01  # 1% for limbs
                    
                    if relative_area > min_area_threshold:
                        detected_parts.append(part_id)
                        part_areas[part_id] = relative_area
            
            # Calculate completeness with weighted scoring
            essential_score = 0.0
            limb_score = 0.0
            
            # Essential parts scoring (60% weight)
            detected_essential = [p for p in detected_parts if p in essential_parts]
            essential_score = len(detected_essential) / len(essential_parts)
            
            # Limb parts scoring (40% weight)
            detected_limbs = [p for p in detected_parts if p in limb_parts]
            limb_score = len(detected_limbs) / len(limb_parts)
            
            # Combined score with bonus for symmetry
            basic_score = 0.6 * essential_score + 0.4 * limb_score
            
            # Symmetry bonus (arms and legs should be balanced)
            symmetry_bonus = 0.0
            if 3 in detected_parts and 4 in detected_parts:  # both arms
                symmetry_bonus += 0.05
            if 5 in detected_parts and 6 in detected_parts:  # both legs
                symmetry_bonus += 0.05
            
            # Area distribution bonus (parts should have reasonable relative sizes)
            area_bonus = 0.0
            if 1 in part_areas and 2 in part_areas:  # head and body present
                head_body_ratio = part_areas[1] / part_areas[2]
                if 0.1 < head_body_ratio < 0.5:  # reasonable head-to-body ratio
                    area_bonus += 0.05
            
            final_score = min(basic_score + symmetry_bonus + area_bonus, 1.0)
            completeness_scores.append(final_score)
        
        return np.mean(completeness_scores) if completeness_scores else 0.0
    
    def _compute_crop_quality(self, crop_sequence: List[np.ndarray]) -> float:
        """
        Compute quality based on crop image characteristics.
        
        Args:
            crop_sequence: List of crop images
            
        Returns:
            Crop quality score between 0 and 1
        """
        if not crop_sequence:
            return 0.0
            
        quality_scores = []
        
        for crop in crop_sequence:
            if crop.size == 0:
                quality_scores.append(0.0)
                continue
                
            # Convert to grayscale for analysis
            if len(crop.shape) == 3:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = crop
            
            # 1. Sharpness (Laplacian variance) - Enhanced
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 100.0, 1.0)  # Normalize
            
            # 2. Contrast (standard deviation of pixel values) - Enhanced
            contrast_score = min(np.std(gray) / 64.0, 1.0)  # Normalize
            
            # 3. Brightness adequacy (not too dark, not too bright)
            mean_brightness = np.mean(gray)
            brightness_score = 1.0 - abs(mean_brightness - 128) / 128.0
            
            # 4. Size adequacy (prefer reasonable crop sizes)
            crop_area = crop.shape[0] * crop.shape[1]
            size_score = 1.0 if crop_area > 32 * 64 else crop_area / (32 * 64)
            
            # 5. Edge density (more edges = better defined person)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            edge_score = min(edge_density * 5.0, 1.0)  # Normalize
            
            # 6. Histogram distribution (avoid too uniform images)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_std = np.std(hist)
            hist_score = min(hist_std / 1000.0, 1.0)  # Normalize
            
            # Combine crop quality metrics with enhanced weights
            crop_quality = (0.25 * sharpness_score + 0.25 * contrast_score + 
                          0.15 * brightness_score + 0.15 * size_score + 
                          0.10 * edge_score + 0.10 * hist_score)
            quality_scores.append(crop_quality)
        
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def _compute_parsing_consistency_quality(self, parsing_sequence: List[np.ndarray]) -> float:
        """
        Compute quality based on temporal consistency of parsing masks.
        
        Args:
            parsing_sequence: List of parsing masks
            
        Returns:
            Consistency quality score between 0 and 1
        """
        if not parsing_sequence or len(parsing_sequence) < 2:
            return 0.0
            
        consistency_scores = []
        
        # Compare consecutive parsing masks
        for i in range(len(parsing_sequence) - 1):
            mask1 = parsing_sequence[i]
            mask2 = parsing_sequence[i + 1]
            
            if mask1.size == 0 or mask2.size == 0 or mask1.shape != mask2.shape:
                consistency_scores.append(0.0)
                continue
            
            # Calculate consistency for each body part with weighted importance
            part_consistencies = []
            expected_parts = [1, 2, 3, 4, 5, 6]  # head, body, r_arm, l_arm, r_leg, l_leg
            part_weights = [0.25, 0.35, 0.10, 0.10, 0.10, 0.10]  # head and body are more important
            
            weighted_consistency = 0.0
            total_weight = 0.0
            
            for part_id, weight in zip(expected_parts, part_weights):
                part_mask1 = (mask1 == part_id)
                part_mask2 = (mask2 == part_id)
                
                # Only consider parts that exist in at least one frame
                if np.any(part_mask1) or np.any(part_mask2):
                    # Calculate IoU (Intersection over Union) for this part
                    intersection = np.sum(part_mask1 & part_mask2)
                    union = np.sum(part_mask1 | part_mask2)
                    
                    if union > 0:
                        iou = intersection / union
                        weighted_consistency += weight * iou
                        total_weight += weight
            
            # Calculate final consistency score
            if total_weight > 0:
                frame_consistency = weighted_consistency / total_weight
                consistency_scores.append(frame_consistency)
        
        if not consistency_scores:
            return 0.0
        
        # Add bonus for stable sequences (low variance in consistency)
        mean_consistency = np.mean(consistency_scores)
        consistency_std = np.std(consistency_scores)
        stability_bonus = max(0.0, 0.1 * (1.0 - consistency_std))
        
        final_consistency = min(mean_consistency + stability_bonus, 1.0)
        return final_consistency
    
    def _compute_parsing_confidence_quality(self, parsing_sequence: List[np.ndarray]) -> float:
        """
        Compute quality based on parsing model confidence.
        This is estimated from the clarity and distinctness of parsing boundaries.
        
        Args:
            parsing_sequence: List of parsing masks
            
        Returns:
            Confidence quality score between 0 and 1
        """
        if not parsing_sequence:
            return 0.0
            
        confidence_scores = []
        
        for parsing_mask in parsing_sequence:
            if parsing_mask.size == 0:
                confidence_scores.append(0.0)
                continue
            
            # 1. Boundary clarity (sharp transitions between parts)
            # Create edge map of parsing boundaries
            edges = cv2.Canny(parsing_mask.astype(np.uint8), 1, 2)
            edge_density = np.sum(edges > 0) / edges.size
            boundary_score = min(edge_density * 20.0, 1.0)  # Normalize
            
            # 2. Part separation (distinct regions for different parts)
            unique_parts = np.unique(parsing_mask)
            num_parts = len(unique_parts) - 1  # Exclude background (0)
            separation_score = min(num_parts / 6.0, 1.0)  # 6 is max expected parts
            
            # 3. Regional coherence (parts should be connected regions)
            coherence_scores = []
            expected_parts = [1, 2, 3, 4, 5, 6]
            
            for part_id in expected_parts:
                if part_id in unique_parts:
                    part_mask = (parsing_mask == part_id).astype(np.uint8)
                    # Count connected components
                    num_labels, _ = cv2.connectedComponents(part_mask)
                    # Ideally each part should be 1 connected component
                    coherence = 1.0 / max(num_labels - 1, 1)  # -1 to exclude background
                    coherence_scores.append(coherence)
            
            avg_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
            
            # Combine confidence metrics
            confidence = 0.4 * boundary_score + 0.3 * separation_score + 0.3 * avg_coherence
            confidence_scores.append(confidence)
        
        return np.mean(confidence_scores) if confidence_scores else 0.0

    # ...existing code...
    
    def _collect_parsing_results(self, frame_count: int) -> None:
        """Collect completed parsing results from the queue"""
        completed_tasks = []
        
        # Check for completed tasks
        while not self.parsing_queue.empty():
            try:
                track_id, future = self.parsing_queue.get_nowait()
                if future.done():
                    completed_tasks.append((track_id, future))
                else:
                    self.parsing_queue.put_nowait((track_id, future))
                    break
            except queue.Empty:
                break
        
        # Process completed results
        for track_id, future in completed_tasks:
            try:
                result = future.result()
                if result.get('success', False):
                    self._store_parsing_result(result)
            except Exception as e:
                if self.config.verbose:
                    print(f"⚠️  Error collecting parsing result for track {track_id}: {e}")
    
    def _store_parsing_result(self, result: Dict) -> None:
        """Store parsing result for a track"""
        track_id = result['track_id']
        
        # Store results with buffer management
        self.track_parsing_results[track_id].append(result)
        
        # Keep only recent results
        if len(self.track_parsing_results[track_id]) > self.sequence_buffer_size:
            self.track_parsing_results[track_id].pop(0)
        
        # Queue visualization task if needed
        if not self.visualization_queue.full():
            try:
                self.visualization_queue.put_nowait(result.copy())
            except queue.Full:
                pass
    
    def get_frame_track_embeddings(self, tracking_results: List[Tuple[int, any, float]]) -> Dict:
        """
        Get embeddings for tracks in the current frame.
        
        Args:
            tracking_results: List of (track_id, box, confidence) tuples
            
        Returns:
            Dictionary mapping track_id to (embedding, quality)
        """
        frame_track_embeddings = {}
        for track_id, _, _ in tracking_results:
            if (track_id in self.identity_manager.track_embedding_buffer and 
                self.identity_manager.track_embedding_buffer[track_id]):
                
                last_embedding = self.identity_manager.track_embedding_buffer[track_id][-1]
                last_quality = (self.identity_manager.track_quality_buffer[track_id][-1] 
                              if self.identity_manager.track_quality_buffer[track_id] else 0.5)
                frame_track_embeddings[track_id] = (last_embedding, last_quality)
        
        return frame_track_embeddings
    
    def finalize_processing(self, frame_count: int) -> None:
        """Finalize processing and cleanup"""
        if self.parsing_executor:
            try:
                self.parsing_executor.shutdown(wait=True)
                self._collect_parsing_results(frame_count)
            except Exception as e:
                if self.config.verbose:
                    print(f"⚠️  Error during parsing executor shutdown: {e}")
    
    def cleanup(self) -> None:
        """Clean up resources"""
        if hasattr(self, 'parsing_executor') and self.parsing_executor:
            try:
                self.parsing_executor.shutdown(wait=False)
            except:
                pass
    
    def get_statistics(self) -> Dict:
        """Get comprehensive gait parsing statistics including quality metrics"""
        total_tracks_processed = len(self.track_parsing_results)
        total_results = sum(len(results) for results in self.track_parsing_results.values())
        
        # Calculate average processing time
        all_times = []
        for results in self.track_parsing_results.values():
            all_times.extend([r.get('processing_time', 0) for r in results])
        avg_processing_time = np.mean(all_times) if all_times else 0
        
        # Get quality statistics
        quality_stats = self.get_quality_statistics()
        
        # Track sequence lengths
        sequence_lengths = []
        for track_id in self.track_crops.keys():
            if self.track_crops[track_id]:
                sequence_lengths.append(len(self.track_crops[track_id]))
        
        base_stats = {
            "tracks_processed": total_tracks_processed,
            "total_parsing_results": total_results,
            "avg_processing_time": avg_processing_time,
            "debug_images_saved": len(list(self.debug_output_dir.glob("*.png"))) if self.debug_output_dir.exists() else 0,
            "avg_sequence_length": np.mean(sequence_lengths) if sequence_lengths else 0,
            "max_sequence_length": np.max(sequence_lengths) if sequence_lengths else 0,
            "tracks_with_sequences": len(sequence_lengths)
        }
        
        # Merge with quality statistics
        base_stats.update(quality_stats)
        
        return base_stats
    
    def _save_debug_visualization(self, frame: np.ndarray, tracking_results: List[Tuple[int, any, float]], frame_count: int) -> None:
        """Save comprehensive debug visualization"""
        try:
            # Create comprehensive visualization
            max_tracks_to_show = min(len(tracking_results), 4)
            if max_tracks_to_show == 0:
                return
            
            fig, axes = plt.subplots(5, max_tracks_to_show, figsize=(max_tracks_to_show * 4, 20))
            if max_tracks_to_show == 1:
                axes = axes.reshape(5, 1)
            
            fig.suptitle(f'Frame {frame_count} - Complete GaitParsing Pipeline Results', 
                        fontsize=16, fontweight='bold')
            
            # Row labels
            row_labels = ['Person Crop', 'U²-Net Silhouette', 'GaitParsing Mask', 'XGait Features', 'Cosine Similarity']
            
            for idx, (track_id, box, conf) in enumerate(tracking_results[:max_tracks_to_show]):
                col_idx = idx
                
                # Get crop
                x1, y1, x2, y2 = box.astype(int)
                padding = 15
                h, w = frame.shape[:2]
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)
                crop = frame[y1:y2, x1:x2]
                
                if crop.size > 0:
                    crop_resized = cv2.resize(crop, (128, 256))
                    
                    # Row 1: Original crop
                    ax = axes[0, col_idx] if max_tracks_to_show > 1 else axes[0]
                    ax.imshow(cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB))
                    ax.set_title(f'Track {track_id}\\nConf: {conf:.2f}', fontsize=10)
                    ax.axis('off')
                    if col_idx == 0:
                        ax.text(-0.1, 0.5, row_labels[0], rotation=90, va='center', ha='right', 
                               transform=ax.transAxes, fontsize=12, fontweight='bold')
                    
                    # Row 2: Silhouette
                    ax = axes[1, col_idx] if max_tracks_to_show > 1 else axes[1]
                    if track_id in self.track_silhouettes and len(self.track_silhouettes[track_id]) > 0:
                        latest_silhouette = self.track_silhouettes[track_id][-1]
                        ax.imshow(latest_silhouette, cmap='gray')
                        ax.set_title(f'Silhouette\\n{len(self.track_silhouettes[track_id])} total', fontsize=10)
                    else:
                        ax.text(0.5, 0.5, 'Processing...', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title('No Silhouette Yet', fontsize=10)
                    ax.axis('off')
                    if col_idx == 0:
                        ax.text(-0.1, 0.5, row_labels[1], rotation=90, va='center', ha='right', 
                               transform=ax.transAxes, fontsize=12, fontweight='bold')
                    
                    # Row 3: Parsing mask
                    ax = axes[2, col_idx] if max_tracks_to_show > 1 else axes[2]
                    if track_id in self.track_parsing_results and len(self.track_parsing_results[track_id]) > 0:
                        latest_result = self.track_parsing_results[track_id][-1]
                        parsing_mask = latest_result.get('parsing_mask', np.zeros((256, 128), dtype=np.uint8))
                        
                        # Create color map for parsing
                        gait_parsing_colors = np.array([
                            [0, 0, 0],       # background
                            [255, 0, 0],     # head
                            [255, 255, 0],   # body
                            [0, 0, 255],     # right arm
                            [255, 0, 255],   # left arm
                            [0, 255, 0],     # right leg
                            [0, 255, 255]    # left leg
                        ]) / 255.0
                        
                        # Convert to RGB
                        parsing_rgb = np.zeros((parsing_mask.shape[0], parsing_mask.shape[1], 3))
                        for i in range(min(7, len(gait_parsing_colors))):
                            mask = parsing_mask == i
                            if np.any(mask):
                                parsing_rgb[mask] = gait_parsing_colors[i]
                        
                        ax.imshow(parsing_rgb)
                        unique_parts = len(np.unique(parsing_mask))
                        ax.set_title(f'Human Parsing\\n{unique_parts} parts', fontsize=10)
                    else:
                        ax.text(0.5, 0.5, 'Processing...', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title('No Parsing Yet', fontsize=10)
                    ax.axis('off')
                    if col_idx == 0:
                        ax.text(-0.1, 0.5, row_labels[2], rotation=90, va='center', ha='right', 
                               transform=ax.transAxes, fontsize=12, fontweight='bold')
                    
                    # Row 4: XGait features
                    ax = axes[3, col_idx] if max_tracks_to_show > 1 else axes[3]
                    if track_id in self.track_gait_features and len(self.track_gait_features[track_id]) > 0:
                        latest_features = self.track_gait_features[track_id][-1]
                        if latest_features.size > 0 and np.any(latest_features != 0):
                            # Reshape for visualization
                            if latest_features.size == 256:
                                feature_2d = latest_features.reshape(16, 16)
                            else:
                                feature_2d = np.zeros((16, 16))
                                flat = latest_features.flatten()
                                n = min(flat.size, 256)
                                feature_2d.flat[:n] = flat[:n]
                            
                            vmin = np.percentile(feature_2d, 1)
                            vmax = np.percentile(feature_2d, 99)
                            im = ax.imshow(feature_2d, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
                            ax.set_title(f'XGait Features\\n{latest_features.size}D vector', fontsize=10)
                        else:
                            ax.text(0.5, 0.5, 'Zero Features', ha='center', va='center', 
                                   transform=ax.transAxes, color='red')
                            ax.set_title('XGait: No Features', fontsize=10)
                    else:
                        ax.text(0.5, 0.5, 'Processing...', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title('No Features Yet', fontsize=10)
                    ax.axis('off')
                    if col_idx == 0:
                        ax.text(-0.1, 0.5, row_labels[3], rotation=90, va='center', ha='right', 
                               transform=ax.transAxes, fontsize=12, fontweight='bold')
            
            # Row 5: Cosine similarity matrix
            frame_features = []
            frame_track_ids = []
            for t_id, _, _ in tracking_results[:max_tracks_to_show]:
                feats = self.track_gait_features.get(t_id, [])
                if feats:
                    frame_features.append(feats[-1])
                    frame_track_ids.append(t_id)
            
            # Get gallery data from enhanced gallery
            gallery_stats = self.identity_manager.enhanced_gallery.get_gallery_statistics()
            gallery_names = list(gallery_stats.get('persons', {}).keys())
            
            # For now, we'll skip the similarity visualization since enhanced gallery
            # doesn't expose individual prototypes in the same way as simple gallery
            # This would need to be refactored to work with the enhanced gallery structure
            
            sim_ax = axes[4, 0]
            # Skip similarity visualization for now as it needs refactoring for enhanced gallery
            sim_ax.text(0.5, 0.5, 'Gallery Similarity\n(Enhanced Gallery\nintegration pending)', 
                       ha='center', va='center', transform=sim_ax.transAxes, fontsize=10)
            sim_ax.set_title('Track vs Gallery Cosine Similarity', fontsize=10)
            
            # Hide unused similarity plots
            for col_idx in range(1, max_tracks_to_show):
                axes[4, col_idx].axis('off')
            
            # Save visualization
            output_path = self.debug_output_dir / f"frame_{frame_count:05d}_complete_pipeline.png"
            plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            if self.config.verbose:
                print(f"🎨 Saved complete pipeline visualization: {output_path}")
                
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Error saving visualization: {e}")
                import traceback
                traceback.print_exc()
    
    def _analyze_sequence_quality_breakdown(self, crop_sequence: List[np.ndarray], parsing_sequence: List[np.ndarray]) -> Dict[str, float]:
        """
        Analyze and return a detailed breakdown of sequence quality metrics.
        
        Args:
            crop_sequence: List of crop images
            parsing_sequence: List of human parsing masks
            
        Returns:
            Dictionary with detailed quality breakdown
        """
        if not crop_sequence or not parsing_sequence or len(crop_sequence) != len(parsing_sequence):
            return {
                'overall_quality': 0.0,
                'parsing_completeness': 0.0,
                'crop_quality': 0.0,
                'parsing_consistency': 0.0,
                'parsing_confidence': 0.0,
                'sequence_length_bonus': 0.0,
                'sequence_length': 0
            }
        
        # Calculate individual quality components
        parsing_completeness = self._compute_parsing_completeness_quality(parsing_sequence)
        crop_quality = self._compute_crop_quality(crop_sequence)
        parsing_consistency = self._compute_parsing_consistency_quality(parsing_sequence)
        parsing_confidence = self._compute_parsing_confidence_quality(parsing_sequence)
        length_bonus = min(len(crop_sequence) / 30.0, 1.0)
        
        # Calculate overall quality using the same weights as the main method
        weights = [0.35, 0.35, 0.15, 0.10, 0.05]
        qualities = [parsing_completeness, crop_quality, parsing_consistency, parsing_confidence, length_bonus]
        overall_quality = sum(q * w for q, w in zip(qualities, weights)) / sum(weights)
        
        return {
            'overall_quality': overall_quality,
            'parsing_completeness': parsing_completeness,
            'crop_quality': crop_quality,
            'parsing_consistency': parsing_consistency,
            'parsing_confidence': parsing_confidence,
            'sequence_length_bonus': length_bonus,
            'sequence_length': len(crop_sequence)
        }
    
    def get_quality_statistics(self) -> Dict[str, any]:
        """
        Get comprehensive quality statistics for all tracks.
        
        Returns:
            Dictionary with quality statistics across all tracks
        """
        all_qualities = []
        quality_components = {
            'parsing_completeness': [],
            'crop_quality': [],
            'parsing_consistency': [],
            'parsing_confidence': [],
            'sequence_length_bonus': []
        }
        
        for track_id in self.track_crops.keys():
            if (track_id in self.track_parsing_masks and 
                len(self.track_crops[track_id]) > 0 and 
                len(self.track_parsing_masks[track_id]) > 0):
                
                crop_seq = self.track_crops[track_id]
                parsing_seq = self.track_parsing_masks[track_id]
                
                if len(crop_seq) >= 2:  # Minimum sequence for quality calculation
                    quality_breakdown = self._analyze_sequence_quality_breakdown(crop_seq, parsing_seq)
                    all_qualities.append(quality_breakdown['overall_quality'])
                    
                    for component, value in quality_breakdown.items():
                        if component in quality_components:
                            quality_components[component].append(value)
        
        # Calculate statistics
        stats = {
            'total_tracks_with_quality': len(all_qualities),
            'mean_overall_quality': np.mean(all_qualities) if all_qualities else 0.0,
            'std_overall_quality': np.std(all_qualities) if all_qualities else 0.0,
            'min_overall_quality': np.min(all_qualities) if all_qualities else 0.0,
            'max_overall_quality': np.max(all_qualities) if all_qualities else 0.0,
            'component_means': {}
        }
        
        # Add component statistics
        for component, values in quality_components.items():
            if values:
                stats['component_means'][component] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return stats
    
    def log_quality_information(self, track_id: int, verbose: bool = False) -> None:
        """
        Log detailed quality information for a specific track.
        
        Args:
            track_id: Track identifier
            verbose: Whether to print detailed breakdown
        """
        if (track_id not in self.track_crops or 
            track_id not in self.track_parsing_masks or
            len(self.track_crops[track_id]) < 2):
            if verbose:
                print(f"📊 Track {track_id}: Insufficient data for quality analysis")
            return
        
        crop_seq = self.track_crops[track_id]
        parsing_seq = self.track_parsing_masks[track_id]
        
        quality_breakdown = self._analyze_sequence_quality_breakdown(crop_seq, parsing_seq)
        
        if verbose:
            print(f"📊 Track {track_id} Quality Analysis:")
            print(f"   Overall Quality: {quality_breakdown['overall_quality']:.3f}")
            print(f"   └─ Parsing Completeness (35%): {quality_breakdown['parsing_completeness']:.3f}")
            print(f"   └─ Crop Quality (35%): {quality_breakdown['crop_quality']:.3f}")
            print(f"   └─ Parsing Consistency (15%): {quality_breakdown['parsing_consistency']:.3f}")
            print(f"   └─ Parsing Confidence (10%): {quality_breakdown['parsing_confidence']:.3f}")
            print(f"   └─ Sequence Length Bonus (5%): {quality_breakdown['sequence_length_bonus']:.3f}")
            print(f"   Sequence Length: {quality_breakdown['sequence_length']} frames")
        
        return quality_breakdown
