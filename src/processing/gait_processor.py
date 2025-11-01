"""
Gait processing module for XGait feature extraction and analysis.
"""

import sys
import os
import cv2
import numpy as np
import time
import torch
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config import xgaitConfig
from src.models.parsing_model import HumanParsingModel
from src.models.xgait_model import create_xgait_inference


class GaitProcessor:
    """Handles gait parsing and XGait feature extraction"""
    
    def __init__(self, config, identity_manager):
        self.config = config
        self.identity_manager = identity_manager
        
        # Initialize models - YOLO masks come directly from tracker
        self.silhouette_extractor = None  # Using YOLO segmentation masks from tracker
        self.parsing_model = HumanParsingModel(
            model_path=config.model.parsing_model_path, 
            device=config.model.device
        )
        self.xgait_model = create_xgait_inference(
            model_path=config.model.xgait_model_path, 
            device=config.model.get_model_device("xgait")
        )
        
        # Create directories
        self.debug_output_dir = Path("debug_gait_parsing")
        self.debug_output_dir.mkdir(exist_ok=True)
        self.visualization_output_dir = Path("visualization_analysis")
        self.visualization_output_dir.mkdir(exist_ok=True)
        
        # Remove threading for GPU-bound operations to prevent CUDA context issues
        # Use sequential processing to avoid thread-unsafe PyTorch operations
        self.processing_queue = []  # Simple list for sequential processing (no threading)
        # Removed visualization_queue to prevent resource leaks - not used in sequential mode
        
        # PERF-007 fix: Use deque for efficient FIFO operations instead of list.pop(0)
        # Buffer management configuration
        self.sequence_buffer_size = xgaitConfig.sequence_buffer_size
        self.min_sequence_length = xgaitConfig.min_sequence_length
        self.xgait_extraction_interval = xgaitConfig.xgait_extraction_interval
        
        # Data storage with efficient deque buffers
        def create_bounded_deque():
            return deque(maxlen=self.sequence_buffer_size)
        
        def create_feature_deque():
            return deque(maxlen=10)  # Feature history limit
            
        self.track_parsing_results = defaultdict(create_bounded_deque)
        self.track_silhouettes = defaultdict(create_bounded_deque)
        self.track_parsing_masks = defaultdict(create_bounded_deque)
        self.track_crops = defaultdict(create_bounded_deque)
        self.track_gait_features = defaultdict(create_feature_deque)
        self.track_last_xgait_extraction = defaultdict(int)
        
        # Performance optimization settings
        self.parsing_frame_counter = 0
        self.parsing_skip_interval = getattr(config.identity, 'parsing_skip_interval', 3)
        self.enable_debug_outputs = getattr(config.identity, 'enable_debug_outputs', False)
        
        print(f"   XGait model weights loaded: {self.xgait_model.is_model_loaded()}")
        print(f"   Performance mode: Parsing every {self.parsing_skip_interval} frames")
        print(f"   Debug outputs: {'Enabled' if self.enable_debug_outputs else 'Disabled'}")
        
        # Test model functionality
        print("🧪 Testing model functionality...")
        
        # Test silhouette extractor - skip since we use masks from tracker
        print("   Silhouette extraction: ✅ USING YOLO MASKS FROM TRACKER")
        silhouette_test_passed = True  # Always pass since we use masks from tracker
        
        # GPU Warmup for steady utilization
        print("🔥 Warming up GPU for steady performance...")
        try:
            # Warmup parsing model with consistent batch sizes
            warmup_images = [np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8) for _ in range(6)]
            for _ in range(3):  # Run 3 warmup iterations
                _ = self.parsing_model.extract_parsing(warmup_images)
            print("   Parsing model warmup: ✅ COMPLETE")
            parsing_test_passed = True
        except Exception as e:
            print(f"   Parsing model warmup: ❌ FAILED ({e})")
            parsing_test_passed = False
        
        # Test XGait model with warmup
        try:
            # Use longer sequence for XGait test (needs minimum length)
            test_silhouettes = [np.random.randint(0, 255, (256, 128), dtype=np.uint8) for _ in range(10)]
            test_parsing = [np.random.randint(0, 7, (256, 128), dtype=np.uint8) for _ in range(10)]
            
            # Warmup XGait model
            for _ in range(2):  # Run 2 warmup iterations
                xgait_test = self.xgait_model.extract_features_from_sequence(
                    silhouettes=test_silhouettes, parsing_masks=test_parsing
                )
            
            xgait_test_passed = xgait_test is not None and xgait_test.size > 0
            print(f"   XGait model warmup: {'✅ COMPLETE' if xgait_test_passed else '❌ FAILED'}")
        except Exception as e:
            print(f"   XGait model warmup: ❌ FAILED ({e})")
            xgait_test_passed = False
        
        if not (silhouette_test_passed and parsing_test_passed and xgait_test_passed):
            print("⚠️  Some models failed tests - silhouette extraction might not work properly")
            if not silhouette_test_passed:
                print("   - Consider checking U²-Net model weights")
            if not parsing_test_passed:
                print("   - Consider checking human parsing model weights")  
            if not xgait_test_passed:
                print("   - Consider checking XGait model weights")
        else:
            print("✅ All model tests passed - GaitParsing pipeline ready")
    
    def clear_data(self) -> None:
        """Clear all processing data"""
        self.track_gait_features.clear()
        self.track_silhouettes.clear()
        self.track_parsing_masks.clear()
        self.track_crops.clear()
        self.track_parsing_results.clear()
        self.track_last_xgait_extraction.clear()
    
    def process_frame(self, frame: np.ndarray, tracking_results: List, frame_count: int) -> None:
        """
        Process gait parsing for all tracks in the current frame with performance optimizations.
        
        Args:
            frame: Input frame
            tracking_results: List of (track_id, box, confidence) or (track_id, box, confidence, mask) tuples
            frame_count: Current frame number
        """        
        # Performance optimization: Skip parsing on some frames for speed
        self.parsing_frame_counter += 1
        should_process_parsing = (self.parsing_frame_counter % self.parsing_skip_interval == 0)
        
        # Always track detections, but skip heavy parsing operations based on interval
        # Batch collection for parsing processing
        parsing_batch = []  # List of (track_id, crop_resized, crop_mask)
        
        for result in tracking_results:
            # Handle both old (3-tuple) and new (4-tuple) formats
            if len(result) == 4:
                track_id, box, conf, mask = result
            else:
                track_id, box, conf = result
                mask = None
            
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
            
            # Extract crop mask if available
            crop_mask = None
            if mask is not None:
                crop_mask = mask[y1:y2, x1:x2]
            
            # Update identity manager with context data for enhanced gallery
            if hasattr(self.identity_manager, 'update_track_context'):
                self.identity_manager.update_track_context(track_id, crop, (x1, y1, x2, y2))
            
            # Process crops that have minimum size
            if crop.shape[0] > 40 and crop.shape[1] > 20:
                # ALWAYS store silhouettes (from YOLO masks) - they're already computed, no overhead
                if crop_mask is not None:
                    # Resize crop and mask for storage
                    crop_resized = cv2.resize(crop, (128, 256))
                    mask_resized = cv2.resize(crop_mask, (128, 256), interpolation=cv2.INTER_NEAREST)
                    silhouette = (mask_resized > 127).astype(np.uint8) * 255
                    
                    # Store silhouette and crop EVERY frame (no skip)
                    self.track_silhouettes[track_id].append(silhouette)
                    self.track_crops[track_id].append(crop_resized)
                
                    # Collect for batch parsing if needed
                    if should_process_parsing:
                        parsing_batch.append((track_id, crop_resized, mask_resized))
        
        # Batch process parsing for all tracks at once (much faster!)
        if should_process_parsing and parsing_batch:
            self._process_batch_parsing(parsing_batch, frame_count)
        
        # No longer need to collect from queue since we process directly
        pass
        
        # Debug visualizations - save more frequently to show silhouette, parsing, and gait processing
        if (self.enable_debug_outputs and
            frame_count % 30 == 0 and  # Every 30 frames for better debugging
            len(tracking_results) > 0):  # Only when there are actual tracks
            # Convert back to 3-tuple format for visualization compatibility
            vis_results = [(r[0], r[1], r[2]) for r in tracking_results]
            self._save_debug_visualization(frame, vis_results, frame_count)
    
    def _process_batch_parsing(self, parsing_batch: List[Tuple[int, np.ndarray, np.ndarray]], frame_count: int):
        """
        Process parsing for multiple tracks in a batch for improved speed and steady GPU utilization
        
        Args:
            parsing_batch: List of (track_id, crop_resized, mask_resized) tuples
            frame_count: Current frame number
        """
        if not parsing_batch:
            return
        
        try:
            # Get optimal batch size from config for steady GPU usage
            max_batch_size = getattr(self.config.model.device_config, 'max_batch_size', 6)
            
            # Extract batch data
            track_ids = [item[0] for item in parsing_batch]
            crops_batch = [item[1] for item in parsing_batch]
            
            if self.config.verbose:
                print(f"🔄 Batch parsing {len(track_ids)} tracks at frame {frame_count}")
            
            # Process in optimal-sized batches for steady GPU utilization
            all_parsing_results = []
            for batch_start in range(0, len(crops_batch), max_batch_size):
                batch_end = min(batch_start + max_batch_size, len(crops_batch))
                mini_batch = crops_batch[batch_start:batch_end]
                
                # Batch process parsing with consistent batch sizes
                batch_results = self.parsing_model.extract_parsing(mini_batch)
                all_parsing_results.extend(batch_results)
            
            parsing_results = all_parsing_results
            
            if self.config.verbose:
                print(f"✅ Batch parsing complete: {len(parsing_results)} results")
            
            # Store results for each track
            for i, track_id in enumerate(track_ids):
                parsing_mask = parsing_results[i] if i < len(parsing_results) else np.zeros((256, 128), dtype=np.uint8)
                self.track_parsing_masks[track_id].append(parsing_mask)
                
                if self.config.verbose:
                    unique_parts = len(np.unique(parsing_mask))
                    # print(f"  Track {track_id}: {unique_parts} body parts, Parse buffer: {len(self.track_parsing_masks[track_id])}")
                
                # Check if ready for XGait extraction
                if (len(self.track_silhouettes[track_id]) >= self.min_sequence_length and 
                    frame_count - self.track_last_xgait_extraction[track_id] >= self.xgait_extraction_interval):
                    self._extract_xgait_features(track_id, frame_count)
            
            # Synchronize GPU operations for steady utilization (MPS)
            if self.config.model.device == "mps" and hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()
                    
        except Exception as e:
            # Always print batch parsing errors, even if verbose is off
            print(f"❌ Batch parsing failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_xgait_features(self, track_id: int, frame_count: int):
        """
        Extract XGait features for a track
        
        Args:
            track_id: Track identifier
            frame_count: Current frame number
        """
        try:
            # Get sequences
            silhouette_sequence = list(self.track_silhouettes[track_id])
            parsing_sequence = list(self.track_parsing_masks[track_id])
            crop_sequence = list(self.track_crops[track_id])
            
            # Synchronize sequences
            min_length = min(len(silhouette_sequence), len(parsing_sequence), len(crop_sequence))
            if min_length < self.min_sequence_length:
                return
            
            # Use most recent frames
            silhouette_sequence = silhouette_sequence[-min_length:]
            parsing_sequence = parsing_sequence[-min_length:]
            crop_sequence = crop_sequence[-min_length:]
            
            # Extract features
            feature_vector = self.xgait_model.extract_features_from_sequence(
                silhouettes=silhouette_sequence,
                parsing_masks=parsing_sequence
            )
            
            if feature_vector is not None and hasattr(feature_vector, 'size') and feature_vector.size > 0:
                # Compute quality
                sequence_quality = self._compute_sequence_quality(crop_sequence, parsing_sequence)
                
                # Update identity manager
                self.identity_manager.update_track_embeddings(track_id, feature_vector, sequence_quality)
                self.track_last_xgait_extraction[track_id] = frame_count
                
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️ XGait extraction failed for track {track_id}: {e}")
    
    def _process_single_track_parsing(self, track_id: int, crop: np.ndarray, frame_count: int, crop_mask: np.ndarray = None) -> Dict:
        """
        Process gait parsing pipeline for a single track.
        
        Args:
            track_id: Track identifier
            crop: Cropped person image
            frame_count: Current frame number
            crop_mask: Optional segmentation mask for the crop
            
        Returns:
            Dictionary containing processing results
        """
        try:
            start_time = time.time()

            # Resize crop to standard size (width, height) = (128, 256)
            crop_resized = cv2.resize(crop, (128, 256))
            
            # Step 1: Use segmentation mask directly for silhouette
            try:
                if crop_mask is not None:
                    # Use YOLO segmentation mask directly for silhouette
                    mask_resized = cv2.resize(crop_mask, (128, 256), interpolation=cv2.INTER_NEAREST)
                    # Ensure binary mask
                    silhouette = (mask_resized > 127).astype(np.uint8) * 255
                else:
                    # Create fallback silhouette if no mask available
                    if self.config.verbose:
                        print(f"⚠️  No segmentation mask available for track {track_id}, using fallback")
                    silhouette = np.zeros((256, 128), dtype=np.uint8)
                    silhouette[50:200, 30:98] = 255  # Basic person-like rectangle
                
                # Validate silhouette quality
                if np.sum(silhouette > 0) < 100:  # Too few foreground pixels
                    if self.config.verbose:
                        print(f"⚠️  Poor silhouette quality for track {track_id}, using fallback")
                    # Create a basic rectangular silhouette as fallback
                    silhouette = np.zeros((256, 128), dtype=np.uint8)
                    silhouette[50:200, 30:98] = 255  # Basic person-like rectangle
                    
            except Exception as e:
                if self.config.verbose:
                    print(f"⚠️  Silhouette processing failed for track {track_id}: {e}")
                silhouette = np.zeros((256, 128), dtype=np.uint8)
                silhouette[50:200, 30:98] = 255  # Basic fallback silhouette
            
            # Step 2: Extract human parsing with error handling
            try:
                parsing_results = self.parsing_model.extract_parsing([crop_resized])
                parsing_mask = parsing_results[0] if parsing_results else np.zeros((256, 128), dtype=np.uint8)
                
                if self.config.verbose and np.sum(parsing_mask > 0) > 0:
                    unique_parts = len(np.unique(parsing_mask))
                    # print(f"✅ Parsing extracted for track {track_id}: {unique_parts} body parts detected")
            except Exception as e:
                if self.config.verbose:
                    print(f"⚠️  Parsing extraction failed for track {track_id}: {e}")
                parsing_mask = np.zeros((256, 128), dtype=np.uint8)
            
            # Step 3: Store parsing masks (silhouettes and crops are stored in process_frame)
            # Note: Silhouettes are already stored in process_frame every frame
            # We only need to store parsing masks here (which are computed less frequently)
            self.track_parsing_masks[track_id].append(parsing_mask)
            
            if self.config.verbose:
                print(f"📊 Track {track_id} buffers - Sil: {len(self.track_silhouettes[track_id])}, "
                      f"Parse: {len(self.track_parsing_masks[track_id])}, "
                      f"Crops: {len(self.track_crops[track_id])}")
            
            # Step 4: Extract XGait features if conditions are met
            feature_vector = np.zeros(16384)  # XGait full feature dimension (256x64)
            xgait_extracted = False
            
            if (len(self.track_silhouettes[track_id]) >= self.min_sequence_length and 
                frame_count - self.track_last_xgait_extraction[track_id] >= self.xgait_extraction_interval):
                try:
                    # Get sequences - silhouettes stored every frame, parsing stored less frequently
                    silhouette_sequence = list(self.track_silhouettes[track_id])
                    parsing_sequence = list(self.track_parsing_masks[track_id])
                    crop_sequence = list(self.track_crops[track_id])
                    
                    # Synchronize sequences: use the shorter length to avoid mismatches
                    # This happens because parsing is done less frequently than silhouettes
                    min_length = min(len(silhouette_sequence), len(parsing_sequence), len(crop_sequence))
                    if min_length < self.min_sequence_length:
                        if self.config.verbose:
                            print(f"⚠️ Insufficient synchronized sequence length for track {track_id}: {min_length} < {self.min_sequence_length}")
                    else:
                        # Use the most recent min_length frames from each sequence
                        silhouette_sequence = silhouette_sequence[-min_length:]
                        parsing_sequence = parsing_sequence[-min_length:]
                        crop_sequence = crop_sequence[-min_length:]
                        
                    # Additional safety checks before XGait extraction
                    if not silhouette_sequence or not parsing_sequence:
                        if self.config.verbose:
                            print(f"⚠️ Empty sequences for track {track_id}")
                    elif len(silhouette_sequence) != len(parsing_sequence):
                        if self.config.verbose:
                            print(f"⚠️ Sequence length mismatch after sync for track {track_id}: sil={len(silhouette_sequence)}, par={len(parsing_sequence)}")
                    else:
                        # Safely extract features with enhanced error handling
                        feature_vector = None
                        extraction_attempts = 0
                        max_attempts = 2  # Allow one retry
                        
                        while extraction_attempts < max_attempts and feature_vector is None:
                            extraction_attempts += 1
                            try:
                                # Validate inputs before extraction
                                if not silhouette_sequence or not parsing_sequence:
                                    if self.config.verbose:
                                        print(f"⚠️ Empty sequences for track {track_id}, attempt {extraction_attempts}")
                                    break
                                    
                                # Check for dimension consistency
                                expected_shape = (256, 128)
                                valid_silhouettes = all(s.shape == expected_shape for s in silhouette_sequence)
                                valid_parsing = all(p.shape == expected_shape for p in parsing_sequence)
                                
                                if not valid_silhouettes or not valid_parsing:
                                    if self.config.verbose:
                                        print(f"⚠️ Inconsistent sequence dimensions for track {track_id}")
                                    break
                                
                                # Attempt feature extraction
                                feature_vector = self.xgait_model.extract_features_from_sequence(
                                    silhouettes=silhouette_sequence,
                                    parsing_masks=parsing_sequence
                                )
                                
                                # Validate output
                                if feature_vector is None or not hasattr(feature_vector, 'size') or feature_vector.size == 0:
                                    if self.config.verbose:
                                        print(f"⚠️ Invalid feature vector returned for track {track_id}, attempt {extraction_attempts}")
                                    feature_vector = None
                                    if extraction_attempts < max_attempts:
                                        continue  # Try again
                                    
                            except Exception as extraction_error:
                                if self.config.verbose:
                                    print(f"⚠️ XGait extraction failed for track {track_id}, attempt {extraction_attempts}: {extraction_error}")
                                feature_vector = None
                                if extraction_attempts < max_attempts:
                                    # time module already imported at module level
                                    import time as time_module  # Use alias to avoid shadowing
                                    time_module.sleep(0.1)  # Brief pause before retry
                                    continue
                    
                    # Only proceed if feature extraction was successful
                    if feature_vector is not None and hasattr(feature_vector, 'size') and feature_vector.size > 0:
                        # Use parsing-based quality calculation
                        sequence_quality = self._compute_sequence_quality(crop_sequence, parsing_sequence)
                        
                        # Update identity manager with embeddings
                        try:
                            self.identity_manager.update_track_embeddings(track_id, feature_vector, sequence_quality)
                            self.track_last_xgait_extraction[track_id] = frame_count
                            xgait_extracted = True
                            
                            # PERF-007 fix: deque with maxlen automatically manages size efficiently
                            # No need for manual pop(0) - deque handles overflow automatically
                            self.track_gait_features[track_id].append(feature_vector)
                        except Exception as update_error:
                            if self.config.verbose:
                                print(f"⚠️ Failed to update embeddings for track {track_id}: {update_error}")
                    else:
                        if self.config.verbose:
                            print(f"⚠️ XGait feature extraction failed for track {track_id} (invalid or empty features)")
                        
                except Exception as e:
                    if self.config.verbose:
                        print(f"⚠️ XGait extraction error for track {track_id}: {e}")
                        import traceback
                        print(f"   Traceback: {traceback.format_exc()}")
                    # Don't set feature_vector to zeros - leave as initialized
            
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
            import traceback
            error_details = traceback.format_exc()
            if self.config.verbose:
                print(f"❌ Exception in _process_single_track_parsing for track {track_id}:")
                print(error_details)
            return {
                'track_id': track_id,
                'frame_count': frame_count,
                'success': False,
                'error': str(e),
                'traceback': error_details
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

    # Legacy threading methods removed - using sequential batch processing instead
    
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
        # Processing is now synchronous, no cleanup needed
        if self.config.verbose:
            print(f"🏁 Processing completed at frame {frame_count}")
    
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
        """Save debug visualization with silhouettes and parsing to verify pipeline correctness"""
        try:
            max_tracks_to_show = min(len(tracking_results), 3)
            if max_tracks_to_show == 0:
                return
            
            # Create 4-row visualization: Crop, Silhouette, Parsing, Status
            fig, axes = plt.subplots(4, max_tracks_to_show, figsize=(max_tracks_to_show * 3, 12))
            if max_tracks_to_show == 1:
                axes = axes.reshape(4, 1)
            elif axes.ndim == 1:
                axes = axes.reshape(-1, 1)
            
            # Show parsing interval info in title
            fig.suptitle(f'Frame {frame_count} - Silhouette & Parsing Debug (Parsing every {self.parsing_skip_interval} frames)', fontsize=14)
            
            for idx, (track_id, box, conf) in enumerate(tracking_results[:max_tracks_to_show]):
                col_idx = idx
                x1, y1, x2, y2 = box.astype(int)
                
                # Ensure coordinates are within frame bounds
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                crop = frame[y1:y2, x1:x2]
                
                if crop.size > 0:
                    crop_resized = cv2.resize(crop, (96, 192))
                    
                    # Row 1: Person crop
                    ax = axes[0, col_idx]
                    ax.imshow(cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB))
                    ax.set_title(f'Track {track_id} - Crop', fontsize=10)
                    ax.axis('off')
                    
                    # Row 2: Silhouette (from YOLO segmentation mask)
                    ax = axes[1, col_idx]
                    if track_id in self.track_silhouettes and len(self.track_silhouettes[track_id]) > 0:
                        latest_silhouette = self.track_silhouettes[track_id][-1]
                        ax.imshow(latest_silhouette, cmap='gray')
                        ax.set_title(f'Silhouette ({len(self.track_silhouettes[track_id])} total)', fontsize=10)
                    else:
                        ax.text(0.5, 0.5, 'No Silhouette\nYet', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title('Silhouette Processing...', fontsize=10)
                    ax.axis('off')
                    
                    # Row 3: Human Parsing
                    ax = axes[2, col_idx]
                    if track_id in self.track_parsing_masks and len(self.track_parsing_masks[track_id]) > 0:
                        # Get latest parsing mask directly from buffer
                        parsing_mask = self.track_parsing_masks[track_id][-1]
                        
                        # Create color map for parsing visualization
                        parsing_colors = np.array([
                            [0, 0, 0],       # background - black
                            [255, 0, 0],     # head - red
                            [0, 255, 0],     # torso - green  
                            [0, 0, 255],     # right arm - blue
                            [255, 255, 0],   # left arm - yellow
                            [255, 0, 255],   # right leg - magenta
                            [0, 255, 255]    # left leg - cyan
                        ]) / 255.0
                        
                        # Convert parsing mask to RGB
                        parsing_rgb = np.zeros((parsing_mask.shape[0], parsing_mask.shape[1], 3))
                        for i in range(min(7, len(parsing_colors))):
                            mask = parsing_mask == i
                            if np.any(mask):
                                parsing_rgb[mask] = parsing_colors[i]
                        
                        ax.imshow(parsing_rgb)
                        unique_parts = len(np.unique(parsing_mask))
                        ax.set_title(f'Parsing ({unique_parts} parts)', fontsize=10)
                    else:
                        ax.text(0.5, 0.5, 'No Parsing\nYet', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title('Parsing Processing...', fontsize=10)
                    ax.axis('off')
                    
                    # Row 4: Status and metrics
                    ax = axes[3, col_idx]
                    sil_count = len(self.track_silhouettes.get(track_id, []))
                    parse_count = len(self.track_parsing_masks.get(track_id, []))  # Fixed: use track_parsing_masks not track_parsing_results
                    gait_count = len(self.track_gait_features.get(track_id, []))
                    
                    status_text = f'ID: {track_id}\nConf: {conf:.2f}\nSil: {sil_count}\nParse: {parse_count}\nGait: {gait_count}'
                    ax.text(0.5, 0.5, status_text, ha='center', va='center', 
                           transform=ax.transAxes, fontsize=9, family='monospace')
                    ax.set_title('Pipeline Status', fontsize=10)
                    ax.axis('off')
                else:
                    # Handle empty crops
                    for row in range(4):
                        axes[row, col_idx].text(0.5, 0.5, 'No Crop', ha='center', va='center')
                        axes[row, col_idx].set_title(['Crop', 'Silhouette', 'Parsing', 'Status'][row], fontsize=10)
                        axes[row, col_idx].axis('off')
            
            # Save debug visualization
            output_path = self.debug_output_dir / f"frame_{frame_count:05d}_debug.png"
            plt.savefig(output_path, dpi=80, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            if self.config.verbose:
                print(f"🎨 Saved silhouette & parsing debug: {output_path}")
                
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Error saving debug visualization: {e}")
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
