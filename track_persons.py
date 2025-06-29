#!/usr/bin/env python3
"""
Person Tracking Application with XGait Feature Extraction
A modular, high-performance person tracking system using YOLO + XGait Feature Extraction

Features:
- YOLO-based person detection and tracking
- XGait-based feature extraction for gait analysis
- Modular architecture with clean separation of concerns
- High accuracy multi-person tracking
- Real-time visualization and statistics
"""

import cv2
import sys
import traceback
import os
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import argparse
import threading
import queue
import time
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import SystemConfig, xgaitConfig
from src.trackers.person_tracker import PersonTracker
from src.utils.visualization import TrackingVisualizer, VideoWriter
from src.models.silhouette_model import SilhouetteExtractor
from src.models.parsing_model import HumanParsingModel
from src.models.xgait_model import create_xgait_inference

class PersonTrackingApp:
    """
    Main application for person tracking with XGait feature extraction
    """
    def __init__(self, config: SystemConfig, enable_identification: bool = False, enable_gait_parsing: bool = True):
        self.config = config
        self.config.validate()
        self.enable_identification = enable_identification
        self.enable_gait_parsing = enable_gait_parsing
        self.gallery_loaded = False

        # Initialize components
        self.tracker = PersonTracker(
            yolo_model_path=config.model.yolo_model_path,
            device=config.model.device,
            config=config.tracker
        )
        
        self.visualizer = TrackingVisualizer()
        self.video_writer = None
        
        # Initialize GaitParsing pipeline which includes the real XGait model
        if self.enable_gait_parsing:
            # Import gallery manager and visualization
            from src.utils.identity_gallery import IdentityGalleryManager
            from src.utils.embedding_visualization import EmbeddingVisualizer
            
            self.silhouette_extractor = SilhouetteExtractor(device=config.model.device)
            self.parsing_model = HumanParsingModel(model_path='weights/parsing_u2net.pth', device=config.model.device)
            self.xgait_model = create_xgait_inference(model_path='weights/Gait3D-XGait-120000.pt', device=config.model.get_model_device("xgait"))
            
            # Initialize identity gallery manager with more permissive settings
            self.gallery_manager = IdentityGalleryManager(
                similarity_threshold=0.6,  # More permissive for initial matching
                min_quality_threshold=0.4  # Lower quality threshold to catch more tracks
                # embedding_update_strategy="weighted_average"
            )
            
            # Connect gallery manager to XGait model
            self.xgait_model.set_gallery_manager(self.gallery_manager)
            
            # Initialize embedding visualizer
            self.embedding_visualizer = EmbeddingVisualizer()
            
            # Create debug output directory
            self.debug_output_dir = Path("debug_gait_parsing")
            self.debug_output_dir.mkdir(exist_ok=True)
            
            # Create visualization output directory
            self.visualization_output_dir = Path("visualization_analysis")
            self.visualization_output_dir.mkdir(exist_ok=True)
            
            # Thread pool for parallel processing
            self.parsing_executor = ThreadPoolExecutor(max_workers=2)
            self.parsing_queue = queue.Queue(maxsize=50)  # Limit queue size to prevent memory issues
            
            # Queue for visualization tasks to be processed in main thread
            self.visualization_queue = queue.Queue(maxsize=20)
            
            print("✅ GaitParsing pipeline initialized")
            print(f"   Debug output: {self.debug_output_dir}")
            print(f"   Visualization output: {self.visualization_output_dir}")
            print(f"   XGait model weights loaded: {self.xgait_model.is_model_loaded()}")
            print(f"   Gallery manager active: {self.gallery_manager is not None}")
        else:
            self.silhouette_extractor = None
            self.parsing_model = None
            self.xgait_model = None
            self.gallery_manager = None
            self.embedding_visualizer = None
            self.parsing_executor = None
            self.parsing_queue = None
            self.visualization_queue = None
            print("⚠️  GaitParsing disabled")
        
        # Tracking statistics
        self.track_history = defaultdict(list)
        self.id_switches = []
        self.stable_tracks = set()
        self.new_id_creations = []
        self.max_track_id_seen = 0
        
        # GaitParsing data - Enhanced sequence buffering
        self.track_parsing_results = defaultdict(list)  # Store parsing results for each track
        self.track_silhouettes = defaultdict(list)  # Store silhouettes for each track
        self.track_parsing_masks = defaultdict(list)  # Store parsing masks for each track
        self.track_gait_features = defaultdict(list)  # Store gait features for each track
        self.track_last_xgait_extraction = defaultdict(int)  # Track when we last extracted XGait features
        
        # Buffer management - increased for better sequence analysis
        self.sequence_buffer_size = xgaitConfig.sequence_buffer_size  # Number of frames to keep for XGait sequence analysis
        self.min_sequence_length = xgaitConfig.min_sequence_length   # Minimum frames needed for XGait extraction
        self.xgait_extraction_interval = xgaitConfig.xgait_extraction_interval  # Extract XGait features every N frames per track
        
        print(f"🚀 Person Tracking App initialized")
        print(f"   Video: {config.video.input_path}")
        print(f"   Model: {config.model.yolo_model_path}")
        print(f"   Device: {config.model.device}")
        print(f"   Data type: {config.model.dtype}")
        print(f"   Autocast: {config.model.use_autocast}")
        print(f"   Model compilation: {config.model.use_compile}")
        print(f"   GaitParsing: {self.enable_gait_parsing}")
        
        # Print device information
        device_info = self.tracker.get_device_info()
        print(f"📱 Device Information:")
        print(f"   • Device: {device_info['device']}")
        print(f"   • Data type: {device_info['dtype']}")
        print(f"   • Autocast: {device_info['autocast']}")
        print(f"   • Model compilation: {device_info['compile']}")
        if 'gpu_name' in device_info:
            print(f"   • GPU: {device_info['gpu_name']}")
        if 'cuda_version' in device_info:
            print(f"   • CUDA version: {device_info['cuda_version']}")
        print()
    
    def _compute_sequence_quality(self, silhouette_sequence: List[np.ndarray]) -> float:
        """
        Compute quality score for a silhouette sequence based on consistency and completeness
        
        Args:
            silhouette_sequence: List of silhouette masks
            
        Returns:
            Quality score between 0 and 1
        """
        if not silhouette_sequence or len(silhouette_sequence) < 2:
            return 0.0
        
        # Metrics for quality assessment
        qualities = []
        
        # 1. Silhouette completeness (average non-zero pixels)
        completeness_scores = []
        for sil in silhouette_sequence:
            if sil.size > 0:
                non_zero_ratio = np.sum(sil > 0) / sil.size
                completeness_scores.append(non_zero_ratio)
        
        if completeness_scores:
            avg_completeness = np.mean(completeness_scores)
            qualities.append(min(avg_completeness * 2, 1.0))  # Scale up moderate completeness
        
        # 2. Temporal consistency (similarity between consecutive frames)
        if len(silhouette_sequence) >= 2:
            consistency_scores = []
            for i in range(len(silhouette_sequence) - 1):
                sil1, sil2 = silhouette_sequence[i], silhouette_sequence[i + 1]
                if sil1.shape == sil2.shape and sil1.size > 0:
                    # Simple overlap-based similarity
                    intersection = np.sum((sil1 > 0) & (sil2 > 0))
                    union = np.sum((sil1 > 0) | (sil2 > 0))
                    if union > 0:
                        consistency_scores.append(intersection / union)
            
            if consistency_scores:
                avg_consistency = np.mean(consistency_scores)
                qualities.append(avg_consistency)
        
        # 3. Sequence length bonus
        length_bonus = min(len(silhouette_sequence) / 30.0, 1.0)  # Prefer longer sequences up to 30 frames
        qualities.append(length_bonus)
        
        # Combine qualities with weights
        if qualities:
            weights = [0.4, 0.4, 0.2][:len(qualities)]  # Completeness, consistency, length
            final_quality = sum(q * w for q, w in zip(qualities, weights)) / sum(weights)
            return min(max(final_quality, 0.0), 1.0)
        
        return 0.0
    
    def __del__(self):
        """Destructor to ensure proper cleanup"""
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'parsing_executor') and self.parsing_executor:
            try:
                self.parsing_executor.shutdown(wait=False)
            except:
                pass
        
        # Cleanup video writer
        if hasattr(self, 'video_writer') and self.video_writer:
            try:
                self.video_writer.release()
                self.video_writer = None
            except:
                pass
    
    def process_video(self) -> None:
        """Process the input video and perform tracking"""
        # Open video
        cap = cv2.VideoCapture(self.config.video.input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.config.video.input_path}")
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Initialize video writer if saving is enabled
        if self.config.video.save_annotated_video and self.config.video.output_video_path:
            output_fps = self.config.video.output_fps or fps
            codec = self.config.video.output_codec
            self.video_writer = VideoWriter(
                output_path=self.config.video.output_video_path,
                fps=output_fps,
                frame_size=(frame_width, frame_height),
                codec=codec,
                quality=self.config.video.output_quality
            )
            self.video_writer.open()
        # Hide all prints (except errors)
        verbose = self.config.verbose
        self.config.verbose = False
        frame_count = 0
        paused = False
        pbar_total = self.config.video.max_frames if self.config.video.max_frames else total_frames
        self.track_gait_features.clear()
        self.track_silhouettes.clear()
        self.track_parsing_masks.clear()
        self.track_parsing_results.clear()
        self.track_last_xgait_extraction.clear()
        start_time = time.time()
        last_frame_time = start_time
        fps_list = []
        # Load gallery state if file exists
        try:
            gallery_state_path = Path("visualization_analysis") / f"gallery_state.json"
            if os.path.exists(gallery_state_path):
                self.gallery_manager.load_gallery_state(gallery_state_path)
                self.gallery_loaded = True
        except Exception as e:
            print(f"⚠️  Error loading gallery state: {e}")
        with tqdm(total=pbar_total, desc="Processing frames", unit="frame") as pbar:
            while cap.isOpened():
                if not paused:
                    success, frame = cap.read()
                    if not success:
                        break
                    frame_count += 1
                    # Check max_frames limit
                    if self.config.video.max_frames and frame_count > self.config.video.max_frames:
                        break
                    # Perform tracking
                    tracking_results = self.tracker.track_persons(frame, frame_count)
                    # Process GaitParsing pipeline for every frame for each track (parallel)
                    if self.enable_gait_parsing and tracking_results:
                        self._process_gait_parsing_parallel(frame, tracking_results, frame_count)
                        self._collect_parsing_results(frame_count)
                    self._update_statistics(tracking_results, frame_count)
                    identification_results = {}
                    identification_confidence = {}
                    gallery_stats = None
                    identification_stats = None
                    if self.enable_gait_parsing and hasattr(self, 'track_identities'):
                        for track_id, identity_info in self.track_identities.items():
                            identification_results[track_id] = identity_info['identity']
                            identification_confidence[track_id] = identity_info['confidence']
                        if self.gallery_manager:
                            gallery_stats = self.gallery_manager.get_gallery_summary()
                            identification_stats = {
                                'total_identities': gallery_stats.get('num_identities', 0),
                                'collision_avoided': gallery_stats.get('collision_avoided_count', 0),
                                'embeddings_processed': gallery_stats.get('total_embeddings_processed', 0)
                            }
                    # --- FPS calculation ---
                    now = time.time()
                    current_fps = 1.0 / max(now - last_frame_time, 1e-6)
                    last_frame_time = now
                    fps_list.append(current_fps)
                    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0.0
                    annotated_frame = self.visualizer.draw_tracking_results(
                        frame=frame,
                        tracking_results=tracking_results,
                        track_history=self.track_history,
                        stable_tracks=self.stable_tracks,
                        frame_count=frame_count,
                        max_track_id=self.max_track_id_seen,
                        identification_results=identification_results,
                        identification_confidence=identification_confidence,
                        gallery_stats=gallery_stats,
                        identification_stats=identification_stats,
                        current_fps=current_fps,
                        avg_fps=avg_fps,
                        is_new_identity=self.get_is_new_identity_dict(),
                        gallery_loaded=self.gallery_loaded
                    )
                    if self.video_writer and self.video_writer.is_opened():
                        self.video_writer.write_frame(annotated_frame)
                    if self.config.video.display_window:
                        cv2.imshow("Person Tracking & XGait Analysis", annotated_frame)
                    if frame_count % 500 == 0:
                        self.tracker.clear_memory_cache()
                    if frame_count % 300 == 0 and self.gallery_manager:
                        current_identities = len(self.gallery_manager.identities)
                        if current_identities > 8:
                            consolidation_map = self.gallery_manager.consolidate_fragmented_tracks(
                                consolidation_threshold=0.65
                            )
                            if consolidation_map and hasattr(self, 'track_identities'):
                                for primary_id, track_list in consolidation_map.items():
                                    for track_id in track_list:
                                        if track_id in self.track_identities:
                                            self.track_identities[track_id]['identity'] = primary_id
                    pbar.update(1)
                if self.config.video.display_window:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):
                        paused = not paused
        cap.release()
        if self.config.video.display_window:
            cv2.destroyAllWindows()
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        if self.enable_gait_parsing and self.parsing_executor:
            try:
                self.parsing_executor.shutdown(wait=True)
                self._collect_parsing_results(frame_count if 'frame_count' in locals() else 999)
                gait_stats = self.get_gait_parsing_stats()
            except Exception as e:
                pass
        self.tracker.clear_memory_cache()
        self.tracker.synchronize_device()
        if self.enable_gait_parsing:
            self._generate_comprehensive_analysis(frame_count if 'frame_count' in locals() else 999)
        if self.enable_gait_parsing and self.gallery_manager:
            self._perform_track_consolidation()
        self.config.verbose = verbose
    
    def _update_statistics(self, tracking_results: List[Tuple[int, any, float]], frame_count: int) -> None:
        """Update tracking statistics"""
        
        current_tracks = {}
        for track_id, box, conf in tracking_results:
            x1, y1, x2, y2 = box.astype(int)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Monitor new ID creation
            if track_id > self.max_track_id_seen:
                self.max_track_id_seen = track_id
                self.new_id_creations.append((track_id, frame_count, center_x, center_y))
                
                # Debug new ID creation
                if frame_count > 50 and self.config.debug_mode:
                    print(f"🆕 New ID {track_id} at frame {frame_count}, position ({center_x}, {center_y})")
            
            # Store current track info
            current_tracks[track_id] = (center_x, center_y, x1, y1, x2, y2, conf)
            
            # Update track history
            self.track_history[track_id].append((center_x, center_y, frame_count, conf))
            
            # Mark tracks as stable after threshold frames
            if len(self.track_history[track_id]) >= self.config.tracker.stable_track_threshold:
                self.stable_tracks.add(track_id)
                
                # Check for ID switches
                is_switch, distance = self.tracker.detect_id_switches(track_id)
                if is_switch:
                    self.id_switches.append((track_id, distance, frame_count))
                    if self.config.debug_mode:
                        print(f"⚠️ ID Switch detected: Track {track_id}, Jump {distance:.1f}px at frame {frame_count}")
            
            # Keep only recent history
            if len(self.track_history[track_id]) > self.config.tracker.track_history_length:
                self.track_history[track_id].pop(0)
    
    def _print_final_results(self, total_frames: int) -> None:
        """Print final tracking results"""
        
        # Get tracker statistics with device info
        stats = self.tracker.get_statistics()
        device_info = stats.get('device_info', {})
        memory_info = device_info.get('memory_usage', {})
        
        print(f"\n📱 Final Device Statistics:")
        print(f"   • Device: {device_info.get('device', 'Unknown')}")
        print(f"   • Data type: {device_info.get('dtype', 'Unknown')}")
        print(f"   • Autocast enabled: {device_info.get('autocast', 'Unknown')}")
        if 'gpu_name' in device_info:
            print(f"   • GPU: {device_info['gpu_name']}")
        
        # Memory usage
        if 'allocated' in memory_info:
            allocated_mb = memory_info['allocated'] / (1024**2)
            cached_mb = memory_info.get('cached', 0) / (1024**2)
            print(f"   • Memory allocated: {allocated_mb:.1f}MB")
            print(f"   • Memory cached: {cached_mb:.1f}MB")
        elif 'system_memory' in memory_info:
            print(f"   • System memory usage: {memory_info['system_memory']:.1f}%")
        
        # Print comprehensive results
        self.visualizer.print_summary(
            max_track_id=self.max_track_id_seen,
            total_frames=total_frames,
            target_people=7  # Known from the video
        )
        
        print(f"\n📊 Detailed Statistics:")
        print(f"   • Total track IDs created: {self.max_track_id_seen}")
        print(f"   • New ID events: {len(self.new_id_creations)}")
        print(f"   • ID switches detected: {len(self.id_switches)}")
        print(f"   • Stable tracks: {len(self.stable_tracks)}")
        
        # Performance metrics
        if self.max_track_id_seen <= 8:
            performance = "EXCELLENT"
        elif self.max_track_id_seen <= 12:
            performance = "GOOD"
        elif self.max_track_id_seen <= 16:
            performance = "MODERATE"
        else:
            performance = "POOR"
        
        print(f"   • Overall performance: {performance}")
        
        # Configuration summary
        print(f"\n⚙️  Configuration Used:")
        print(f"   • Similarity threshold: {self.config.tracker.similarity_threshold}")
        print(f"   • Max missing frames: {self.config.tracker.max_missing_frames}")
        print(f"   • Confidence threshold: {self.config.tracker.confidence_threshold}")
    
    def _process_gait_parsing_parallel(self, frame: np.ndarray, tracking_results: List[Tuple[int, any, float]], frame_count: int) -> None:
        """Process GaitParsing pipeline for ALL tracks in parallel for EVERY frame"""
        if not self.enable_gait_parsing:
            return
        
        # Submit parsing tasks for EVERY track on EVERY frame (no stability requirement)
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
            
            # Process all crops that have minimum size
            if crop.shape[0] > 40 and crop.shape[1] > 20:  # Lowered threshold for more processing
                # Submit to thread pool (non-blocking)
                if not self.parsing_queue.full():
                    future = self.parsing_executor.submit(
                        self._process_single_track_parsing, 
                        track_id, crop.copy(), frame_count
                    )
                    try:
                        self.parsing_queue.put_nowait((track_id, future))
                    except queue.Full:
                        pass  # Skip if queue is full
        
        # Process completed parsing results (non-blocking)
        self._collect_parsing_results(frame_count)
        
        # Save debug visualizations every 10 frames
        if frame_count % 10 == 0:
            self._save_debug_visualization(frame, tracking_results, frame_count)
    
    def _process_single_track_parsing(self, track_id: int, crop: np.ndarray, frame_count: int) -> Dict:
        """Process GaitParsing pipeline for a single track with sequence management"""
        try:
            start_time = time.time()
            
            # Resize crop to standard size for processing
            crop_resized = cv2.resize(crop, (128, 256))
            
            # Step 1: Extract silhouette
            silhouettes = self.silhouette_extractor.extract_silhouettes([crop_resized])
            silhouette = silhouettes[0] if silhouettes else np.zeros((256, 128), dtype=np.uint8)
            
            # Step 2: Extract human parsing
            parsing_results = self.parsing_model.extract_parsing([crop_resized])
            parsing_mask = parsing_results[0] if parsing_results else np.zeros((256, 128), dtype=np.uint8)
            
            # Step 3: Store silhouette and parsing mask in sequence buffers for visualization
            self.track_silhouettes[track_id].append(silhouette)
            self.track_parsing_masks[track_id].append(parsing_mask)
            
            # Keep only recent frames for sequence analysis
            if len(self.track_silhouettes[track_id]) > self.sequence_buffer_size:
                self.track_silhouettes[track_id].pop(0)
                self.track_parsing_masks[track_id].pop(0)
            
            # Step 4: Extract XGait features if we have enough frames and it's time
            feature_vector = np.zeros(256)  # Default empty feature vector
            xgait_extracted = False
            
            if (len(self.track_silhouettes[track_id]) >= self.min_sequence_length and 
                frame_count - self.track_last_xgait_extraction[track_id] >= self.xgait_extraction_interval):
                
                try:
                    # Extract XGait features for identification
                    silhouette_sequence = self.track_silhouettes[track_id].copy()
                    parsing_sequence = self.track_parsing_masks[track_id].copy()
                    
                    # Use the enhanced XGait method with both silhouettes and parsing masks
                    feature_vector = self.xgait_model.extract_features_from_sequence(
                        silhouettes=silhouette_sequence,
                        parsing_masks=parsing_sequence
                    )
                    
                    # Process embedding with gallery manager for identification
                    if feature_vector.size > 0 and self.gallery_manager is not None:
                        # Compute sequence quality based on silhouette consistency
                        sequence_quality = self._compute_sequence_quality(silhouette_sequence)
                        
                        print(f"🔍 Processing track {track_id} for XGait extraction frame number {frame_count} seq Q {sequence_quality}")
                        # Process through gallery manager with collision avoidance
                        assigned_identity, confidence, is_new_identity = self.gallery_manager.process_track_embedding(
                            track_id=track_id,
                            embedding=feature_vector,
                            frame_number=frame_count,
                            sequence_quality=sequence_quality
                        )
                        
                        # Update track with identity information
                        if assigned_identity:
                            print(f"🎯 Track {track_id} -> {assigned_identity} "
                                  f"(confidence: {confidence:.3f}, new: {is_new_identity})")
                            
                            # Store identity info for visualization
                            if not hasattr(self, 'track_identities'):
                                self.track_identities = {}
                            self.track_identities[track_id] = {
                                'identity': assigned_identity,
                                'confidence': confidence,
                                'is_new': is_new_identity,
                                'frame_assigned': frame_count
                            }
                    
                    # Store the extracted features for legacy compatibility
                    self.track_gait_features[track_id].append(feature_vector)
                    
                    # Keep only recent features
                    if len(self.track_gait_features[track_id]) > 10:
                        self.track_gait_features[track_id].pop(0)
                    
                    # Update last extraction time
                    self.track_last_xgait_extraction[track_id] = frame_count
                    xgait_extracted = True
                    
                    if self.config.verbose:
                        print(f"🚶 XGait features extracted for track {track_id}: "
                              f"sequence_length={len(silhouette_sequence)}, "
                              f"feature_shape={feature_vector.shape}")
                        
                        # Print gallery stats periodically
                        if self.gallery_manager and frame_count % 100 == 0:
                            stats = self.gallery_manager.get_gallery_summary()
                            print(f"📚 Gallery: {stats['num_identities']} identities, "
                                  f"{stats['total_embeddings_processed']} embeddings processed")
                        
                except Exception as e:
                    if self.config.verbose:
                        print(f"⚠️  XGait feature extraction failed for track {track_id}: {e}")
                    feature_vector = np.zeros(256)
            
            processing_time = time.time() - start_time
            
            # Create result dictionary
            result = {
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
            
            return result
            
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  GaitParsing error for track {track_id}: {e}")
            
            return {
                'track_id': track_id,
                'frame_count': frame_count,
                'success': False,
                'error': str(e)
            }
    
    def _collect_parsing_results(self, frame_count: int) -> None:
        """Collect completed parsing results from the queue"""
        completed_tasks = []
        
        # Check for completed tasks (non-blocking)
        while not self.parsing_queue.empty():
            try:
                track_id, future = self.parsing_queue.get_nowait()
                if future.done():
                    completed_tasks.append((track_id, future))
                else:
                    # Put back if not done
                    self.parsing_queue.put_nowait((track_id, future))
                    break  # Don't check more if this one isn't done
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
        """Store parsing result for a track with enhanced sequence management"""
        track_id = result['track_id']
        
        # Store results with buffer management
        self.track_parsing_results[track_id].append(result)
        
        # Keep only recent results (memory management)
        if len(self.track_parsing_results[track_id]) > self.sequence_buffer_size:
            self.track_parsing_results[track_id].pop(0)
        
        # Note: silhouettes and parsing masks are now managed in _process_single_track_parsing
        # to ensure proper sequence buffering for XGait feature extraction
        
        # Queue visualization task for main thread (thread-safe)
        if not self.visualization_queue.full():
            try:
                self.visualization_queue.put_nowait(result.copy())
            except queue.Full:
                pass  # Skip if queue is full
    
    def get_gait_parsing_stats(self) -> Dict:
        """Get GaitParsing statistics"""
        if not self.enable_gait_parsing:
            return {}
        
        total_tracks_processed = len(self.track_parsing_results)
        total_results = sum(len(results) for results in self.track_parsing_results.values())
        
        # Calculate average processing time
        all_times = []
        for results in self.track_parsing_results.values():
            all_times.extend([r.get('processing_time', 0) for r in results])
        avg_processing_time = np.mean(all_times) if all_times else 0
        
        return {
            "tracks_processed": total_tracks_processed,
            "total_parsing_results": total_results,
            "avg_processing_time": avg_processing_time,
            "debug_images_saved": len(list(self.debug_output_dir.glob("*.png"))) if self.debug_output_dir.exists() else 0
        }
    
    def _save_debug_visualization(self, frame: np.ndarray, tracking_results: List[Tuple[int, any, float]], frame_count: int) -> None:
        """Save comprehensive debug visualization for all active tracks with enhanced features"""
        try:
            if not self.enable_gait_parsing:
                return
                
            # Create comprehensive visualization with all tracks (4 rows: crop, silhouette, parsing, xgait)
            max_tracks_to_show = min(len(tracking_results), 4)
            if max_tracks_to_show == 0:
                return
                
            fig, axes = plt.subplots(4, max_tracks_to_show, figsize=(max_tracks_to_show * 4, 16))
            if max_tracks_to_show == 1:
                axes = axes.reshape(4, 1)
                
            fig.suptitle(f'Frame {frame_count} - Complete GaitParsing Pipeline Results', 
                        fontsize=16, fontweight='bold')
            
            # Row labels
            row_labels = ['Person Crop', 'U²-Net Silhouette', 'GaitParsing Mask', 'XGait Features']
            
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
                    ax.set_title(f'Track {track_id}\nConf: {conf:.2f}', fontsize=10)
                    ax.axis('off')
                    if col_idx == 0:
                        ax.text(-0.1, 0.5, row_labels[0], rotation=90, va='center', ha='right', 
                               transform=ax.transAxes, fontsize=12, fontweight='bold')
                    
                    # Row 2: Silhouette
                    ax = axes[1, col_idx] if max_tracks_to_show > 1 else axes[1]
                    if track_id in self.track_silhouettes and len(self.track_silhouettes[track_id]) > 0:
                        latest_silhouette = self.track_silhouettes[track_id][-1]
                        ax.imshow(latest_silhouette, cmap='gray')
                        ax.set_title(f'Silhouette\n{len(self.track_silhouettes[track_id])} total', fontsize=10)
                    else:
                        ax.text(0.5, 0.5, 'Processing...', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title('No Silhouette Yet', fontsize=10)
                    ax.axis('off')
                    if col_idx == 0:
                        ax.text(-0.1, 0.5, row_labels[1], rotation=90, va='center', ha='right', 
                               transform=ax.transAxes, fontsize=12, fontweight='bold')
                    
                    # Row 3: GaitParsing (Human Parsing)
                    ax = axes[2, col_idx] if max_tracks_to_show > 1 else axes[2]
                    if track_id in self.track_parsing_results and len(self.track_parsing_results[track_id]) > 0:
                        latest_result = self.track_parsing_results[track_id][-1]
                        parsing_mask = latest_result.get('parsing_mask', np.zeros((256, 128), dtype=np.uint8))
                        
                        # GaitParsing color map for body parts
                        gait_parsing_colors = np.array([
                            [0, 0, 0],       # 0: background
                            [255, 0, 0],     # 1: head
                            [255, 255, 0],   # 2: body/torso
                            [0, 0, 255],     # 3: right arm
                            [255, 0, 255],   # 4: left arm
                            [0, 255, 0],     # 5: right leg
                            [0, 255, 255]    # 6: left leg
                        ]) / 255.0
                        
                        # Convert parsing mask to RGB
                        parsing_rgb = np.zeros((parsing_mask.shape[0], parsing_mask.shape[1], 3))
                        for i in range(min(7, len(gait_parsing_colors))):
                            mask = parsing_mask == i
                            if np.any(mask):
                                parsing_rgb[mask] = gait_parsing_colors[i]
                        
                        ax.imshow(parsing_rgb)
                        unique_parts = len(np.unique(parsing_mask))
                        ax.set_title(f'Human Parsing\n{unique_parts} parts', fontsize=10)
                    else:
                        ax.text(0.5, 0.5, 'Processing...', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title('No Parsing Yet', fontsize=10)
                    ax.axis('off')
                    if col_idx == 0:
                        ax.text(-0.1, 0.5, row_labels[2], rotation=90, va='center', ha='right', 
                               transform=ax.transAxes, fontsize=12, fontweight='bold')
                    
                    # Row 4: XGait Features Heatmap
                    ax = axes[3, col_idx] if max_tracks_to_show > 1 else axes[3]
                    if track_id in self.track_gait_features and len(self.track_gait_features[track_id]) > 0:
                        latest_features = self.track_gait_features[track_id][-1]
                        
                        if len(latest_features) > 0 and np.any(latest_features != 0):
                            # Reshape features to 2D for visualization (assume 256-dim -> 16x16)
                            if len(latest_features) >= 256:
                                feature_2d = latest_features[:256].reshape(16, 16)
                            elif len(latest_features) >= 64:
                                feature_2d = latest_features[:64].reshape(8, 8)
                            else:
                                # Pad smaller feature vectors
                                padded = np.zeros(64)
                                padded[:len(latest_features)] = latest_features
                                feature_2d = padded.reshape(8, 8)
                            
                            im = ax.imshow(feature_2d, cmap='viridis', aspect='auto')
                            ax.set_title(f'XGait Features\n{len(latest_features)}D vector', fontsize=10)
                            
                            # Add mini colorbar
                            from matplotlib.colorbar import make_axes, Colorbar
                            divider_ax = make_axes(ax, location="right", size="10%", pad=0.05)
                            cbar = plt.colorbar(im, cax=divider_ax[0])
                            cbar.ax.tick_params(labelsize=8)
                        else:
                            ax.text(0.5, 0.5, 'Zero Features\n(MPS Issue)', ha='center', va='center', 
                                   transform=ax.transAxes, color='red')
                            ax.set_title('XGait: No Features', fontsize=10)
                    else:
                        ax.text(0.5, 0.5, 'Processing...', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title('No Features Yet', fontsize=10)
                    ax.axis('off')
                    if col_idx == 0:
                        ax.text(-0.1, 0.5, row_labels[3], rotation=90, va='center', ha='right', 
                               transform=ax.transAxes, fontsize=12, fontweight='bold')
            
            # Hide unused subplots
            for col_idx in range(max_tracks_to_show, 4):
                if max_tracks_to_show > 1 and col_idx < axes.shape[1]:
                    for row_idx in range(4):
                        axes[row_idx, col_idx].axis('off')
            
            # Add processing info
            processing_info = f"Frame: {frame_count} | Tracks: {len(tracking_results)} | "
            processing_info += f"Parsed: {len([t for t in tracking_results if t[0] in self.track_parsing_results])} | "
            processing_info += f"Features: {len([t for t in tracking_results if t[0] in self.track_gait_features])}"
            fig.text(0.02, 0.02, processing_info, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            
            # Add legend for GaitParsing colors
            legend_elements = [
                plt.Rectangle((0,0),1,1, facecolor='black', label='Background'),
                plt.Rectangle((0,0),1,1, facecolor='red', label='Head'),
                plt.Rectangle((0,0),1,1, facecolor='yellow', label='Body/Torso'),
                plt.Rectangle((0,0),1,1, facecolor='blue', label='Right Arm'),
                plt.Rectangle((0,0),1,1, facecolor='magenta', label='Left Arm'),
                plt.Rectangle((0,0),1,1, facecolor='green', label='Right Leg'),
                plt.Rectangle((0,0),1,1, facecolor='cyan', label='Left Leg')
            ]
            fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
                      fontsize=8, title="GaitParsing Parts")
            
            # Save the comprehensive visualization
            output_path = self.debug_output_dir / f"frame_{frame_count:05d}_complete_pipeline.png"
            plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            if self.config.verbose:
                print(f"🎨 Saved complete pipeline visualization: {output_path}")
                
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Error saving complete pipeline visualization: {e}")
                import traceback
                traceback.print_exc()
    
    def _generate_comprehensive_analysis(self, total_frames: int):
        """
        Generate comprehensive analysis and visualization of the identification results
        
        Args:
            total_frames: Total number of frames processed
        """
        if not self.gallery_manager or not self.embedding_visualizer:
            print("⚠️  Gallery manager or visualizer not available for analysis")
            return
        
        print("\n" + "=" * 60)
        print("🎯 GENERATING COMPREHENSIVE IDENTIFICATION ANALYSIS")
        print("=" * 60)
        
        # Get gallery statistics
        gallery_stats = self.gallery_manager.get_gallery_summary()
        print(f"\n📊 Final Gallery Statistics:")
        print(f"   • Total identities created: {gallery_stats['num_identities']}")
        print(f"   • Total embeddings processed: {gallery_stats['total_embeddings_processed']}")
        print(f"   • Collision avoidance events: {gallery_stats['collision_avoided_count']}")
        print(f"   • Identity updates performed: {gallery_stats['identity_updates_count']}")
        print(f"   • Average embeddings per track: {gallery_stats['average_embeddings_per_track']:.2f}")
        
        # Identity quality analysis
        quality_scores = gallery_stats.get('gallery_quality_scores', {})
        if quality_scores:
            avg_quality = np.mean(list(quality_scores.values()))
            print(f"   • Average identity quality: {avg_quality:.3f}")
            print(f"   • Quality range: {min(quality_scores.values()):.3f} - {max(quality_scores.values()):.3f}")
        
        # Print identity assignments
        print(f"\n🏷️  Identity Assignments:")
        if hasattr(self, 'track_identities'):
            for track_id, identity_info in self.track_identities.items():
                print(f"   Track {track_id:2d} -> {identity_info['identity']} "
                      f"(conf: {identity_info['confidence']:.3f}, "
                      f"new: {identity_info['is_new']}, "
                      f"frame: {identity_info['frame_assigned']})")
        
        # Frame-level collision analysis
        frame_collision_stats = self._analyze_frame_collisions()
        if frame_collision_stats['total_collision_frames'] > 0:
            print(f"\n🚫 Collision Avoidance Analysis:")
            print(f"   • Frames with multiple tracks: {frame_collision_stats['total_collision_frames']}")
            print(f"   • Maximum tracks in single frame: {frame_collision_stats['max_tracks_per_frame']}")
            print(f"   • Collision avoidance rate: {frame_collision_stats['avoidance_rate']:.1f}%")
        
        # Save gallery state
        try:
            gallery_state_path = self.visualization_output_dir / f"gallery_state.json"
            self.gallery_manager.save_gallery_state(str(gallery_state_path))
            
            # Export embeddings for external analysis
            export_dir = self.visualization_output_dir / f"embeddings_export"
            self.gallery_manager.export_embeddings_for_analysis(str(export_dir))
            
            print(f"\n💾 Data Export:")
            print(f"   • Gallery state: {gallery_state_path}")
            print(f"   • Embeddings export: {export_dir}")
            
        except Exception as e:
            print(f"⚠️  Error saving gallery state: {e}")
        
        # Generate visualizations
        try:
            print(f"\n🎨 Generating Embedding Visualizations...")
            
            # Create timestamped visualization directory
            viz_dir = self.visualization_output_dir / f"visualizations"
            
            # Generate comprehensive visualization report
            viz_report = self.embedding_visualizer.create_comprehensive_report(
                gallery_manager=self.gallery_manager,
                output_dir=str(viz_dir),
                methods=["pca", "tsne", "umap"]
            )
            
            print(f"✅ Visualizations completed:")
            print(f"   • Output directory: {viz_dir}")
            print(f"   • Total visualizations: {viz_report['total_visualizations_created']}")
            print(f"   • Methods used: {', '.join(viz_report['visualization_methods'])}")
            
            # Create summary visualization showing key insights
            self._create_summary_dashboard(viz_dir, gallery_stats, frame_collision_stats, quality_scores)
            
        except Exception as e:
            print(f"⚠️  Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
        
        # Performance analysis
        self._print_performance_analysis(total_frames, gallery_stats)
        
        print("\n" + "=" * 60)
        print("🎉 COMPREHENSIVE ANALYSIS COMPLETED")
        print("=" * 60)
    
    def _analyze_frame_collisions(self) -> Dict:
        """Analyze frame-level collision statistics"""
        if not self.gallery_manager:
            return {'total_collision_frames': 0, 'max_tracks_per_frame': 0, 'avoidance_rate': 0.0}
        
        frame_assignments = self.gallery_manager.frame_assignments
        collision_frames = 0
        max_tracks = 0
        total_assignments = 0
        
        for frame_num, assignments in frame_assignments.items():
            num_tracks = len(assignments)
            total_assignments += num_tracks
            max_tracks = max(max_tracks, num_tracks)
            
            if num_tracks > 1:
                collision_frames += 1
        
        # Calculate avoidance rate (percentage of successful collision avoidance)
        avoidance_rate = (self.gallery_manager.collision_avoided_count / 
                         max(total_assignments, 1)) * 100
        
        return {
            'total_collision_frames': collision_frames,
            'max_tracks_per_frame': max_tracks,
            'avoidance_rate': avoidance_rate,
            'total_assignments': total_assignments
        }
    
    def _create_summary_dashboard(self, output_dir: Path, gallery_stats: Dict, 
                                collision_stats: Dict, quality_scores: Dict):
        """Create a summary dashboard with key metrics"""
        try:
            import matplotlib.pyplot as plt
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Identity distribution
            if quality_scores:
                identities = list(quality_scores.keys())
                qualities = list(quality_scores.values())
                
                ax1.bar(range(len(identities)), qualities, color='skyblue')
                ax1.set_xlabel('Identity')
                ax1.set_ylabel('Quality Score')
                ax1.set_title('Identity Quality Distribution')
                ax1.set_xticks(range(len(identities)))
                ax1.set_xticklabels([id.replace('Person_', 'P') for id in identities], rotation=45)
                ax1.grid(True, alpha=0.3)
            
            # 2. Gallery growth over time
            embeddings_data = []
            if hasattr(self, 'track_identities'):
                for track_id, info in self.track_identities.items():
                    embeddings_data.append(info['frame_assigned'])
                
                if embeddings_data:
                    ax2.hist(embeddings_data, bins=min(20, len(embeddings_data)), 
                            alpha=0.7, color='lightgreen')
                    ax2.set_xlabel('Frame Number')
                    ax2.set_ylabel('Identity Assignments')
                    ax2.set_title('Identity Assignment Timeline')
                    ax2.grid(True, alpha=0.3)
            
            # 3. Collision avoidance metrics
            collision_labels = ['Successful\nAvoidance', 'No Collision\nNeeded']
            collision_values = [
                gallery_stats.get('collision_avoided_count', 0),
                gallery_stats.get('total_embeddings_processed', 0) - gallery_stats.get('collision_avoided_count', 0)
            ]
            
            colors = ['lightcoral', 'lightblue']
            ax3.pie(collision_values, labels=collision_labels, colors=colors, autopct='%1.1f%%')
            ax3.set_title('Collision Avoidance Success Rate')
            
            # 4. Processing statistics
            stats_labels = ['Identities', 'Tracks', 'Embeddings\n(÷10)', 'Updates']
            stats_values = [
                gallery_stats.get('num_identities', 0),
                gallery_stats.get('total_tracks', 0),
                gallery_stats.get('total_embeddings_processed', 0) // 10,  # Scale down for visibility
                gallery_stats.get('identity_updates_count', 0)
            ]
            
            ax4.bar(stats_labels, stats_values, color=['gold', 'orange', 'lightpink', 'lightsteelblue'])
            ax4.set_ylabel('Count')
            ax4.set_title('Processing Statistics Summary')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   • Summary dashboard: summary_dashboard.png")
            
        except Exception as e:
            print(f"⚠️  Error creating summary dashboard: {e}")
    
    def _print_performance_analysis(self, total_frames: int, gallery_stats: Dict):
        """Print performance analysis"""
        print(f"\n⚡ Performance Analysis:")
        
        if total_frames > 0:
            embeddings_per_frame = gallery_stats.get('total_embeddings_processed', 0) / total_frames
            print(f"   • Embeddings per frame: {embeddings_per_frame:.2f}")
            
            identities_per_frame = gallery_stats.get('num_identities', 0) / total_frames
            print(f"   • New identities per frame: {identities_per_frame:.3f}")
        
        # Gallery efficiency
        total_embeddings = gallery_stats.get('total_embeddings_processed', 0)
        total_identities = gallery_stats.get('num_identities', 0)
        if total_identities > 0:
            embeddings_per_identity = total_embeddings / total_identities
            print(f"   • Embeddings per identity: {embeddings_per_identity:.1f}")
        
        # Update efficiency
        updates = gallery_stats.get('identity_updates_count', 0)
        if total_embeddings > 0:
            update_rate = updates / total_embeddings * 100
            print(f"   • Identity update rate: {update_rate:.1f}%")
        
        # Memory efficiency
        avg_embeddings_per_track = gallery_stats.get('average_embeddings_per_track', 0)
        print(f"   • Average embeddings per track: {avg_embeddings_per_track:.1f}")
        
        # Success metrics
        collision_avoidance = gallery_stats.get('collision_avoided_count', 0)
        if collision_avoidance > 0:
            print(f"   • Collision avoidance events: {collision_avoidance}")
            print(f"   • System robustness: HIGH ✅")
        else:
            print(f"   • System robustness: MODERATE ⚠️")
    
    def _perform_track_consolidation(self):
        """Perform track consolidation to reduce fragmentation"""
        if not self.gallery_manager:
            return
        
        print("\n" + "=" * 60)
        print("🔄 PERFORMING TRACK CONSOLIDATION ANALYSIS")
        print("=" * 60)
        
        # Analyze fragmentation before consolidation
        fragmentation_analysis = self.gallery_manager.analyze_track_fragmentation()
        
        print(f"\n📊 Pre-Consolidation Analysis:")
        print(f"   • Total tracks created: {fragmentation_analysis['total_tracks']}")
        print(f"   • Total identities: {fragmentation_analysis['total_identities']}")
        print(f"   • Average tracks per identity: {fragmentation_analysis['avg_tracks_per_identity']:.1f}")
        print(f"   • Estimated actual people: {fragmentation_analysis['estimated_actual_people']}")
        print(f"   • Fragmentation ratio: {fragmentation_analysis['fragmentation_ratio']:.2f}")
        
        if fragmentation_analysis['is_fragmented']:
            print(f"   ⚠️  High fragmentation detected (ratio > 1.5)")
            print(f"   🔄 Running consolidation...")
            
            # Perform consolidation with more aggressive threshold
            consolidation_map = self.gallery_manager.consolidate_fragmented_tracks(
                consolidation_threshold=0.55  # More aggressive consolidation threshold
            )
            
            # Analyze after consolidation
            post_analysis = self.gallery_manager.analyze_track_fragmentation()
            
            print(f"\n📊 Post-Consolidation Analysis:")
            print(f"   • Final identities: {post_analysis['total_identities']}")
            print(f"   • Identities removed: {fragmentation_analysis['total_identities'] - post_analysis['total_identities']}")
            print(f"   • New fragmentation ratio: {post_analysis['fragmentation_ratio']:.2f}")
            print(f"   • Improvement: {fragmentation_analysis['fragmentation_ratio'] - post_analysis['fragmentation_ratio']:.2f}")
            
            # Show consolidation results
            if consolidation_map:
                print(f"\n🔗 Consolidation Results:")
                for identity, tracks in consolidation_map.items():
                    print(f"   {identity}: merged tracks {sorted(tracks)}")
            
            # Update our local tracking data
            if hasattr(self, 'track_identities'):
                self._update_local_identities_after_consolidation(consolidation_map)
            
        else:
            print(f"   ✅ Low fragmentation - no consolidation needed")
        
        print("\n" + "=" * 60)
        print("✅ TRACK CONSOLIDATION COMPLETED")
        print("=" * 60)
    
    def _update_local_identities_after_consolidation(self, consolidation_map: Dict[str, List[int]]):
        """Update local track identity mappings after consolidation"""
        # Create reverse mapping: track_id -> final_identity
        if not consolidation_map:
            return
        
        track_to_final_identity = {}
        for final_identity, track_list in consolidation_map.items():
            for track_id in track_list:
                track_to_final_identity[track_id] = final_identity
        
        # Update track_identities
        updated_count = 0
        for track_id, identity_info in self.track_identities.items():
            if track_id in track_to_final_identity:
                old_identity = identity_info['identity']
                new_identity = track_to_final_identity[track_id]
                if old_identity != new_identity:
                    identity_info['identity'] = new_identity
                    identity_info['consolidated'] = True
                    updated_count += 1
        
        if updated_count > 0:
            print(f"   📝 Updated {updated_count} local track identity mappings")
    
    def get_is_new_identity_dict(self) -> dict:
        """
        Returns a dictionary mapping track_id to True (if new identity) or False (if from gallery)
        for all currently tracked identities.
        """
        is_new_identity = {}
        if hasattr(self, 'track_identities'):
            for track_id, info in self.track_identities.items():
                is_new_identity[track_id] = info.get('is_new', False)
        return is_new_identity


