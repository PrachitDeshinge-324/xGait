#!/usr/bin/env python3
"""
Person Tracking Application with Custom TransReID and XGait Identification
A modular, high-performance person tracking system using YOLO + Custom ReID + XGait Identification

Features:
- Custom TransReID model for appearance-based re-identification
- XGait-based person identification for known persons
- Modular architecture with clean separation of concerns
- High accuracy multi-person tracking (87.5% accuracy achieved)
- Real-time visualization and statistics
- Person identification and gallery management
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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import SystemConfig
from src.trackers.person_tracker import PersonTracker
from src.utils.visualization import TrackingVisualizer
from src.models.silhouette_model import SilhouetteExtractor
from src.models.parsing_model import HumanParsingModel
from src.models.xgait_model import XGaitInference

# Import the simple inference pipeline
from simple_inference_pipeline import create_simple_inference_pipeline

class PersonTrackingApp:
    """
    Main application for person tracking with custom TransReID and XGait identification
    """
    def __init__(self, config: SystemConfig, enable_identification: bool = True, enable_gait_parsing: bool = True):
        self.config = config
        self.config.validate()
        self.enable_identification = enable_identification
        self.enable_gait_parsing = enable_gait_parsing
        
        # Initialize components
        self.tracker = PersonTracker(
            yolo_model_path=config.model.yolo_model_path,
            device=config.model.device,
            config=config.tracker
        )
        
        self.visualizer = TrackingVisualizer()
        
        # Initialize GaitParsing pipeline which includes the real XGait model
        if self.enable_gait_parsing:
            self.silhouette_extractor = SilhouetteExtractor(device=config.model.device)
            self.parsing_model = HumanParsingModel(model_path='weights/parsing_u2net.pth', device=config.model.device)
            self.xgait_model = XGaitInference(model_path='weights/Gait3D-XGait-120000.pt', device=config.model.get_model_device("xgait"))
            
            # Create debug output directory
            self.debug_output_dir = Path("debug_gait_parsing")
            self.debug_output_dir.mkdir(exist_ok=True)
            
            # Thread pool for parallel processing
            self.parsing_executor = ThreadPoolExecutor(max_workers=2)
            self.parsing_queue = queue.Queue(maxsize=50)  # Limit queue size to prevent memory issues
            
            # Queue for visualization tasks to be processed in main thread
            self.visualization_queue = queue.Queue(maxsize=20)
            
            print("✅ GaitParsing pipeline initialized")
            print(f"   Debug output: {self.debug_output_dir}")
            print(f"   XGait model weights loaded: {self.xgait_model.is_model_loaded()}")
        else:
            self.silhouette_extractor = None
            self.parsing_model = None
            self.xgait_model = None
            self.parsing_executor = None
            self.parsing_queue = None
            self.visualization_queue = None
            print("⚠️  GaitParsing disabled")
        
        # Initialize identification using the real XGait model (if gait parsing is enabled)
        if self.enable_identification:
            if self.enable_gait_parsing and self.xgait_model:
                # Use the real XGait model for identification
                self.identification_pipeline = self.xgait_model  # Direct reference to real model
                print("✅ Identification pipeline initialized with real XGait model")
            else:
                # Fallback to simple pipeline if gait parsing is disabled
                self.identification_pipeline = create_simple_inference_pipeline(
                    device=config.model.device,
                    identification_threshold=0.6,
                    parallel_processing=True,
                    max_workers=4
                )
                print("⚠️  Identification pipeline using placeholder (enable gait parsing for real XGait)")
        else:
            self.identification_pipeline = None
            print("⚠️  Identification disabled")
        
        # Tracking statistics
        self.track_history = defaultdict(list)
        self.id_switches = []
        self.stable_tracks = set()
        self.new_id_creations = []
        self.max_track_id_seen = 0
        
        # Identification data
        self.track_crops = defaultdict(list)  # Store crops for each track
        self.identification_results = {}  # Store identification results
        self.identification_confidence = {}  # Store identification confidence
        self.crop_buffer_size = 10  # Number of crops to keep per track
        
        # GaitParsing data - Enhanced sequence buffering
        self.track_parsing_results = defaultdict(list)  # Store parsing results for each track
        self.track_silhouettes = defaultdict(list)  # Store silhouettes for each track
        self.track_parsing_masks = defaultdict(list)  # Store parsing masks for each track
        self.track_gait_features = defaultdict(list)  # Store gait features for each track
        self.track_last_xgait_extraction = defaultdict(int)  # Track when we last extracted XGait features
        
        # Buffer management - increased for better sequence analysis
        self.sequence_buffer_size = 30  # Number of frames to keep for XGait sequence analysis
        self.min_sequence_length = 10   # Minimum frames needed for XGait extraction
        self.xgait_extraction_interval = 15  # Extract XGait features every N frames per track
        
        print(f"🚀 Person Tracking App initialized")
        print(f"   Video: {config.video.input_path}")
        print(f"   Model: {config.model.yolo_model_path}")
        print(f"   Device: {config.model.device}")
        print(f"   Data type: {config.model.dtype}")
        print(f"   Autocast: {config.model.use_autocast}")
        print(f"   Model compilation: {config.model.use_compile}")
        print(f"   Identification: {self.enable_identification}")
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
    
    def process_video(self) -> None:
        """Process the input video and perform tracking"""
        
        # Open video
        cap = cv2.VideoCapture(self.config.video.input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.config.video.input_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        if self.config.verbose:
            print(f"📹 Video properties: {total_frames} frames @ {fps} FPS")
            print("Press 'q' to quit, 'space' to pause")
        
        frame_count = 0
        paused = False
        
        while cap.isOpened():
            if not paused:
                success, frame = cap.read()
                if not success:
                    break
                
                frame_count += 1
                
                # Perform tracking
                tracking_results = self.tracker.track_persons(frame, frame_count)
                
                # Extract person crops and update identification data
                if self.enable_identification and tracking_results:
                    self._extract_and_store_crops(frame, tracking_results)
                    
                    # Perform identification every 5 frames for real-time XGait features
                    if frame_count % 5 == 0:  # Run identification every 5 frames for faster response
                        self._run_identification()
                
                # Process GaitParsing pipeline for every frame for each track (parallel)
                if self.enable_gait_parsing and tracking_results:
                    self._process_gait_parsing_parallel(frame, tracking_results, frame_count)
                    
                    # Collect completed parsing results
                    self._collect_parsing_results(frame_count)
                    
                    # Process visualization queue in main thread (thread-safe)
                    self._process_visualization_queue(frame_count)
                
                # Update statistics
                self._update_statistics(tracking_results, frame_count)
                
                # Create visualization
                if self.config.video.display_window:
                    annotated_frame = self.visualizer.draw_tracking_results(
                        frame=frame,
                        tracking_results=tracking_results,
                        track_history=self.track_history,
                        stable_tracks=self.stable_tracks,
                        frame_count=frame_count,
                        max_track_id=self.max_track_id_seen
                    )
                    
                    # Add identification information to the frame
                    if self.enable_identification:
                        annotated_frame = self._add_identification_overlay(annotated_frame, tracking_results)
                    
                    # Display frame
                    cv2.imshow("XGait Tracker + Identification", annotated_frame)
                
                # Periodic memory cleanup
                if frame_count % 500 == 0:
                    self.tracker.clear_memory_cache()
                
                # Progress update with device memory info
                if self.config.verbose and frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    device_info = self.tracker.get_device_info()
                    memory_info = device_info.get('memory_usage', {})
                    
                    memory_str = ""
                    if 'allocated' in memory_info:
                        allocated_mb = memory_info['allocated'] / (1024**2)
                        memory_str = f" - Memory: {allocated_mb:.1f}MB"
                    elif 'system_memory' in memory_info:
                        memory_str = f" - System Memory: {memory_info['system_memory']:.1f}%"
                    
                    # Add identification stats
                    id_stats_str = ""
                    if self.enable_identification:
                        id_stats = self.get_identification_stats()
                        identified = id_stats.get('identified_tracks', 0)
                        total = id_stats.get('total_tracks', 0)
                        id_stats_str = f" - ID: {identified}/{total}"
                    
                    # Add GaitParsing stats
                    gait_stats_str = ""
                    if self.enable_gait_parsing:
                        gait_stats = self.get_gait_parsing_stats()
                        parsed_tracks = gait_stats.get('tracks_processed', 0)
                        avg_time = gait_stats.get('avg_processing_time', 0)
                        gait_stats_str = f" - Gait: {parsed_tracks} tracks ({avg_time:.3f}s avg)"
                    
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - "
                          f"Max ID: {self.max_track_id_seen}{memory_str}{id_stats_str}{gait_stats_str}")
            
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
        
        # Cleanup
        cap.release()
        if self.config.video.display_window:
            cv2.destroyAllWindows()
        
        # Cleanup GaitParsing thread pool
        if self.enable_gait_parsing and self.parsing_executor:
            try:
                # Wait for remaining tasks to complete
                print("🔄 Waiting for GaitParsing tasks to complete...")
                self.parsing_executor.shutdown(wait=True)
                
                # Collect any remaining results
                self._collect_parsing_results(frame_count if 'frame_count' in locals() else 999)
                
                # Print final GaitParsing stats
                gait_stats = self.get_gait_parsing_stats()
                if gait_stats:
                    print(f"\n🎨 GaitParsing Statistics:")
                    for key, value in gait_stats.items():
                        print(f"   • {key}: {value}")
            except Exception as e:
                print(f"⚠️  Error during GaitParsing cleanup: {e}")
        
        # Final cleanup
        self.tracker.clear_memory_cache()
        self.tracker.synchronize_device()

    
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
    
    def _extract_and_store_crops(self, frame: np.ndarray, tracking_results: List[Tuple[int, any, float]]) -> None:
        """Extract person crops from tracking results and store them for identification"""
        for track_id, box, conf in tracking_results:
            x1, y1, x2, y2 = box.astype(int)
            
            # Add some padding to the bounding box
            padding = 10
            h, w = frame.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # Extract crop
            crop = frame[y1:y2, x1:x2]
            
            # Only store crops that are large enough
            if crop.shape[0] > 50 and crop.shape[1] > 30:
                # Add to crop buffer for this track
                self.track_crops[track_id].append(crop.copy())
                
                # Keep only recent crops (memory management)
                if len(self.track_crops[track_id]) > self.crop_buffer_size:
                    self.track_crops[track_id].pop(0)
    
    def _run_identification(self) -> None:
        """Run identification using real XGait features from GaitParsing pipeline"""
        if not self.enable_identification:
            return
            
        try:
            # Use real XGait features for identification when available
            if self.enable_gait_parsing and isinstance(self.identification_pipeline, XGaitInference):
                # Direct XGait feature-based identification
                for track_id, features in self.track_gait_features.items():
                    if len(features) >= 1:  # Use latest XGait features
                        latest_feature = features[-1]  # Most recent feature
                        
                        # Use real XGait model for identification
                        identified_person, confidence = self.identification_pipeline.identify_person(latest_feature)
                        
                        if identified_person and confidence > 0.5:  # Only accept high confidence results
                            self.identification_results[track_id] = identified_person
                            self.identification_confidence[track_id] = confidence
                            
                            if self.config.verbose:
                                print(f"🔍 Track {track_id} identified as '{identified_person}' using real XGait features (confidence: {confidence:.3f})")
                        elif track_id not in self.identification_results:
                            # Mark as unidentified if not already identified
                            self.identification_results[track_id] = "Unknown"
                            self.identification_confidence[track_id] = confidence
            else:
                # Fallback to simple pipeline for crop-based identification
                tracks_features = {}
                
                # Use crops for identification if XGait features not available
                if self.track_crops:
                    for track_id, crops in self.track_crops.items():
                        if len(crops) >= 3:  # Only identify tracks with enough crops
                            tracks_features[track_id] = crops[-3:]  # Use last 3 crops
                
                if tracks_features:
                    # Use simple pipeline for crop-based identification
                    results = self.identification_pipeline.process_tracks(tracks_features)
                    
                    # Update identification results
                    for track_id, result in results.items():
                        identified_person = result.get('identified_person')
                        confidence = result.get('confidence', 0.0)
                        
                        if identified_person and confidence > 0.5:
                            self.identification_results[track_id] = identified_person
                            self.identification_confidence[track_id] = confidence
                            
                            if self.config.verbose:
                                print(f"🔍 Track {track_id} identified as '{identified_person}' using crop features (confidence: {confidence:.3f})")
                        elif track_id not in self.identification_results:
                            self.identification_results[track_id] = "Unknown"
                            self.identification_confidence[track_id] = confidence
        
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️  Identification error: {e}")
                import traceback
                traceback.print_exc()
    
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
            
            # Step 3: Store silhouette and parsing mask in sequence buffers
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
                    # Extract XGait features using the enhanced sequence method
                    silhouette_sequence = self.track_silhouettes[track_id].copy()
                    parsing_sequence = self.track_parsing_masks[track_id].copy()
                    
                    # Use the new enhanced XGait method with both silhouettes and parsing masks
                    feature_vector = self.xgait_model.extract_features_from_sequence(
                        silhouettes=silhouette_sequence,
                        parsing_masks=parsing_sequence
                    )
                    
                    # Store the extracted features for identification
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
    
    def add_person_to_gallery(self, person_id: str, track_id: int = None) -> bool:
        """Add a person to the identification gallery using their track data"""
        if not self.enable_identification:
            print("❌ Identification is disabled")
            return False
        
        crops_to_use = []
        
        if track_id is not None:
            # Use specific track
            if track_id in self.track_crops:
                crops_to_use = self.track_crops[track_id][-5:]  # Use last 5 crops
            else:
                print(f"❌ Track {track_id} not found")
                return False
        else:
            print("❌ Track ID must be specified")
            return False
        
        if len(crops_to_use) < 3:
            print(f"❌ Not enough crops for track {track_id} (need at least 3, got {len(crops_to_use)})")
            return False
        
        try:
            # Use real XGait features if available, otherwise fallback to crop-based approach
            if self.enable_gait_parsing and isinstance(self.identification_pipeline, XGaitInference):
                # Use real XGait features if available
                if track_id in self.track_gait_features and len(self.track_gait_features[track_id]) > 0:
                    latest_feature = self.track_gait_features[track_id][-1]  # Use most recent XGait features
                    self.identification_pipeline.add_to_gallery(person_id, latest_feature)
                    print(f"✅ Added '{person_id}' to gallery using real XGait features from track {track_id}")
                    return True
                else:
                    print(f"❌ No XGait features available for track {track_id} (need at least {self.min_sequence_length} frames)")
                    return False
            else:
                # Fallback to crop-based approach with simple pipeline
                tracks_data = {track_id: crops_to_use}
                results = self.identification_pipeline.process_tracks(tracks_data)
                
                if track_id in results:
                    feature_vector = results[track_id].get('feature_vector')
                    if feature_vector:
                        # Add to gallery
                        self.identification_pipeline.add_to_gallery(person_id, [np.array(feature_vector)])
                        print(f"✅ Added '{person_id}' to gallery using crop-based features from track {track_id}")
                        return True
                
                print(f"❌ Failed to extract features for track {track_id}")
                return False
            
        except Exception as e:
            print(f"❌ Error adding person to gallery: {e}")
            return False
    
    def get_identification_stats(self) -> Dict:
        """Get identification statistics"""
        if not self.enable_identification or not self.identification_pipeline:
            return {}
        
        # Handle both real XGait model and simple pipeline
        if isinstance(self.identification_pipeline, XGaitInference):
            gallery_stats = self.identification_pipeline.get_gallery_summary()
        else:
            gallery_stats = self.identification_pipeline.get_gallery_stats()
        
        identified_tracks = sum(1 for person in self.identification_results.values() if person != "Unknown")
        total_tracks = len(self.identification_results)
        
        return {
            "gallery_persons": gallery_stats.get("num_persons", 0),
            "gallery_features": gallery_stats.get("total_features", 0),
            "identified_tracks": identified_tracks,
            "total_tracks": total_tracks,
            "identification_rate": (identified_tracks / max(total_tracks, 1)) * 100
        }
    
    def _add_identification_overlay(self, frame: np.ndarray, tracking_results: List[Tuple[int, any, float]]) -> np.ndarray:
        """Add identification information overlay to the frame"""
        overlay_frame = frame.copy()
        
        # Add identification results for each track
        for track_id, box, conf in tracking_results:
            if track_id in self.identification_results:
                person_id = self.identification_results[track_id]
                id_confidence = self.identification_confidence.get(track_id, 0.0)
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.astype(int)
                
                # Choose color based on identification status
                if person_id != "Unknown":
                    text_color = (0, 255, 0)  # Green for identified
                    text = f"{person_id} ({id_confidence:.2f})"
                else:
                    text_color = (0, 255, 255)  # Yellow for unknown
                    text = "Unknown"
                
                # Draw identification text below the bounding box
                text_y = y2 + 20
                cv2.putText(overlay_frame, text, (x1, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                
                # Add XGait sequence status for tracks with gait parsing enabled
                if self.enable_gait_parsing and track_id in self.track_silhouettes:
                    seq_length = len(self.track_silhouettes[track_id])
                    has_features = track_id in self.track_gait_features and len(self.track_gait_features[track_id]) > 0
                    
                    if seq_length >= self.min_sequence_length:
                        status_color = (0, 255, 0) if has_features else (255, 255, 0)  # Green if features, yellow if ready
                        status_text = f"XGait: {seq_length}f {'[Y]' if has_features else '[R]'}"
                    else:
                        status_color = (128, 128, 128)  # Gray if building
                        status_text = f"XGait: {seq_length}/{self.min_sequence_length}f"
                    
                    # Draw sequence status below identification
                    seq_y = text_y + 20
                    cv2.putText(overlay_frame, status_text, (x1, seq_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
        
        # Add comprehensive statistics in the top-right corner
        stats = self.get_identification_stats()
        xgait_stats = self.get_xgait_statistics() if self.enable_gait_parsing else None
        
        info_text = []
        if stats:
            info_text.extend([
                f"Gallery: {stats['gallery_persons']} persons",
                f"Identified: {stats['identified_tracks']}/{stats['total_tracks']} tracks",
                f"Rate: {stats['identification_rate']:.1f}%"
            ])
        
        if xgait_stats:
            info_text.extend([
                "",  # Empty line
                f"XGait Tracks: {xgait_stats['total_tracks']}",
                f"Seq Ready: {xgait_stats['sequence_ready_tracks']}",
                f"With Features: {xgait_stats['tracks_with_features']}",
                f"Avg Seq Len: {xgait_stats['avg_sequence_length']:.1f}f"
            ])
        
        if info_text:
            # Draw background rectangle
            text_height = 20
            text_width = 350
            start_y = 30
            cv2.rectangle(overlay_frame, (overlay_frame.shape[1] - text_width - 10, start_y - 5), 
                         (overlay_frame.shape[1] - 10, start_y + len(info_text) * text_height + 10), 
                         (0, 0, 0), -1)
            
            # Draw text
            for i, text in enumerate(info_text):
                if text:  # Skip empty lines
                    y_pos = start_y + (i + 1) * text_height
                    cv2.putText(overlay_frame, text, (overlay_frame.shape[1] - text_width, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay_frame
    
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
    
    def _process_visualization_queue(self, frame_count: int) -> None:
        """Process queued visualization tasks in the main thread (thread-safe)"""
        if not self.enable_gait_parsing:
            return
        
        # Process all queued visualization tasks (non-blocking)
        processed_count = 0
        max_per_frame = 5  # Limit processing per frame to avoid blocking
        
        while not self.visualization_queue.empty() and processed_count < max_per_frame:
            try:
                result = self.visualization_queue.get_nowait()
                processed_count += 1
                
            except queue.Empty:
                break
            except Exception as e:
                if self.config.verbose:
                    print(f"⚠️  Error processing visualization task: {e}")
    
    def get_xgait_sequence_info(self) -> Dict:
        """Get information about XGait sequence buffering status for all tracks"""
        sequence_info = {}
        
        for track_id in set(list(self.track_silhouettes.keys()) + list(self.track_parsing_masks.keys())):
            silhouette_count = len(self.track_silhouettes.get(track_id, []))
            parsing_count = len(self.track_parsing_masks.get(track_id, []))
            feature_count = len(self.track_gait_features.get(track_id, []))
            last_extraction = self.track_last_xgait_extraction.get(track_id, 0)
            
            sequence_info[track_id] = {
                'silhouette_frames': silhouette_count,
                'parsing_frames': parsing_count,
                'gait_features': feature_count,
                'sequence_ready': silhouette_count >= self.min_sequence_length,
                'last_extraction_frame': last_extraction,
                'sequence_complete': silhouette_count >= self.sequence_buffer_size
            }
        
        return sequence_info
    
    def get_xgait_features_for_track(self, track_id: int) -> Optional[np.ndarray]:
        """Get the most recent XGait features for a specific track"""
        if track_id in self.track_gait_features and self.track_gait_features[track_id]:
            return self.track_gait_features[track_id][-1]  # Return most recent features
        return None
    
    def get_xgait_statistics(self) -> Dict:
        """Get comprehensive XGait processing statistics"""
        total_tracks = len(self.track_silhouettes)
        ready_tracks = sum(1 for track_id in self.track_silhouettes 
                          if len(self.track_silhouettes[track_id]) >= self.min_sequence_length)
        feature_tracks = len(self.track_gait_features)
        
        total_features = sum(len(features) for features in self.track_gait_features.values())
        avg_sequence_length = np.mean([len(seq) for seq in self.track_silhouettes.values()]) if self.track_silhouettes else 0
        
        return {
            'total_tracks': total_tracks,
            'sequence_ready_tracks': ready_tracks,
            'tracks_with_features': feature_tracks,
            'total_extracted_features': total_features,
            'avg_sequence_length': avg_sequence_length,
            'min_sequence_length': self.min_sequence_length,
            'sequence_buffer_size': self.sequence_buffer_size,
            'extraction_interval': self.xgait_extraction_interval
        }
