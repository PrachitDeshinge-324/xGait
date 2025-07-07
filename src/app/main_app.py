#!/usr/bin/env python3
"""
Main application class for person tracking with XGait feature extraction.
"""

import warnings
import os
import contextlib
import sys
import cv2
import numpy as np
import time
import queue
import threading
import logging
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Context manager to suppress stdout warnings
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config import SystemConfig, xgaitConfig
from src.trackers.person_tracker import PersonTracker
from src.utils.visualization import TrackingVisualizer, VideoWriter
from src.models.silhouette_model import SilhouetteExtractor
from src.models.parsing_model import HumanParsingModel
from src.models.xgait_model import create_xgait_inference
from src.utils.simple_identity_gallery import SimpleIdentityGallery
from src.utils.embedding_visualization import EmbeddingVisualizer
from src.utils.visual_track_reviewer import VisualTrackReviewer
from src.processing.video_processor import VideoProcessor
from src.processing.gait_processor import GaitProcessor
from src.processing.statistics_manager import StatisticsManager
from src.processing.identity_manager import IdentityManager
from src.processing.enhanced_identity_manager import EnhancedIdentityManager


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

        # Initialize core components
        self.tracker = PersonTracker(
            yolo_model_path=config.model.yolo_model_path,
            device=config.model.device,
            config=config.tracker
        )
        
        self.visualizer = TrackingVisualizer()
        self.video_writer = None
        
        # Initialize processing components
        self.video_processor = VideoProcessor(config)
        self.statistics_manager = StatisticsManager(config)
        
        # Use enhanced identity manager with configurable option
        use_enhanced_gallery = self.config.identity.use_enhanced_gallery
        if use_enhanced_gallery:
            self.identity_manager = EnhancedIdentityManager(config, use_enhanced_gallery=True)
            print("✅ Enhanced Identity Manager initialized")
        else:
            self.identity_manager = IdentityManager(config)
            print("✅ Standard Identity Manager initialized")
        
        # Initialize GaitParsing pipeline
        if self.enable_gait_parsing:
            self.gait_processor = GaitProcessor(config, self.identity_manager)
            print("✅ GaitParsing pipeline initialized")
            print(f"   Debug output: {self.gait_processor.debug_output_dir}")
            print(f"   Visualization output: {self.gait_processor.visualization_output_dir}")
        else:
            self.gait_processor = None
            print("⚠️  GaitParsing disabled")
        
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
    
    def __del__(self):
        """Destructor to ensure proper cleanup"""
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.gait_processor:
            self.gait_processor.cleanup()
        
        if self.video_processor:
            self.video_processor.cleanup()
    
    def process_video(self) -> None:
        """Process the input video and perform tracking"""
        # Initialize video processing
        cap, total_frames, fps, frame_width, frame_height = self.video_processor.initialize_video(self.config.video.input_path)
        
        # Initialize video writer if needed
        if self.config.video.save_annotated_video and self.config.video.output_video_path:
            self.video_writer = self.video_processor.initialize_video_writer(
                self.config.video.output_video_path,
                fps, frame_width, frame_height
            )
        
        # Load gallery state
        self.gallery_loaded = self.identity_manager.load_gallery()
        
        # Process video frames
        frame_count = 0
        paused = False
        pbar_total = self.config.video.max_frames if self.config.video.max_frames else total_frames
        start_time = time.time()
        last_frame_time = start_time
        fps_list = []
        
        # Clear processing data
        if self.gait_processor:
            self.gait_processor.clear_data()
        
        with tqdm(total=pbar_total, desc="Processing frames", unit="frame", 
                  disable=False, dynamic_ncols=True, smoothing=0.01, 
                  mininterval=0.01, maxinterval=0.1, file=sys.stdout, 
                  ascii=False, colour='green') as pbar:
            
            while cap.isOpened():
                # Only pause if display window is enabled and user pressed space
                should_pause = paused and self.config.video.display_window
                
                if not should_pause:
                    success, frame = cap.read()
                    if not success:
                        break
                    frame_count += 1
                    
                    # Check max_frames limit
                    if self.config.video.max_frames and frame_count > self.config.video.max_frames:
                        break
                    
                    # Perform tracking
                    tracking_results = self.tracker.track_persons(frame, frame_count)
                    
                    # Process gait parsing if enabled
                    if self.enable_gait_parsing and self.gait_processor and tracking_results:
                        self.gait_processor.process_frame(frame, tracking_results, frame_count)
                    
                    # Update statistics
                    self.statistics_manager.update_statistics(tracking_results, frame_count)
                    
                    # Handle identity assignment
                    frame_track_embeddings = {}
                    if self.gait_processor:
                        frame_track_embeddings = self.gait_processor.get_frame_track_embeddings(tracking_results)
                    
                    # Process identities
                    frame_assignments = self.identity_manager.assign_or_update_identities(
                        frame_track_embeddings, frame_count
                    )
                    
                    # Prepare visualization data
                    identification_results = {}
                    identification_confidence = {}
                    for track_id, person_name in frame_assignments.items():
                        identification_results[track_id] = person_name
                        identification_confidence[track_id] = 1.0
                    
                    gallery_stats = self.identity_manager.get_gallery_stats()
                    
                    # FPS calculation
                    now = time.time()
                    current_fps = 1.0 / max(now - last_frame_time, 1e-6)
                    last_frame_time = now
                    fps_list.append(current_fps)
                    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0.0
                    
                    # Create annotated frame
                    annotated_frame = self.visualizer.draw_tracking_results(
                        frame=frame,
                        tracking_results=tracking_results,
                        track_history=self.statistics_manager.track_history,
                        stable_tracks=self.statistics_manager.stable_tracks,
                        frame_count=frame_count,
                        max_track_id=self.statistics_manager.max_track_id_seen,
                        identification_results=identification_results,
                        identification_confidence=identification_confidence,
                        gallery_stats=gallery_stats,
                        identification_stats=None,
                        current_fps=current_fps,
                        avg_fps=avg_fps,
                        is_new_identity=self.identity_manager.get_is_new_identity_dict(),
                        gallery_loaded=self.gallery_loaded
                    )
                    
                    # Write video frame if enabled
                    if self.video_writer and self.video_writer.is_opened():
                        self.video_writer.write_frame(annotated_frame)
                    
                    # Display frame if enabled
                    if self.config.video.display_window:
                        cv2.imshow("Person Tracking & XGait Analysis", annotated_frame)
                    
                    # Memory management
                    if frame_count % 500 == 0:
                        self.tracker.clear_memory_cache()
                    
                    # Update progress bar
                    pbar.update(1)
                    
                    # Update progress display
                    if frame_count % 5 == 0:
                        pbar.refresh()
                        sys.stdout.flush()
                    
                    if frame_count % 10 == 0:
                        pbar.set_description(f"Processing frames (FPS: {current_fps:.1f})")
                        pbar.refresh()
                        sys.stdout.flush()
                
                # Handle display window and user input
                if self.config.video.display_window:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord(' '):
                        paused = not paused
                else:
                    time.sleep(0.001)
        
        # Cleanup video processing
        self.video_processor.cleanup_video(cap)
        
        # Finalize processing
        if self.gait_processor:
            self.gait_processor.finalize_processing(frame_count)
        
        # Run interactive track review if enabled
        if self.config.video.interactive_mode:
            self.run_interactive_track_review()
        
        # Save final results
        self.identity_manager.save_gallery()
        self.identity_manager.print_final_summary()
        
        # Generate visualizations
        self._generate_final_visualizations()
        
        # Print final statistics
        self._print_final_results(frame_count)
    
    def _generate_final_visualizations(self):
        """Generate final embedding visualizations"""
        print("[EmbeddingVisualizer] Visualizing all gallery and track embeddings...")
        
        all_embeddings = self.identity_manager.get_all_embeddings()
        embeddings_by_track = self.identity_manager.get_track_embeddings_by_track()
        
        visualizer = EmbeddingVisualizer()
        
        # Visualize gallery + track embeddings together
        for method in ["umap", "pca", "tsne"]:
            fig = visualizer.visualize_identity_gallery(
                all_embeddings=all_embeddings,
                method=method,
                save_path=f"visualization_analysis/embedding_{method}.png",
                show_labels=True,
                plot_type="2d"
            )
            if fig is None:
                print(f"[EmbeddingVisualizer] No gallery embeddings to visualize for {method}.")
        
        # Visualize track embeddings only (by track)
        for method in ["umap", "pca", "tsne"]:
            fig = visualizer.visualize_track_embeddings(
                embeddings_by_track=embeddings_by_track,
                method=method,
                save_path=f"visualization_analysis/track_embedding_{method}.png",
                show_labels=True,
                plot_type="2d"
            )
            if fig is None:
                print(f"[EmbeddingVisualizer] No track embeddings to visualize for {method}.")
    
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
            max_track_id=self.statistics_manager.max_track_id_seen,
            total_frames=total_frames,
            target_people=7  # Known from the video
        )
        
        # Print statistics
        self.statistics_manager.print_final_statistics()
        
        # Print gait parsing statistics if enabled
        if self.gait_processor:
            gait_stats = self.gait_processor.get_statistics()
            print(f"\n🚶 GaitParsing Statistics:")
            print(f"   • Tracks processed: {gait_stats.get('tracks_processed', 0)}")
            print(f"   • Total parsing results: {gait_stats.get('total_parsing_results', 0)}")
            print(f"   • Average processing time: {gait_stats.get('avg_processing_time', 0):.3f}s")
            if self.config.debug_mode:
                print(f"   • Debug images saved: {gait_stats.get('debug_images_saved', 0)}")
    
    def get_gait_parsing_stats(self) -> Dict:
        """Get GaitParsing statistics"""
        if not self.gait_processor:
            return {}
        return self.gait_processor.get_statistics()
    
    def run_interactive_track_review(self) -> None:
        """Run interactive track review for manual person identification"""
        if not self.config.video.interactive_mode:
            return
            
        print("\n" + "=" * 60)
        print("🎮 INTERACTIVE PERSON IDENTIFICATION")
        print("=" * 60)
        
        # Use existing in-memory data from identity manager instead of loading from files
        if not hasattr(self.identity_manager, 'track_embedding_buffer') or not self.identity_manager.track_embedding_buffer:
            print("❌ No track data available in memory. Make sure video processing completed.")
            return
        
        # Initialize reviewer but skip loading track data since we have it in memory
        reviewer = VisualTrackReviewer()
        
        # Set the track data directly from the identity manager
        reviewer.track_embedding_buffer = dict(self.identity_manager.track_embedding_buffer)
        reviewer.track_quality_buffer = dict(self.identity_manager.track_quality_buffer)
        reviewer.track_to_person = dict(getattr(self.identity_manager, 'track_to_person', {}))
        
        # Set crop and bbox data if available (for enhanced gallery)
        if hasattr(self.identity_manager, 'track_crop_buffer'):
            reviewer.track_crop_buffer = dict(self.identity_manager.track_crop_buffer)
        if hasattr(self.identity_manager, 'track_bbox_buffer'):
            reviewer.track_bbox_buffer = dict(self.identity_manager.track_bbox_buffer)
        if hasattr(self.identity_manager, 'track_parsing_buffer'):
            reviewer.track_parsing_buffer = dict(self.identity_manager.track_parsing_buffer)
        
        # Identify unassigned tracks
        reviewer.unassigned_tracks = set()
        for track_id in reviewer.track_embedding_buffer.keys():
            if track_id not in reviewer.track_to_person:
                reviewer.unassigned_tracks.add(track_id)
        
        print(f"📊 Using in-memory track data:")
        print(f"   Total tracks: {len(reviewer.track_embedding_buffer)}")
        print(f"   Assigned tracks: {len(reviewer.track_to_person)}")
        print(f"   Unassigned tracks: {len(reviewer.unassigned_tracks)}")
        
        if not reviewer.track_embedding_buffer:
            print("❌ No track embeddings available for review.")
            return
        
        # Show overview
        reviewer.show_all_tracks_overview()
        
        # Ask if user wants to proceed
        proceed = input("\n🚀 Start interactive person identification? (y/n, default=y): ").strip().lower()
        if proceed != 'n':
            # Review all tracks (both assigned and unassigned)
            assignments = reviewer.review_all_tracks()
            
            if assignments:
                print(f"\n💾 Applying {len(assignments)} person assignments to ENHANCED GALLERY...")
                
                # We can use the existing identity manager instead of creating a new one
                # since we're already in the app context
                
                # Create or update persons from assignments - Focus on Enhanced Gallery
                created_persons = []
                updated_persons = []
                for track_id, person_name in assignments.items():
                    # Get track data directly from identity manager
                    if track_id in self.identity_manager.track_embedding_buffer:
                        embeddings = self.identity_manager.track_embedding_buffer[track_id]
                        qualities = self.identity_manager.track_quality_buffer.get(track_id, [0.5] * len(embeddings))
                        
                        # Check if this is a reassignment of an existing track
                        previous_person = self.identity_manager.simple_gallery.track_to_person.get(track_id)
                        if previous_person and previous_person != person_name:
                            print(f"🔄 Reassigning track {track_id} from '{previous_person}' to '{person_name}'")
                        
                        # Check if person already exists in gallery
                        person_exists = (person_name in self.identity_manager.simple_gallery.gallery)
                        
                        # Create or update person in simple gallery
                        if person_exists:
                            # Update existing person with new embeddings
                            simple_success = self.identity_manager.simple_gallery._add_embeddings_to_person(
                                person_name, embeddings, qualities, track_id
                            )
                            if simple_success:
                                updated_persons.append(person_name)
                                print(f"✅ Updated person '{person_name}' in simple gallery with track {track_id}")
                        else:
                            # Create new person
                            simple_success = self.identity_manager.simple_gallery.create_person_from_track(
                                person_name, track_id, embeddings, qualities
                            )
                            if simple_success:
                                created_persons.append(person_name)
                                print(f"✅ Created new person '{person_name}' in simple gallery from track {track_id}")
                        
                        # Then handle Enhanced Gallery processing similarly
                        enhanced_success = False
                        if hasattr(self.identity_manager, 'enhanced_gallery') and self.identity_manager.enhanced_gallery and hasattr(self.identity_manager, 'track_crop_buffer') and self.identity_manager.track_crop_buffer:
                            if track_id in self.identity_manager.track_crop_buffer and track_id in self.identity_manager.track_bbox_buffer:
                                crops = self.identity_manager.track_crop_buffer[track_id]
                                bboxes = self.identity_manager.track_bbox_buffer[track_id]
                                parsing_masks = self.identity_manager.track_parsing_buffer.get(track_id, [])
                                
                                # Add embeddings to enhanced gallery with movement analysis
                                embeddings_added = 0
                                for i, (embedding, quality) in enumerate(zip(embeddings, qualities)):
                                    if i < len(crops) and i < len(bboxes):
                                        crop = crops[i]
                                        bbox = bboxes[i]
                                        frame_number = i  # Use index as frame number
                                        parsing_mask = parsing_masks[i] if i < len(parsing_masks) else None
                                        
                                        success = self.identity_manager.enhanced_gallery.add_person_embedding(
                                            person_name, track_id, embedding, bbox, 
                                            crop, frame_number, quality, parsing_mask
                                        )
                                        
                                        if success:
                                            embeddings_added += 1
                                
                                if embeddings_added > 0:
                                    enhanced_success = True
                                    print(f"✅ Enhanced Gallery: Created person '{person_name}' from track {track_id}")
                                    print(f"   📊 Added {embeddings_added}/{len(embeddings)} embeddings with movement context")
                                    created_persons.append(person_name)
                        
                        if simple_success and enhanced_success:
                            print(f"✅ Person '{person_name}' created successfully in both galleries")
                        elif simple_success:
                            print(f"✅ Person '{person_name}' created in simple gallery only")
                        elif enhanced_success:
                            print(f"✅ Person '{person_name}' created in enhanced gallery only")
                        else:
                            print(f"❌ Failed to create person '{person_name}' from track {track_id}")
                        
                        # IMPORTANT: Update track_to_person mapping to mark this track as assigned
                        if simple_success or enhanced_success:
                            self.identity_manager.track_to_person[track_id] = person_name
                            print(f"   📝 Updated track-to-person mapping: Track {track_id} -> {person_name}")
                
                # Update the reviewer's track_to_person mapping as well for consistency
                for track_id, person_name in assignments.items():
                    reviewer.track_to_person[track_id] = person_name
                
                # No need to save gallery here, it will be saved after this method returns
                
                # Show Enhanced Gallery visualization if persons were created
                if created_persons and hasattr(self.identity_manager, 'enhanced_gallery') and self.identity_manager.enhanced_gallery:
                    print("\n" + "=" * 60)
                    print("🎨 ENHANCED GALLERY VISUALIZATION")
                    print("=" * 60)
                    
                    # Show detailed enhanced gallery report
                    self.identity_manager.enhanced_gallery.print_gallery_report()
                    
                    # Ask if user wants to see individual person details
                    show_details = input(f"\n📋 Show detailed person analysis? (y/n, default=y): ").strip().lower()
                    if show_details != 'n':
                        for person_name in created_persons:
                            if person_name in self.identity_manager.enhanced_gallery.gallery:
                                person_data = self.identity_manager.enhanced_gallery.gallery[person_name]
                                print(f"\n👤 {person_name} - Context Analysis:")
                                print(f"   📊 Total embeddings: {person_data.total_embeddings}")
                                print(f"   🎯 Contexts captured: {len(person_data.embeddings_by_context)}")
                                
                                for context_key, context_embeddings in person_data.embeddings_by_context.items():
                                    movement_type, orientation_type = context_key.split('_', 1)
                                    print(f"   • {movement_type.title()} + {orientation_type.replace('_', ' ').title()}: {len(context_embeddings)} embeddings")
                
                # Print final comprehensive report
                print("\n" + "=" * 60)
                print("📊 FINAL IDENTIFICATION SUMMARY")
                print("=" * 60)
                
                # Show both galleries but emphasize the enhanced one
                self.identity_manager.simple_gallery.print_comprehensive_report()
                
                print("\n" + "=" * 60)
                if hasattr(self.identity_manager, 'enhanced_gallery') and self.identity_manager.enhanced_gallery:
                    self.identity_manager.enhanced_gallery.print_gallery_report()
                else:
                    print("   Enhanced gallery not available")
                print("=" * 60)
            else:
                print("ℹ️  No person assignments made")
        else:
            print("❌ Interactive review skipped by user.")
