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
            
            # Keep only recent frames
            if len(self.track_silhouettes[track_id]) > self.sequence_buffer_size:
                self.track_silhouettes[track_id].pop(0)
                self.track_parsing_masks[track_id].pop(0)
            
            # Step 4: Extract XGait features if conditions are met
            feature_vector = np.zeros(256)
            xgait_extracted = False
            
            if (len(self.track_silhouettes[track_id]) >= self.min_sequence_length and 
                frame_count - self.track_last_xgait_extraction[track_id] >= self.xgait_extraction_interval):
                try:
                    silhouette_sequence = self.track_silhouettes[track_id].copy()
                    parsing_sequence = self.track_parsing_masks[track_id].copy()
                    
                    feature_vector = self.xgait_model.extract_features_from_sequence(
                        silhouettes=silhouette_sequence,
                        parsing_masks=parsing_sequence
                    )
                    
                    sequence_quality = self._compute_sequence_quality(silhouette_sequence)
                    
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
    
    def _compute_sequence_quality(self, silhouette_sequence: List[np.ndarray]) -> float:
        """
        Compute quality score for a silhouette sequence.
        
        Args:
            silhouette_sequence: List of silhouette masks
            
        Returns:
            Quality score between 0 and 1
        """
        if not silhouette_sequence or len(silhouette_sequence) < 2:
            return 0.0
        
        qualities = []
        
        # 1. Silhouette completeness
        completeness_scores = []
        for sil in silhouette_sequence:
            if sil.size > 0:
                non_zero_ratio = np.sum(sil > 0) / sil.size
                completeness_scores.append(non_zero_ratio)
        
        if completeness_scores:
            avg_completeness = np.mean(completeness_scores)
            qualities.append(min(avg_completeness * 2, 1.0))
        
        # 2. Temporal consistency
        if len(silhouette_sequence) >= 2:
            consistency_scores = []
            for i in range(len(silhouette_sequence) - 1):
                sil1, sil2 = silhouette_sequence[i], silhouette_sequence[i + 1]
                if sil1.shape == sil2.shape and sil1.size > 0:
                    intersection = np.sum((sil1 > 0) & (sil2 > 0))
                    union = np.sum((sil1 > 0) | (sil2 > 0))
                    if union > 0:
                        consistency_scores.append(intersection / union)
            
            if consistency_scores:
                avg_consistency = np.mean(consistency_scores)
                qualities.append(avg_consistency)
        
        # 3. Sequence length bonus
        length_bonus = min(len(silhouette_sequence) / 30.0, 1.0)
        qualities.append(length_bonus)
        
        # Combine qualities
        if qualities:
            weights = [0.4, 0.4, 0.2][:len(qualities)]
            final_quality = sum(q * w for q, w in zip(qualities, weights)) / sum(weights)
            return min(max(final_quality, 0.0), 1.0)
        
        return 0.0
    
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
        """Get gait parsing statistics"""
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
            
            gallery_names = list(self.identity_manager.simple_gallery.gallery.keys())
            gallery_embs = [self.identity_manager.simple_gallery.gallery[name].prototype for name in gallery_names]
            
            sim_ax = axes[4, 0]
            if frame_features and gallery_embs:
                sims = cosine_similarity(frame_features, gallery_embs)
                im = sim_ax.imshow(sims, cmap='coolwarm', vmin=0, vmax=1)
                sim_ax.set_title('Track vs Gallery Cosine Similarity', fontsize=10)
                sim_ax.set_xticks(range(len(gallery_names)))
                sim_ax.set_yticks(range(len(frame_track_ids)))
                sim_ax.set_xticklabels(gallery_names, fontsize=8, rotation=90)
                sim_ax.set_yticklabels(frame_track_ids, fontsize=8)
                fig.colorbar(im, ax=sim_ax, fraction=0.046, pad=0.04)
            else:
                sim_ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
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
