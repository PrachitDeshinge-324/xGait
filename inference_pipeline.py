"""
XGait Person Identification Inference Pipeline
Complete inference pipeline for person identification using pre-trained XGait model
"""
import torch
import cv2
import numpy as np
import time
import argparse
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
import threading

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.silhouette_model import create_silhouette_extractor
from models.parsing_model import create_human_parsing_model
from models.xgait_model import create_xgait_inference
from utils.device_utils import DeviceManager
from config import get_device_config
from utils.visualization import create_inference_visualizer


class XGaitInferencePipeline:
    """
    Complete inference pipeline for XGait-based person identification
    
    Features:
    - Parallel silhouette extraction and human parsing
    - Pre-trained XGait model for person identification
    - GPU acceleration with device optimization
    - Real-time video processing
    - Gallery-based person identification
    """
    
    def __init__(self,
                 device: str = "mps",
                 xgait_model_path: str = "weights/Gait3D-XGait-120000.pt",
                 parsing_model_path: str = "weights/schp_resnet101.pth",
                 silhouette_model_path: Optional[str] = None,
                 sequence_length: int = 10,
                 identification_threshold: float = 0.6,
                 parallel_processing: bool = True):
        """
        Initialize the inference pipeline
        
        Args:
            device: Computing device ("cpu", "cuda", "mps")
            xgait_model_path: Path to pre-trained XGait model
            parsing_model_path: Path to pre-trained SCHP parsing model
            silhouette_model_path: Path to silhouette extraction model
            sequence_length: Number of frames to use for identification
            identification_threshold: Confidence threshold for person identification
            parallel_processing: Enable parallel processing for speed
        """
        self.device = device
        self.device_config = get_device_config(device)
        self.device_manager = DeviceManager(device, self.device_config["dtype"])
        
        self.sequence_length = sequence_length
        self.identification_threshold = identification_threshold
        self.parallel_processing = parallel_processing
        
        print("🚀 Initializing XGait Inference Pipeline...")
        
        # Initialize models
        self._initialize_models(
            xgait_model_path, parsing_model_path, silhouette_model_path
        )
        
        # Initialize visualization
        self.visualizer = create_inference_visualizer()
        
        # Track management
        self.track_sequences: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=sequence_length)
        )
        self.track_identities: Dict[int, Optional[int]] = {}
        self.track_confidences: Dict[int, float] = {}
        self.track_last_update: Dict[int, int] = {}
        
        # Performance tracking
        self.processing_times = {
            'silhouette': [],
            'parsing': [],
            'identification': [],
            'total': []
        }
        
        # Thread pool for parallel processing
        if parallel_processing:
            self.executor = ThreadPoolExecutor(max_workers=3)
        else:
            self.executor = None
            
        # Threading lock for thread-safe operations
        self.lock = threading.Lock()
        
        print("✅ XGait Inference Pipeline initialized successfully!")
        print(f"   Device: {device}")
        print(f"   Sequence length: {sequence_length}")
        print(f"   Identification threshold: {identification_threshold}")
        print(f"   Parallel processing: {parallel_processing}")
    
    def _initialize_models(self, xgait_path: str, parsing_path: str, silhouette_path: Optional[str]):
        """Initialize all required models"""
        
        # Initialize silhouette extractor
        print("📦 Loading silhouette extraction model...")
        self.silhouette_extractor = create_silhouette_extractor(
            device=self.device,
            model_path=silhouette_path
        )
        
        # Initialize human parsing model
        print("📦 Loading human parsing model...")
        self.parsing_model = create_human_parsing_model(
            device=self.device,
            model_path=parsing_path
        )
        
        # Initialize XGait inference model
        print("📦 Loading XGait model...")
        self.xgait_model = create_xgait_inference(
            device=self.device,
            model_path=xgait_path
        )
    
    def process_track_crops(self, 
                           track_id: int, 
                           crops: List[np.ndarray], 
                           frame_number: int) -> Tuple[Optional[int], float]:
        """
        Process crops for a single track and return identification result
        
        Args:
            track_id: Track ID
            crops: List of person crop images
            frame_number: Current frame number
            
        Returns:
            Tuple of (person_id, confidence)
        """
        if not crops:
            return None, 0.0
        
        start_time = time.time()
        
        # Add crops to track sequence
        with self.lock:
            for crop in crops:
                self.track_sequences[track_id].append(crop)
            self.track_last_update[track_id] = frame_number
        
        # Check if we have enough frames for identification
        current_sequence = list(self.track_sequences[track_id])
        if len(current_sequence) < min(3, self.sequence_length):
            return self.track_identities.get(track_id), self.track_confidences.get(track_id, 0.0)
        
        # Process based on mode
        if self.parallel_processing:
            return self._process_parallel(track_id, current_sequence, start_time)
        else:
            return self._process_sequential(track_id, current_sequence, start_time)
    
    def _process_parallel(self, 
                         track_id: int, 
                         crops: List[np.ndarray],
                         start_time: float) -> Tuple[Optional[int], float]:
        """Process crops in parallel for maximum speed"""
        
        # Submit parallel tasks
        silhouette_future = self.executor.submit(
            self.silhouette_extractor.extract_silhouettes, crops
        )
        parsing_future = self.executor.submit(
            self.parsing_model.parse_humans, crops
        )
        
        try:
            # Wait for both tasks to complete
            silhouettes = silhouette_future.result(timeout=5.0)
            parsing_maps = parsing_future.result(timeout=5.0)
            
            # Record timing
            sil_time = time.time() - start_time
            par_time = sil_time  # They run in parallel
            
            self.processing_times['silhouette'].append(sil_time)
            self.processing_times['parsing'].append(par_time)
            
            # Perform identification
            id_start = time.time()
            person_id, confidence = self.xgait_model.identify_person(
                silhouettes, parsing_maps, track_id, self.identification_threshold
            )
            id_time = time.time() - id_start
            self.processing_times['identification'].append(id_time)
            
            # Update track state
            with self.lock:
                self.track_identities[track_id] = person_id
                self.track_confidences[track_id] = confidence
            
            total_time = time.time() - start_time
            self.processing_times['total'].append(total_time)
            
            return person_id, confidence
            
        except Exception as e:
            print(f"⚠️  Parallel processing error for track {track_id}: {e}")
            return self._process_sequential(track_id, crops, start_time)
    
    def _process_sequential(self, 
                           track_id: int, 
                           crops: List[np.ndarray],
                           start_time: float) -> Tuple[Optional[int], float]:
        """Process crops sequentially"""
        
        try:
            # Extract silhouettes
            sil_start = time.time()
            silhouettes = self.silhouette_extractor.extract_silhouettes(crops)
            sil_time = time.time() - sil_start
            self.processing_times['silhouette'].append(sil_time)
            
            # Extract parsing maps
            par_start = time.time()
            parsing_maps = self.parsing_model.parse_humans(crops)
            par_time = time.time() - par_start
            self.processing_times['parsing'].append(par_time)
            
            # Perform identification
            id_start = time.time()
            person_id, confidence = self.xgait_model.identify_person(
                silhouettes, parsing_maps, track_id, self.identification_threshold
            )
            id_time = time.time() - id_start
            self.processing_times['identification'].append(id_time)
            
            # Update track state
            with self.lock:
                self.track_identities[track_id] = person_id
                self.track_confidences[track_id] = confidence
            
            total_time = time.time() - start_time
            self.processing_times['total'].append(total_time)
            
            return person_id, confidence
            
        except Exception as e:
            print(f"⚠️  Sequential processing error for track {track_id}: {e}")
            return None, 0.0
    
    def process_frame_tracks(self, 
                           frame: np.ndarray,
                           track_data: Dict[int, List[np.ndarray]],
                           frame_number: int) -> Dict[int, Tuple[Optional[int], float]]:
        """
        Process multiple tracks in a single frame
        
        Args:
            frame: Current video frame
            track_data: Dictionary mapping track_id to list of crops
            frame_number: Current frame number
            
        Returns:
            Dictionary mapping track_id to (person_id, confidence)
        """
        results = {}
        
        if self.parallel_processing and len(track_data) > 1:
            # Process multiple tracks in parallel
            futures = {}
            for track_id, crops in track_data.items():
                future = self.executor.submit(
                    self.process_track_crops, track_id, crops, frame_number
                )
                futures[future] = track_id
            
            # Collect results
            for future in as_completed(futures, timeout=10.0):
                track_id = futures[future]
                try:
                    person_id, confidence = future.result()
                    results[track_id] = (person_id, confidence)
                except Exception as e:
                    print(f"⚠️  Error processing track {track_id}: {e}")
                    results[track_id] = (None, 0.0)
        else:
            # Process tracks sequentially
            for track_id, crops in track_data.items():
                person_id, confidence = self.process_track_crops(
                    track_id, crops, frame_number
                )
                results[track_id] = (person_id, confidence)
        
        return results
    
    def cleanup_old_tracks(self, frame_number: int, max_missing_frames: int = 100):
        """Remove tracks that haven't been updated recently"""
        with self.lock:
            tracks_to_remove = []
            for track_id, last_update in self.track_last_update.items():
                if frame_number - last_update > max_missing_frames:
                    tracks_to_remove.append(track_id)
            
            for track_id in tracks_to_remove:
                if track_id in self.track_sequences:
                    del self.track_sequences[track_id]
                if track_id in self.track_identities:
                    del self.track_identities[track_id]
                if track_id in self.track_confidences:
                    del self.track_confidences[track_id]
                if track_id in self.track_last_update:
                    del self.track_last_update[track_id]
                    
            if tracks_to_remove:
                print(f"🧹 Cleaned up {len(tracks_to_remove)} old tracks")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        stats = {}
        for component, times in self.processing_times.items():
            if times:
                stats[component] = {
                    'avg_time': np.mean(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'total_calls': len(times)
                }
        
        # Add gallery stats
        gallery_stats = self.xgait_model.get_gallery_stats()
        stats['gallery'] = gallery_stats
        
        return stats
    
    def reset_performance_stats(self):
        """Reset performance tracking"""
        for key in self.processing_times:
            self.processing_times[key].clear()
    
    def save_person_gallery(self, filepath: str):
        """Save the person gallery to file"""
        self.xgait_model.save_gallery(filepath)
    
    def load_person_gallery(self, filepath: str):
        """Load person gallery from file"""
        self.xgait_model.load_gallery(filepath)
    
    def get_track_identities(self) -> Dict[int, Optional[int]]:
        """Get current track identities"""
        with self.lock:
            return self.track_identities.copy()
    
    def get_track_confidences(self) -> Dict[int, float]:
        """Get current track confidences"""
        with self.lock:
            return self.track_confidences.copy()
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor') and self.executor:
            self.executor.shutdown(wait=True)


def create_inference_pipeline(device: str = "mps", **kwargs) -> XGaitInferencePipeline:
    """
    Create an XGait inference pipeline instance
    
    Args:
        device: Computing device
        **kwargs: Additional arguments for pipeline configuration
        
    Returns:
        Configured inference pipeline
    """
    return XGaitInferencePipeline(device=device, **kwargs)


def demo_inference_pipeline():
    """
    Demonstration of the inference pipeline with dummy data
    """
    print("🎯 Running XGait Inference Pipeline Demo")
    
    # Initialize pipeline
    pipeline = create_inference_pipeline(
        device="mps",  # Change to "cuda" or "cpu" as needed
        identification_threshold=0.6,
        parallel_processing=True
    )
    
    # Create dummy track data
    dummy_crop = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
    track_data = {
        1: [dummy_crop, dummy_crop],
        2: [dummy_crop],
    }
    
    # Process frame
    results = pipeline.process_frame_tracks(
        frame=np.zeros((480, 640, 3), dtype=np.uint8),
        track_data=track_data,
        frame_number=1
    )
    
    print("📊 Processing Results:")
    for track_id, (person_id, confidence) in results.items():
        print(f"   Track {track_id}: Person {person_id} (confidence: {confidence:.3f})")
    
    # Print performance stats
    stats = pipeline.get_performance_stats()
    print("\n📈 Performance Statistics:")
    for component, metrics in stats.items():
        if component != 'gallery':
            print(f"   {component}: {metrics['avg_time']:.4f}s avg, {metrics['total_calls']} calls")
    
    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGait Inference Pipeline Demo")
    parser.add_argument("--device", type=str, default="mps", 
                       choices=["cpu", "cuda", "mps"], help="Device to use")
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_inference_pipeline()
    else:
        print("XGait Inference Pipeline")
        print("Use --demo to run demonstration")
        print("Import this module to use the pipeline in your application")
