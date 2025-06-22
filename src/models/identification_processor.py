"""
Identification Processor for XGait-based Person Identification
Coordinates silhouette extraction, human parsing, and XGait inference in parallel
"""
import sys
import os
from pathlib import Path

# Add parent directory to path for utils and config imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import torch
import numpy as np
import cv2
import time
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict, deque

from models.silhouette_model import create_silhouette_extractor
from models.parsing_model import create_human_parsing_model  
from models.xgait_model import create_xgait_inference
from utils.device_utils import DeviceManager
from config import get_device_config

class IdentificationProcessor:
    """
    Main processor for person identification using XGait pipeline
    Handles parallel silhouette extraction, human parsing, and identification
    """
    def __init__(self, 
                 device: str = "mps",
                 xgait_model_path: str = "weights/Gait3D-XGait-120000.pt",
                 parsing_model_path: str = "weights/schp_resnet101.pth",
                 silhouette_model_path: Optional[str] = None,
                 sequence_length: int = 10,
                 identification_threshold: float = 0.6,
                 parallel_processing: bool = True):
        
        self.device = device
        self.device_config = get_device_config(device)
        self.device_manager = DeviceManager(device, self.device_config["dtype"])
        
        self.sequence_length = sequence_length
        self.identification_threshold = identification_threshold
        self.parallel_processing = parallel_processing
        
        # Initialize models
        print("🚀 Initializing identification models...")
        
        # Initialize silhouette extractor
        self.silhouette_extractor = create_silhouette_extractor(
            device=device, 
            model_path=silhouette_model_path
        )
        
        # Initialize human parsing model
        self.parsing_model = create_human_parsing_model(
            device=device,
            model_path=parsing_model_path
        )
        
        # Initialize XGait inference model
        self.xgait_model = create_xgait_inference(
            device=device,
            model_path=xgait_model_path
        )
        
        # Track management
        self.track_sequences: Dict[int, deque] = defaultdict(lambda: deque(maxlen=sequence_length))
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
        
        print(f"✅ IdentificationProcessor initialized")
        print(f"   Device: {device}")
        print(f"   Sequence length: {sequence_length}")
        print(f"   Identification threshold: {identification_threshold}")
        print(f"   Parallel processing: {parallel_processing}")
    
    def process_track_crops(self, 
                           track_id: int, 
                           crops: List[np.ndarray], 
                           frame_number: int) -> Tuple[Optional[int], float]:
        """
        Process person crops for a single track to extract identity
        
        Args:
            track_id: Track identifier
            crops: List of person crop images for this track
            frame_number: Current frame number
            
        Returns:
            Tuple of (person_id, confidence_score)
        """
        if not crops:
            return None, 0.0
        
        start_time = time.time()
        
        # Add crops to track sequence
        for crop in crops:
            self.track_sequences[track_id].append(crop)
        
        self.track_last_update[track_id] = frame_number
        
        # Check if we have enough frames for identification
        if len(self.track_sequences[track_id]) < min(5, self.sequence_length):
            return self.track_identities.get(track_id), self.track_confidences.get(track_id, 0.0)
        
        # Get recent crops for processing
        recent_crops = list(self.track_sequences[track_id])
        
        if self.parallel_processing and self.executor:
            # Parallel processing
            person_id, confidence = self._process_parallel(track_id, recent_crops)
        else:
            # Sequential processing
            person_id, confidence = self._process_sequential(track_id, recent_crops)
        
        # Update track identity
        if person_id is not None:
            self.track_identities[track_id] = person_id
            self.track_confidences[track_id] = confidence
        
        total_time = time.time() - start_time
        self.processing_times['total'].append(total_time)
        
        return person_id, confidence
    
    def _process_sequential(self, track_id: int, crops: List[np.ndarray]) -> Tuple[Optional[int], float]:
        """Process crops sequentially"""
        
        # Step 1: Extract silhouettes
        sil_start = time.time()
        silhouettes = self.silhouette_extractor.extract_silhouettes(crops, target_size=(64, 44))
        self.processing_times['silhouette'].append(time.time() - sil_start)
        
        # Step 2: Extract parsing maps
        par_start = time.time()
        parsing_maps = self.parsing_model.parse_humans(crops, target_size=(64, 44))
        self.processing_times['parsing'].append(time.time() - par_start)
        
        # Step 3: Identify person
        id_start = time.time()
        person_id, confidence = self.xgait_model.identify_person(
            silhouettes, parsing_maps, track_id, self.identification_threshold
        )
        self.processing_times['identification'].append(time.time() - id_start)
        
        return person_id, confidence
    
    def _process_parallel(self, track_id: int, crops: List[np.ndarray]) -> Tuple[Optional[int], float]:
        """Process crops in parallel"""
        
        # Submit parallel tasks
        future_silhouettes = self.executor.submit(
            self.silhouette_extractor.extract_silhouettes, crops, (64, 44)
        )
        future_parsing = self.executor.submit(
            self.parsing_model.parse_humans, crops, (64, 44)
        )
        
        # Wait for results
        sil_start = time.time()
        silhouettes = future_silhouettes.result()
        self.processing_times['silhouette'].append(time.time() - sil_start)
        
        par_start = time.time()
        parsing_maps = future_parsing.result()
        self.processing_times['parsing'].append(time.time() - par_start)
        
        # Identify person
        id_start = time.time()
        person_id, confidence = self.xgait_model.identify_person(
            silhouettes, parsing_maps, track_id, self.identification_threshold
        )
        self.processing_times['identification'].append(time.time() - id_start)
        
        return person_id, confidence
    
    def process_multiple_tracks(self, 
                               track_data: Dict[int, List[np.ndarray]], 
                               frame_number: int) -> Dict[int, Tuple[Optional[int], float]]:
        """
        Process multiple tracks in batch for better efficiency
        
        Args:
            track_data: Dictionary mapping track_id to list of crops
            frame_number: Current frame number
            
        Returns:
            Dictionary mapping track_id to (person_id, confidence)
        """
        if not track_data:
            return {}
        
        start_time = time.time()
        results = {}
        
        # Update track sequences
        for track_id, crops in track_data.items():
            for crop in crops:
                self.track_sequences[track_id].append(crop)
            self.track_last_update[track_id] = frame_number
        
        # Filter tracks that have enough frames
        ready_tracks = {
            track_id: list(self.track_sequences[track_id])
            for track_id in track_data.keys()
            if len(self.track_sequences[track_id]) >= min(5, self.sequence_length)
        }
        
        if not ready_tracks:
            # Return existing identities for tracks not ready
            for track_id in track_data.keys():
                results[track_id] = (
                    self.track_identities.get(track_id),
                    self.track_confidences.get(track_id, 0.0)
                )
            return results
        
        if self.parallel_processing and self.executor:
            # Batch parallel processing
            track_ids = list(ready_tracks.keys())
            crops_batch = list(ready_tracks.values())
            
            # Extract silhouettes and parsing maps in parallel
            future_silhouettes = self.executor.submit(
                self.silhouette_extractor.extract_silhouettes_batch, crops_batch
            )
            future_parsing = self.executor.submit(
                self.parsing_model.parse_humans_batch, crops_batch
            )
            
            # Get results
            sil_start = time.time()
            silhouettes_batch = future_silhouettes.result()
            self.processing_times['silhouette'].append(time.time() - sil_start)
            
            par_start = time.time()
            parsing_maps_batch = future_parsing.result()
            self.processing_times['parsing'].append(time.time() - par_start)
            
            # Identify persons
            id_start = time.time()
            identification_results = self.xgait_model.identify_persons_batch(
                silhouettes_batch, parsing_maps_batch, track_ids, self.identification_threshold
            )
            self.processing_times['identification'].append(time.time() - id_start)
            
            # Update results
            for track_id, (person_id, confidence) in zip(track_ids, identification_results):
                if person_id is not None:
                    self.track_identities[track_id] = person_id
                    self.track_confidences[track_id] = confidence
                results[track_id] = (person_id, confidence)
        
        else:
            # Sequential processing for each track
            for track_id, crops in ready_tracks.items():
                person_id, confidence = self._process_sequential(track_id, crops)
                if person_id is not None:
                    self.track_identities[track_id] = person_id
                    self.track_confidences[track_id] = confidence
                results[track_id] = (person_id, confidence)
        
        # Add existing identities for tracks not processed
        for track_id in track_data.keys():
            if track_id not in results:
                results[track_id] = (
                    self.track_identities.get(track_id),
                    self.track_confidences.get(track_id, 0.0)
                )
        
        total_time = time.time() - start_time
        self.processing_times['total'].append(total_time)
        
        return results
    
    def cleanup_old_tracks(self, current_frame: int, max_age: int = 100):
        """Remove old tracks that haven't been updated recently"""
        old_tracks = [
            track_id for track_id, last_frame in self.track_last_update.items()
            if current_frame - last_frame > max_age
        ]
        
        for track_id in old_tracks:
            if track_id in self.track_sequences:
                del self.track_sequences[track_id]
            if track_id in self.track_identities:
                del self.track_identities[track_id]
            if track_id in self.track_confidences:
                del self.track_confidences[track_id]
            if track_id in self.track_last_update:
                del self.track_last_update[track_id]
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        stats = {}
        for stage, times in self.processing_times.items():
            if times:
                stats[stage] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'count': len(times)
                }
            else:
                stats[stage] = {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0}
        
        return stats
    
    def get_identification_stats(self) -> Dict:
        """Get identification statistics"""
        gallery_stats = self.xgait_model.get_gallery_stats()
        
        return {
            'active_tracks': len(self.track_identities),
            'identified_tracks': len([tid for tid, pid in self.track_identities.items() if pid is not None]),
            'gallery_stats': gallery_stats,
            'average_confidence': np.mean(list(self.track_confidences.values())) if self.track_confidences else 0.0
        }
    
    def save_gallery(self, filepath: str):
        """Save person gallery to file"""
        self.xgait_model.save_gallery(filepath)
    
    def load_gallery(self, filepath: str):
        """Load person gallery from file"""
        self.xgait_model.load_gallery(filepath)
    
    def clear_memory(self):
        """Clear memory and reset state"""
        self.track_sequences.clear()
        self.track_identities.clear()
        self.track_confidences.clear()
        self.track_last_update.clear()
        self.xgait_model.clear_gallery()
        
        # Clear performance history
        for stage in self.processing_times:
            self.processing_times[stage].clear()
    
    def __del__(self):
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=True)

def create_identification_processor(device: str = "mps", **kwargs) -> IdentificationProcessor:
    """Create an IdentificationProcessor instance"""
    return IdentificationProcessor(device=device, **kwargs)
