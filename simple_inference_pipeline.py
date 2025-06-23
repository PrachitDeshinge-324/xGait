"""
Simplified XGait Person Identification Inference Pipeline
A working version that avoids complex import dependencies
"""
import sys
import os
from pathlib import Path
import torch
import cv2
import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
import threading

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Try to import the working components
try:
    from utils.device_utils import DeviceManager
    from config import get_device_config
    DEVICE_UTILS_AVAILABLE = True
except ImportError:
    print("⚠️  Device utils not available, using basic setup")
    DEVICE_UTILS_AVAILABLE = False

class SimpleInferencePipeline:
    """
    Simplified inference pipeline for XGait-based person identification
    
    Features:
    - Basic silhouette extraction placeholder
    - Simple identification logic
    - Device optimization when available
    - Easy to extend and integrate
    """
    
    def __init__(self, 
                 device: str = "cpu",
                 identification_threshold: float = 0.7,
                 parallel_processing: bool = True,
                 max_workers: int = 4):
        """
        Initialize the simplified inference pipeline
        
        Args:
            device: Device to use for inference ("cpu", "cuda", "mps")
            identification_threshold: Threshold for person identification
            parallel_processing: Whether to use parallel processing
            max_workers: Number of worker threads
        """
        self.device = device
        self.identification_threshold = identification_threshold
        self.parallel_processing = parallel_processing
        self.max_workers = max_workers
        
        # Initialize device manager if available
        if DEVICE_UTILS_AVAILABLE:
            try:
                self.device_config = get_device_config(device)
                self.device_manager = DeviceManager(device, self.device_config["dtype"])
                print(f"✅ Device manager initialized for {device}")
            except Exception as e:
                print(f"⚠️  Device manager initialization failed: {e}, using basic setup")
                self.device_manager = None
        else:
            self.device_manager = None
        
        # Initialize components
        self._initialize_models()
        
        # Gallery for storing known person features
        self.gallery = {}
        self.gallery_lock = threading.Lock()
        
        print(f"✅ Simple inference pipeline initialized")
        print(f"   Device: {device}")
        print(f"   Parallel processing: {parallel_processing}")
        print(f"   Identification threshold: {identification_threshold}")
    
    def _initialize_models(self):
        """Initialize the inference models"""
        # Placeholder for model initialization
        # In a real implementation, this would load the actual models
        print("📦 Initializing models...")
        
        # Silhouette extraction (placeholder)
        self.silhouette_extractor = None
        print("   - Silhouette extractor: placeholder")
        
        # Human parsing (placeholder)  
        self.parsing_model = None
        print("   - Human parsing model: placeholder")
        
        # XGait model (placeholder)
        self.xgait_model = None
        print("   - XGait model: placeholder")
        
        print("✅ Models initialized (placeholders)")
    
    def extract_silhouettes(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract silhouettes from person crops (placeholder implementation)
        
        Args:
            crops: List of person crop images
            
        Returns:
            List of silhouette masks
        """
        if not crops:
            return []
        
        # Placeholder: create dummy silhouettes
        silhouettes = []
        for crop in crops:
            h, w = crop.shape[:2]
            # Create a simple rectangular silhouette as placeholder
            silhouette = np.zeros((h, w), dtype=np.uint8)
            center_x, center_y = w // 2, h // 2
            width, height = min(w, h) // 3, min(w, h) // 2
            x1, y1 = center_x - width // 2, center_y - height // 2
            x2, y2 = center_x + width // 2, center_y + height // 2
            silhouette[y1:y2, x1:x2] = 255
            silhouettes.append(silhouette)
        
        return silhouettes
    
    def extract_parsing(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract human parsing from person crops (placeholder implementation)
        
        Args:
            crops: List of person crop images
            
        Returns:
            List of parsing masks
        """
        if not crops:
            return []
        
        # Placeholder: create dummy parsing
        parsing_results = []
        for crop in crops:
            h, w = crop.shape[:2]
            # Create a simple parsing mask as placeholder
            parsing = np.random.randint(0, 20, (h, w), dtype=np.uint8)
            parsing_results.append(parsing)
        
        return parsing_results
    
    def extract_features(self, silhouettes: List[np.ndarray], parsing_masks: List[np.ndarray]) -> np.ndarray:
        """
        Extract XGait features from silhouettes and parsing masks (placeholder)
        
        Args:
            silhouettes: List of silhouette masks
            parsing_masks: List of parsing masks
            
        Returns:
            Feature vectors for each person
        """
        if not silhouettes:
            return np.array([])
        
        # Placeholder: create dummy features
        features = []
        for sil, parsing in zip(silhouettes, parsing_masks):
            # Simple feature extraction as placeholder
            sil_features = np.mean(sil) / 255.0
            parsing_features = np.mean(parsing) / 20.0
            # Combine into a simple feature vector
            feature = np.array([sil_features, parsing_features, np.random.rand()])
            features.append(feature)
        
        return np.array(features)
    
    def process_tracks(self, tracks_data: Dict[int, List[np.ndarray]]) -> Dict[int, Dict]:
        """
        Process person tracks and perform identification
        
        Args:
            tracks_data: Dictionary mapping track_id to list of person crops
            
        Returns:
            Dictionary with identification results for each track
        """
        if not tracks_data:
            return {}
        
        results = {}
        
        if self.parallel_processing and len(tracks_data) > 1:
            # Process tracks in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_track = {
                    executor.submit(self._process_single_track, track_id, crops): track_id
                    for track_id, crops in tracks_data.items()
                }
                
                for future in as_completed(future_to_track):
                    track_id = future_to_track[future]
                    try:
                        results[track_id] = future.result()
                    except Exception as e:
                        print(f"❌ Error processing track {track_id}: {e}")
                        results[track_id] = {"identified_person": None, "confidence": 0.0}
        else:
            # Process tracks sequentially
            for track_id, crops in tracks_data.items():
                results[track_id] = self._process_single_track(track_id, crops)
        
        return results
    
    def _process_single_track(self, track_id: int, crops: List[np.ndarray]) -> Dict:
        """
        Process a single track for identification
        
        Args:
            track_id: ID of the track
            crops: List of person crops for this track
            
        Returns:
            Identification result
        """
        if not crops:
            return {"identified_person": None, "confidence": 0.0}
        
        try:
            # Extract silhouettes and parsing in parallel
            if self.parallel_processing:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    sil_future = executor.submit(self.extract_silhouettes, crops)
                    parsing_future = executor.submit(self.extract_parsing, crops)
                    
                    silhouettes = sil_future.result()
                    parsing_masks = parsing_future.result()
            else:
                silhouettes = self.extract_silhouettes(crops)
                parsing_masks = self.extract_parsing(crops)
            
            # Extract features
            features = self.extract_features(silhouettes, parsing_masks)
            
            if len(features) == 0:
                return {"identified_person": None, "confidence": 0.0}
            
            # Aggregate features for the track (use mean)
            track_feature = np.mean(features, axis=0)
            
            # Perform identification against gallery
            identified_person, confidence = self._identify_person(track_feature)
            
            return {
                "identified_person": identified_person,
                "confidence": confidence,
                "num_crops": len(crops),
                "feature_vector": track_feature.tolist()
            }
            
        except Exception as e:
            print(f"❌ Error processing track {track_id}: {e}")
            return {"identified_person": None, "confidence": 0.0}
    
    def _identify_person(self, feature: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Identify a person against the gallery
        
        Args:
            feature: Feature vector to identify
            
        Returns:
            Tuple of (person_id, confidence)
        """
        with self.gallery_lock:
            if not self.gallery:
                return None, 0.0
            
            best_match = None
            best_confidence = 0.0
            
            for person_id, gallery_features in self.gallery.items():
                # Simple cosine similarity
                for gallery_feature in gallery_features:
                    similarity = self._cosine_similarity(feature, gallery_feature)
                    if similarity > best_confidence:
                        best_confidence = similarity
                        best_match = person_id
            
            if best_confidence >= self.identification_threshold:
                return best_match, best_confidence
            else:
                return None, best_confidence
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)
    
    def add_to_gallery(self, person_id: str, features: List[np.ndarray]):
        """
        Add person features to the gallery
        
        Args:
            person_id: ID of the person
            features: List of feature vectors for this person
        """
        with self.gallery_lock:
            if person_id not in self.gallery:
                self.gallery[person_id] = []
            self.gallery[person_id].extend(features)
        
        print(f"✅ Added {len(features)} features for person '{person_id}' to gallery")
    
    def clear_gallery(self):
        """Clear the gallery"""
        with self.gallery_lock:
            self.gallery.clear()
        print("🗑️  Gallery cleared")
    
    def get_gallery_stats(self) -> Dict:
        """Get statistics about the gallery"""
        with self.gallery_lock:
            stats = {
                "num_persons": len(self.gallery),
                "total_features": sum(len(features) for features in self.gallery.values()),
                "persons": list(self.gallery.keys())
            }
        return stats
    
    def process_gait_features(self, tracks_features: Dict[int, np.ndarray]) -> Dict[int, Dict]:
        """
        Process tracks using pre-extracted XGait features
        
        Args:
            tracks_features: Dictionary mapping track_id -> feature_vector
            
        Returns:
            Dictionary with identification results for each track
        """
        results = {}
        
        for track_id, feature_vector in tracks_features.items():
            try:
                # Directly use the provided XGait features for identification
                identified_person, confidence = self._identify_person(feature_vector)
                
                results[track_id] = {
                    "identified_person": identified_person,
                    "confidence": confidence,
                    "feature_vector": feature_vector,
                    "method": "xgait_features"
                }
                
            except Exception as e:
                print(f"⚠️  Error processing XGait features for track {track_id}: {e}")
                results[track_id] = {
                    "identified_person": None,
                    "confidence": 0.0,
                    "error": str(e),
                    "method": "xgait_features"
                }
        
        return results


def create_simple_inference_pipeline(device: str = "cpu", 
                                   identification_threshold: float = 0.7,
                                   parallel_processing: bool = True,
                                   max_workers: int = 4) -> SimpleInferencePipeline:
    """
    Create a simplified inference pipeline instance
    
    Args:
        device: Device to use for inference
        identification_threshold: Threshold for person identification
        parallel_processing: Whether to use parallel processing
        max_workers: Number of worker threads
        
    Returns:
        SimpleInferencePipeline instance
    """
    return SimpleInferencePipeline(
        device=device,
        identification_threshold=identification_threshold,
        parallel_processing=parallel_processing,
        max_workers=max_workers
    )


def demo_simple_inference():
    """Demonstrate the simple inference pipeline with dummy data"""
    print("\n🎯 Demo: Simple Inference Pipeline")
    print("=" * 50)
    
    # Create pipeline
    pipeline = create_simple_inference_pipeline(device="cpu")
    
    # Create dummy data
    dummy_crops = [
        np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8),
        np.random.randint(0, 255, (120, 60, 3), dtype=np.uint8),
        np.random.randint(0, 255, (140, 70, 3), dtype=np.uint8)
    ]
    
    tracks_data = {
        1: dummy_crops[:2],  # Track 1 has 2 crops
        2: dummy_crops[2:3]  # Track 2 has 1 crop
    }
    
    print(f"📊 Processing {len(tracks_data)} tracks with {sum(len(crops) for crops in tracks_data.values())} total crops")
    
    # Process tracks
    start_time = time.time()
    results = pipeline.process_tracks(tracks_data)
    processing_time = time.time() - start_time
    
    # Display results
    print(f"\n⏱️  Processing completed in {processing_time:.3f} seconds")
    print("\n📋 Results:")
    for track_id, result in results.items():
        print(f"   Track {track_id}:")
        print(f"     Identified person: {result.get('identified_person', 'Unknown')}")
        print(f"     Confidence: {result.get('confidence', 0.0):.3f}")
        print(f"     Number of crops: {result.get('num_crops', 0)}")
    
    # Demonstrate gallery functionality
    print("\n📚 Gallery demonstration:")
    dummy_features = [np.random.rand(3) for _ in range(3)]
    pipeline.add_to_gallery("Person_A", dummy_features[:2])
    pipeline.add_to_gallery("Person_B", dummy_features[2:3])
    
    stats = pipeline.get_gallery_stats()
    print(f"   Gallery stats: {stats}")
    
    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    demo_simple_inference()
