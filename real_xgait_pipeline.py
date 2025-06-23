"""
Real XGait Inference Pipeline
Uses actual U²-Net, SCHP, and XGait models instead of placeholders
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

# Import actual model components
try:
    from models.silhouette_model import create_silhouette_extractor
    from models.parsing_model import create_human_parsing_model
    from models.xgait_model import create_xgait_inference
    REAL_MODELS_AVAILABLE = True
    print("✅ Real XGait model imports successful")
except ImportError as e:
    print(f"❌ Real model imports failed: {e}")
    REAL_MODELS_AVAILABLE = False


class RealXGaitInferencePipeline:
    """
    Real XGait inference pipeline using actual trained models
    
    Features:
    - Actual U²-Net for silhouette extraction
    - Actual SCHP for human parsing
    - Actual XGait for feature extraction and identification
    - Device optimization and parallel processing
    - Gallery management for known persons
    """
    
    def __init__(self, 
                 device: str = "cpu",
                 silhouette_model_path: str = None,
                 parsing_model_path: str = "weights/schp_resnet101.pth",
                 xgait_model_path: str = "weights/Gait3D-XGait-120000.pt",
                 identification_threshold: float = 0.7,
                 parallel_processing: bool = True,
                 max_workers: int = 4):
        """
        Initialize the real XGait inference pipeline
        
        Args:
            device: Device to use for inference ("cpu", "cuda", "mps")
            silhouette_model_path: Path to U²-Net model weights
            parsing_model_path: Path to SCHP model weights
            xgait_model_path: Path to XGait model weights
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
        
        # Initialize real models
        self._initialize_real_models(silhouette_model_path, parsing_model_path, xgait_model_path)
        
        # Gallery for storing known person features
        self.gallery = {}
        self.gallery_lock = threading.Lock()
        
        print(f"✅ Real XGait inference pipeline initialized")
        print(f"   Device: {device}")
        print(f"   Parallel processing: {parallel_processing}")
        print(f"   Identification threshold: {identification_threshold}")
    
    def _initialize_real_models(self, silhouette_model_path: str, parsing_model_path: str, xgait_model_path: str):
        """Initialize the real XGait models"""
        print("📦 Initializing REAL models...")
        
        if not REAL_MODELS_AVAILABLE:
            print("❌ Real models not available, falling back to placeholders")
            self.silhouette_extractor = None
            self.parsing_model = None
            self.xgait_model = None
            self.models_loaded = False
            return
        
        try:
            # Initialize U²-Net for silhouette extraction
            print("   🔄 Loading U²-Net silhouette model...")
            self.silhouette_extractor = create_silhouette_extractor(
                device=self.device,
                model_path=silhouette_model_path
            )
            print("   ✅ U²-Net silhouette model loaded")
            
        except Exception as e:
            print(f"   ❌ Failed to load U²-Net model: {e}")
            self.silhouette_extractor = None
        
        try:
            # Initialize SCHP for human parsing
            print("   🔄 Loading SCHP parsing model...")
            self.parsing_model = create_human_parsing_model(
                device=self.device,
                model_path=parsing_model_path
            )
            print("   ✅ SCHP parsing model loaded")
            
        except Exception as e:
            print(f"   ❌ Failed to load SCHP model: {e}")
            self.parsing_model = None
        
        try:
            # Initialize XGait for feature extraction
            print("   🔄 Loading XGait model...")
            self.xgait_model = create_xgait_inference(
                device=self.device,
                model_path=xgait_model_path
            )
            print("   ✅ XGait model loaded")
            
        except Exception as e:
            print(f"   ❌ Failed to load XGait model: {e}")
            self.xgait_model = None
        
        # Check if all models loaded successfully
        self.models_loaded = all([
            self.silhouette_extractor is not None,
            self.parsing_model is not None,
            self.xgait_model is not None
        ])
        
        if self.models_loaded:
            print("✅ All real models loaded successfully")
        else:
            print("⚠️  Some models failed to load - using available models")
    
    def extract_silhouettes(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract silhouettes from person crops using real U²-Net model
        
        Args:
            crops: List of person crop images
            
        Returns:
            List of silhouette masks
        """
        if not crops:
            return []
        
        if self.silhouette_extractor is None:
            print("⚠️  U²-Net not available, using placeholder silhouettes")
            return self._extract_placeholder_silhouettes(crops)
        
        try:
            # Use real U²-Net model
            silhouettes = self.silhouette_extractor.extract_silhouettes(crops)
            print(f"✅ Extracted {len(silhouettes)} real silhouettes using U²-Net")
            return silhouettes
            
        except Exception as e:
            print(f"❌ U²-Net extraction failed: {e}, falling back to placeholder")
            return self._extract_placeholder_silhouettes(crops)
    
    def extract_parsing(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract human parsing from person crops using real SCHP model
        
        Args:
            crops: List of person crop images
            
        Returns:
            List of parsing masks
        """
        if not crops:
            return []
        
        if self.parsing_model is None:
            print("⚠️  SCHP not available, using placeholder parsing")
            return self._extract_placeholder_parsing(crops)
        
        try:
            # Use real SCHP model
            parsing_masks = self.parsing_model.extract_parsing(crops)
            print(f"✅ Extracted {len(parsing_masks)} real parsing masks using SCHP")
            return parsing_masks
            
        except Exception as e:
            print(f"❌ SCHP extraction failed: {e}, falling back to placeholder")
            return self._extract_placeholder_parsing(crops)
    
    def extract_features(self, silhouettes: List[np.ndarray], parsing_masks: List[np.ndarray]) -> np.ndarray:
        """
        Extract XGait features from silhouettes and parsing masks using real XGait model
        
        Args:
            silhouettes: List of silhouette masks
            parsing_masks: List of parsing masks
            
        Returns:
            Feature vectors for each person
        """
        if not silhouettes:
            return np.array([])
        
        if self.xgait_model is None:
            print("⚠️  XGait not available, using placeholder features")
            return self._extract_placeholder_features(silhouettes, parsing_masks)
        
        try:
            # Use real XGait model
            features = self.xgait_model.extract_features(silhouettes, parsing_masks)
            print(f"✅ Extracted {len(features)} real feature vectors using XGait")
            return features
            
        except Exception as e:
            print(f"❌ XGait extraction failed: {e}, falling back to placeholder")
            return self._extract_placeholder_features(silhouettes, parsing_masks)
    
    def _extract_placeholder_silhouettes(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """Fallback placeholder silhouette extraction"""
        silhouettes = []
        for crop in crops:
            h, w = crop.shape[:2]
            silhouette = np.zeros((h, w), dtype=np.uint8)
            center_x, center_y = w // 2, h // 2
            width, height = min(w, h) // 3, min(w, h) // 2
            x1, y1 = center_x - width // 2, center_y - height // 2
            x2, y2 = center_x + width // 2, center_y + height // 2
            silhouette[y1:y2, x1:x2] = 255
            silhouettes.append(silhouette)
        return silhouettes
    
    def _extract_placeholder_parsing(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """Fallback placeholder parsing extraction"""
        parsing_results = []
        for crop in crops:
            h, w = crop.shape[:2]
            parsing = np.random.randint(0, 20, (h, w), dtype=np.uint8)
            parsing_results.append(parsing)
        return parsing_results
    
    def _extract_placeholder_features(self, silhouettes: List[np.ndarray], parsing_masks: List[np.ndarray]) -> np.ndarray:
        """Fallback placeholder feature extraction"""
        features = []
        for sil, parsing in zip(silhouettes, parsing_masks):
            sil_features = np.mean(sil) / 255.0
            parsing_features = np.mean(parsing) / 20.0
            feature = np.array([sil_features, parsing_features, np.random.rand()])
            features.append(feature)
        return np.array(features)
    
    def process_tracks(self, tracks_data: Dict[int, List[np.ndarray]]) -> Dict[int, Dict]:
        """
        Process person tracks and perform identification using real models
        
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
        Process a single track for identification using real models
        
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
            
            # Extract features using real XGait model
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
                "feature_vector": track_feature.tolist(),
                "using_real_models": self.models_loaded
            }
            
        except Exception as e:
            print(f"❌ Error processing track {track_id}: {e}")
            return {"identified_person": None, "confidence": 0.0}
    
    def _identify_person(self, feature: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Identify a person against the gallery using real features
        
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
                # Use more sophisticated similarity for real features
                for gallery_feature in gallery_features:
                    if self.models_loaded:
                        # Use real feature similarity (e.g., cosine or learned metric)
                        similarity = self._cosine_similarity(feature, gallery_feature)
                    else:
                        # Simple similarity for placeholder features
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
        
        model_type = "real" if self.models_loaded else "placeholder"
        print(f"✅ Added {len(features)} {model_type} features for person '{person_id}' to gallery")
    
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
                "persons": list(self.gallery.keys()),
                "using_real_models": self.models_loaded
            }
        return stats


def create_real_xgait_pipeline(device: str = "cpu", 
                              silhouette_model_path: str = None,
                              parsing_model_path: str = "weights/schp_resnet101.pth",
                              xgait_model_path: str = "weights/Gait3D-XGait-120000.pt",
                              identification_threshold: float = 0.7,
                              parallel_processing: bool = True,
                              max_workers: int = 4) -> RealXGaitInferencePipeline:
    """
    Create a real XGait inference pipeline instance
    
    Args:
        device: Device to use for inference
        silhouette_model_path: Path to U²-Net model weights
        parsing_model_path: Path to SCHP model weights
        xgait_model_path: Path to XGait model weights
        identification_threshold: Threshold for person identification
        parallel_processing: Whether to use parallel processing
        max_workers: Number of worker threads
        
    Returns:
        RealXGaitInferencePipeline instance
    """
    return RealXGaitInferencePipeline(
        device=device,
        silhouette_model_path=silhouette_model_path,
        parsing_model_path=parsing_model_path,
        xgait_model_path=xgait_model_path,
        identification_threshold=identification_threshold,
        parallel_processing=parallel_processing,
        max_workers=max_workers
    )


def demo_real_xgait_inference():
    """Demonstrate the real XGait inference pipeline"""
    print("\n🎯 Demo: Real XGait Inference Pipeline")
    print("=" * 50)
    
    # Create real pipeline
    pipeline = create_real_xgait_pipeline(device="cpu")
    
    # Create dummy data (same as before for comparison)
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
    
    # Process tracks using real models
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
        print(f"     Using real models: {result.get('using_real_models', False)}")
    
    # Get gallery stats
    stats = pipeline.get_gallery_stats()
    print(f"\n📚 Gallery stats: {stats}")
    
    print("\n✅ Real XGait demo completed!")


if __name__ == "__main__":
    demo_real_xgait_inference()
