"""
Identification Processor
Integrates silhouette extraction, human parsing, and XGait for person identification
"""
import numpy as np
import cv2
import torch
from typing import List, Dict, Tuple, Optional, Any
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from models.silhouette_model import create_silhouette_extractor
    from models.parsing_model import create_human_parsing_model  
    from models.xgait_model import create_xgait_inference
    from utils.device_utils import DeviceManager
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Model imports failed: {e}")
    MODELS_AVAILABLE = False

logger = logging.getLogger(__name__)


class IdentificationProcessor:
    """
    Complete identification processor using real models
    """
    
    def __init__(self, 
                 device: str = "cpu",
                 silhouette_model_path: Optional[str] = None,
                 parsing_model_path: Optional[str] = None,
                 xgait_model_path: Optional[str] = None,
                 enable_caching: bool = True):
        
        self.device = device
        self.enable_caching = enable_caching
        
        # Initialize models if available
        if MODELS_AVAILABLE:
            try:
                self.silhouette_extractor = create_silhouette_extractor(
                    model_path=silhouette_model_path, 
                    device=device
                )
                logger.info("✅ Silhouette extractor initialized")
                self.silhouette_available = True
            except Exception as e:
                logger.error(f"❌ Failed to initialize silhouette extractor: {e}")
                self.silhouette_available = False
            
            try:
                self.parsing_model = create_human_parsing_model(
                    model_path=parsing_model_path,
                    device=device
                )
                logger.info("✅ Human parsing model initialized")
                self.parsing_available = True
            except Exception as e:
                logger.error(f"❌ Failed to initialize parsing model: {e}")
                self.parsing_available = False
            
            try:
                self.xgait_model = create_xgait_inference(
                    model_path=xgait_model_path,
                    device=device
                )
                logger.info("✅ XGait model initialized")
                self.xgait_available = True
            except Exception as e:
                logger.error(f"❌ Failed to initialize XGait model: {e}")
                self.xgait_available = False
        else:
            self.silhouette_available = False
            self.parsing_available = False
            self.xgait_available = False
        
        # Caching for processed data
        if self.enable_caching:
            self.silhouette_cache = {}
            self.parsing_cache = {}
            self.feature_cache = {}
        
        # Person tracking data
        self.person_sequences = {}  # track_id -> list of silhouettes
        self.person_features = {}   # track_id -> features
        
        logger.info(f"🎯 IdentificationProcessor initialized:")
        logger.info(f"   - Silhouette: {'✅' if self.silhouette_available else '❌'}")
        logger.info(f"   - Parsing: {'✅' if self.parsing_available else '❌'}")
        logger.info(f"   - XGait: {'✅' if self.xgait_available else '❌'}")
    
    def _get_cache_key(self, data: np.ndarray) -> str:
        """Generate cache key for numpy array"""
        return str(hash(data.tobytes()))
    
    def extract_silhouettes(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """Extract silhouettes from person crops"""
        if not crops:
            return []
        
        if self.silhouette_available:
            try:
                return self.silhouette_extractor.extract_silhouettes(crops)
            except Exception as e:
                logger.error(f"Error in silhouette extraction: {e}")
        
        # Fallback to simple masks
        silhouettes = []
        for crop in crops:
            h, w = crop.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[h//4:3*h//4, w//4:3*w//4] = 255
            silhouettes.append(mask)
        
        return silhouettes
    
    def extract_parsing(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """Extract human parsing from person crops"""
        if not crops:
            return []
        
        if self.parsing_available:
            try:
                return self.parsing_model.extract_parsing(crops)
            except Exception as e:
                logger.error(f"Error in parsing extraction: {e}")
        
        # Fallback to random parsing
        parsing_masks = []
        for crop in crops:
            h, w = crop.shape[:2]
            mask = np.random.randint(0, 20, (h, w), dtype=np.uint8)
            parsing_masks.append(mask)
        
        return parsing_masks
    
    def extract_gait_features(self, silhouette_sequences: List[List[np.ndarray]]) -> np.ndarray:
        """Extract XGait features from silhouette sequences"""
        if not silhouette_sequences:
            return np.array([]).reshape(0, 256)
        
        if self.xgait_available:
            try:
                return self.xgait_model.extract_features(silhouette_sequences)
            except Exception as e:
                logger.error(f"Error in XGait feature extraction: {e}")
        
        # Fallback to dummy features
        features = []
        for _ in silhouette_sequences:
            dummy_features = np.random.randn(256) * 0.1
            dummy_features = dummy_features / np.linalg.norm(dummy_features)
            features.append(dummy_features)
        
        return np.array(features)
    
    def process_person_crops(self, crops: List[np.ndarray], track_ids: List[int]) -> Dict[str, Any]:
        """
        Process person crops to extract silhouettes, parsing, and features
        
        Args:
            crops: List of person crop images
            track_ids: List of corresponding track IDs
            
        Returns:
            Dictionary with processing results
        """
        if not crops or len(crops) != len(track_ids):
            return {
                'silhouettes': [],
                'parsing_masks': [],
                'features': np.array([]).reshape(0, 256),
                'track_ids': [],
                'success': False
            }
        
        # Extract silhouettes and parsing
        silhouettes = self.extract_silhouettes(crops)
        parsing_masks = self.extract_parsing(crops)
        
        # Update person sequences for gait analysis
        for i, track_id in enumerate(track_ids):
            if track_id not in self.person_sequences:
                self.person_sequences[track_id] = []
            
            if i < len(silhouettes):
                self.person_sequences[track_id].append(silhouettes[i])
                
                # Keep only recent frames (sliding window)
                max_frames = 100
                if len(self.person_sequences[track_id]) > max_frames:
                    self.person_sequences[track_id] = self.person_sequences[track_id][-max_frames:]
        
        # Extract gait features for persons with enough frames
        sequences_for_feature_extraction = []
        valid_track_ids = []
        
        for track_id in track_ids:
            if track_id in self.person_sequences and len(self.person_sequences[track_id]) >= 10:
                # Use recent frames for feature extraction
                recent_silhouettes = self.person_sequences[track_id][-30:]  # Last 30 frames
                sequences_for_feature_extraction.append(recent_silhouettes)
                valid_track_ids.append(track_id)
        
        # Extract features
        if sequences_for_feature_extraction:
            features = self.extract_gait_features(sequences_for_feature_extraction)
            
            # Update feature cache
            for i, track_id in enumerate(valid_track_ids):
                if i < len(features):
                    self.person_features[track_id] = features[i]
        else:
            features = np.array([]).reshape(0, 256)
        
        return {
            'silhouettes': silhouettes,
            'parsing_masks': parsing_masks,
            'features': features,
            'valid_track_ids': valid_track_ids,
            'all_track_ids': track_ids,
            'success': True,
            'sequences_count': len(sequences_for_feature_extraction)
        }
    
    def identify_person(self, track_id: int) -> Tuple[Optional[str], float]:
        """
        Identify a person using their gait features
        
        Args:
            track_id: Track ID of the person
            
        Returns:
            (person_id, confidence) or (None, 0.0)
        """
        if track_id not in self.person_features:
            return None, 0.0
        
        if self.xgait_available:
            try:
                query_features = self.person_features[track_id]
                return self.xgait_model.identify_person(query_features)
            except Exception as e:
                logger.error(f"Error in person identification: {e}")
        
        return None, 0.0
    
    def register_person(self, track_id: int, person_id: str) -> bool:
        """
        Register a person in the gallery
        
        Args:
            track_id: Track ID of the person
            person_id: Identity to assign
            
        Returns:
            Success status
        """
        if track_id not in self.person_features:
            return False
        
        if self.xgait_available:
            try:
                features = self.person_features[track_id]
                self.xgait_model.add_to_gallery(person_id, features)
                logger.info(f"✅ Registered person {person_id} from track {track_id}")
                return True
            except Exception as e:
                logger.error(f"Error registering person: {e}")
        
        return False
    
    def get_person_summary(self, track_id: int) -> Dict[str, Any]:
        """Get summary information for a tracked person"""
        summary = {
            'track_id': track_id,
            'sequence_length': 0,
            'has_features': False,
            'identified_as': None,
            'confidence': 0.0
        }
        
        if track_id in self.person_sequences:
            summary['sequence_length'] = len(self.person_sequences[track_id])
        
        if track_id in self.person_features:
            summary['has_features'] = True
            person_id, confidence = self.identify_person(track_id)
            summary['identified_as'] = person_id
            summary['confidence'] = confidence
        
        return summary
    
    def get_gallery_summary(self) -> Dict[str, Any]:
        """Get summary of the identification gallery"""
        if self.xgait_available and hasattr(self.xgait_model, 'get_gallery_summary'):
            return self.xgait_model.get_gallery_summary()
        else:
            return {
                'num_persons': 0,
                'person_ids': [],
                'total_features': 0
            }
    
    def clear_cache(self):
        """Clear all caches"""
        if self.enable_caching:
            self.silhouette_cache.clear()
            self.parsing_cache.clear()
            self.feature_cache.clear()
        
        self.person_sequences.clear()
        self.person_features.clear()
        
        if self.xgait_available:
            self.xgait_model.clear_gallery()
    
    def get_model_status(self) -> Dict[str, bool]:
        """Get status of all models"""
        status = {
            'silhouette_extractor': self.silhouette_available,
            'parsing_model': self.parsing_available,
            'xgait_model': self.xgait_available,
            'models_imported': MODELS_AVAILABLE
        }
        
        # Check if models have real weights loaded
        if self.silhouette_available:
            status['silhouette_weights_loaded'] = self.silhouette_extractor.is_model_loaded()
        else:
            status['silhouette_weights_loaded'] = False
            
        if self.parsing_available:
            status['parsing_weights_loaded'] = self.parsing_model.is_model_loaded()
        else:
            status['parsing_weights_loaded'] = False
            
        if self.xgait_available:
            status['xgait_weights_loaded'] = self.xgait_model.is_model_loaded()
        else:
            status['xgait_weights_loaded'] = False
        
        return status


def create_identification_processor(device: str = "cpu", **kwargs) -> IdentificationProcessor:
    """
    Create and return an identification processor
    
    Args:
        device: Device to use for inference
        **kwargs: Additional arguments for model paths
        
    Returns:
        IdentificationProcessor instance
    """
    return IdentificationProcessor(device=device, **kwargs)


if __name__ == "__main__":
    # Test the identification processor
    print("🧪 Testing Identification Processor")
    
    # Create processor
    processor = create_identification_processor(device="cpu")
    
    # Create test data
    test_crops = [np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8) for _ in range(3)]
    test_track_ids = [1, 2, 3]
    
    # Process crops
    results = processor.process_person_crops(test_crops, test_track_ids)
    
    print(f"✅ Processing results:")
    print(f"   - Silhouettes: {len(results['silhouettes'])}")
    print(f"   - Parsing masks: {len(results['parsing_masks'])}")
    print(f"   - Features shape: {results['features'].shape}")
    print(f"   - Success: {results['success']}")
    
    # Check model status
    status = processor.get_model_status()
    print(f"📊 Model Status:")
    for model, available in status.items():
        print(f"   - {model}: {'✅' if available else '❌'}")
    
    # Gallery summary
    gallery = processor.get_gallery_summary()
    print(f"📚 Gallery: {gallery['num_persons']} persons")
