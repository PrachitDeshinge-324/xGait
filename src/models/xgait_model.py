"""
XGait Model Inference Wrapper
Loads pre-trained XGait model and performs person identification inference
"""
import sys
import os
from pathlib import Path

# Add parent directory to path for utils and config imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
root_dir = src_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional
import cv2

from utils.device_utils import DeviceManager
from config import get_device_config
from xgait import XGait
from modules import *

class XGaitInference:
    """
    XGait model wrapper for inference-only person identification
    """
    def __init__(self, 
                 device: str = "mps", 
                 model_path: str = "weights/Gait3D-XGait-120000.pt",
                 config: Optional[Dict] = None):
        
        self.device = device
        self.device_config = get_device_config(device)
        self.device_manager = DeviceManager(device, self.device_config["dtype"])
        
        # Default XGait configuration
        self.model_config = config or self._get_default_config()
        
        # Initialize XGait model
        self.model = XGait(self.model_config, training=False)
        
        # Load pretrained weights
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            # Load weights with strict=False to handle missing keys
            missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️  Missing keys in checkpoint: {len(missing_keys)}")
            if unexpected_keys:
                print(f"⚠️  Unexpected keys in checkpoint: {len(unexpected_keys)}")
                
            print(f"✅ Loaded XGait model from {model_path}")
        else:
            print(f"⚠️  Model file not found: {model_path}")
            print("   Using randomly initialized XGait model")
        
        # Prepare model with device optimizations
        self.model = self.device_manager.prepare_model(self.model)
        self.model = self.device_manager.optimize_inference(self.model)
        
        # Feature storage for person gallery
        self.person_gallery: Dict[int, torch.Tensor] = {}
        self.person_metadata: Dict[int, Dict] = {}
        
        print(f"✅ XGaitInference initialized")
        print(f"   Device: {device}")
        print(f"   Dtype: {self.device_config['dtype']}")
        print(f"   Autocast: {self.device_config['autocast']}")
    
    def _get_default_config(self) -> Dict:
        """Get default XGait model configuration"""
        return {
            'model': 'XGait',
            'backbone_cfg': {
                'type': 'ResNet9',
                'block': 'BasicBlock',
                'channels': [64, 128, 256, 512],
                'layers': [1, 1, 1, 1],
                'strides': [1, 2, 2, 1],
                'maxpool': False
            },
            'SeparateFCs': {
                'in_channels': 512,
                'out_channels': 256,
                'parts_num': 31
            },
            'SeparateBNNecks': {
                'class_num': 3000,  # Large enough for any dataset
                'in_channels': 256,
                'parts_num': 31
            },
            'CALayers': {
                'channels': 512,
                'reduction': 8
            },
            'CALayersP': {
                'channels': 512,
                'reduction': 8,
                'with_max_pool': True
            },
            'bin_num': [1, 2, 4, 8, 16]
        }
    
    def preprocess_inputs(self, 
                         silhouettes: List[np.ndarray], 
                         parsing_maps: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess silhouettes and parsing maps for XGait input
        
        Args:
            silhouettes: List of silhouette images (H, W)
            parsing_maps: List of parsing maps (H, W)
            
        Returns:
            Tuple of (parsing_tensor, silhouette_tensor)
        """
        if not silhouettes or not parsing_maps:
            empty_tensor = torch.empty(0, 1, 64, 44, device=self.device, dtype=self.device_config["dtype"])
            return empty_tensor, empty_tensor
        
        # Convert to tensors
        sil_tensors = []
        par_tensors = []
        
        for sil, par in zip(silhouettes, parsing_maps):
            # Normalize silhouettes to [0, 1]
            if sil.max() > 1:
                sil = sil.astype(np.float32) / 255.0
            else:
                sil = sil.astype(np.float32)
            
            # Normalize parsing maps to [0, 1] range for each class
            par = par.astype(np.float32) / 19.0  # 20 classes (0-19)
            
            # Add channel dimension
            sil_tensor = torch.from_numpy(sil).unsqueeze(0)  # (1, H, W)
            par_tensor = torch.from_numpy(par).unsqueeze(0)  # (1, H, W)
            
            sil_tensors.append(sil_tensor)
            par_tensors.append(par_tensor)
        
        # Stack into batch tensors
        sil_batch = torch.stack(sil_tensors).unsqueeze(1)  # (N, 1, 1, H, W)
        par_batch = torch.stack(par_tensors).unsqueeze(1)  # (N, 1, 1, H, W)
        
        # Prepare for device
        sil_batch = self.device_manager.prepare_tensor(sil_batch)
        par_batch = self.device_manager.prepare_tensor(par_batch)
        
        return par_batch, sil_batch
    
    def extract_features(self, 
                        silhouettes: List[np.ndarray], 
                        parsing_maps: List[np.ndarray]) -> torch.Tensor:
        """
        Extract XGait features from silhouettes and parsing maps
        
        Args:
            silhouettes: List of silhouette images
            parsing_maps: List of human parsing maps
            
        Returns:
            Feature tensor (N, feature_dim)
        """
        if not silhouettes or not parsing_maps:
            return torch.empty(0, self.model_config['SeparateFCs']['out_channels'] * 4, 
                             device=self.device, dtype=self.device_config["dtype"])
        
        # Preprocess inputs
        par_batch, sil_batch = self.preprocess_inputs(silhouettes, parsing_maps)
        
        if par_batch.numel() == 0:
            return torch.empty(0, self.model_config['SeparateFCs']['out_channels'] * 4, 
                             device=self.device, dtype=self.device_config["dtype"])
        
        # Create dummy labels and sequence lengths for inference
        batch_size = par_batch.shape[0]
        dummy_labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        dummy_types = [''] * batch_size
        dummy_views = [''] * batch_size
        sequence_lengths = torch.ones(batch_size, dtype=torch.long, device=self.device)
        
        # Prepare inputs in XGait format
        inputs = ([par_batch, sil_batch], dummy_labels, dummy_types, dummy_views, sequence_lengths)
        
        # Extract features with autocast if supported
        with self.device_manager.autocast_context():
            with torch.no_grad():
                outputs = self.model(inputs)
                features = outputs['inference_feat']['embeddings']
        
        return features
    
    def identify_person(self, 
                       silhouettes: List[np.ndarray], 
                       parsing_maps: List[np.ndarray],
                       track_id: int,
                       threshold: float = 0.6) -> Tuple[Optional[int], float]:
        """
        Identify a person using XGait features
        
        Args:
            silhouettes: List of silhouette images for the person
            parsing_maps: List of parsing maps for the person
            track_id: Current track ID
            threshold: Similarity threshold for identification
            
        Returns:
            Tuple of (identified_person_id, confidence_score)
        """
        # Extract features for current person
        current_features = self.extract_features(silhouettes, parsing_maps)
        
        if current_features.numel() == 0:
            return None, 0.0
        
        # Average features across sequence
        if current_features.shape[0] > 1:
            current_features = torch.mean(current_features, dim=0, keepdim=True)
        
        # If no gallery, add this person as new identity
        if not self.person_gallery:
            person_id = 1
            self.person_gallery[person_id] = current_features.clone()
            self.person_metadata[person_id] = {
                'first_seen_track': track_id,
                'feature_count': 1
            }
            return person_id, 1.0
        
        # Compare with existing identities
        best_similarity = -1.0
        best_person_id = None
        
        for person_id, gallery_features in self.person_gallery.items():
            # Compute cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                current_features, gallery_features, dim=1
            ).item()
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_person_id = person_id
        
        # Check if similarity is above threshold
        if best_similarity >= threshold:
            # Update gallery features with exponential moving average
            alpha = 0.3  # Update rate
            self.person_gallery[best_person_id] = (
                alpha * current_features + (1 - alpha) * self.person_gallery[best_person_id]
            )
            self.person_metadata[best_person_id]['feature_count'] += 1
            return best_person_id, best_similarity
        else:
            # Create new identity
            new_person_id = max(self.person_gallery.keys()) + 1
            self.person_gallery[new_person_id] = current_features.clone()
            self.person_metadata[new_person_id] = {
                'first_seen_track': track_id,
                'feature_count': 1
            }
            return new_person_id, best_similarity
    
    def identify_persons_batch(self, 
                              silhouettes_batch: List[List[np.ndarray]], 
                              parsing_maps_batch: List[List[np.ndarray]],
                              track_ids: List[int],
                              threshold: float = 0.6) -> List[Tuple[Optional[int], float]]:
        """
        Identify multiple persons in parallel
        
        Args:
            silhouettes_batch: List of silhouette lists (one per track)
            parsing_maps_batch: List of parsing map lists (one per track)
            track_ids: List of track IDs
            threshold: Similarity threshold for identification
            
        Returns:
            List of (person_id, confidence) tuples
        """
        results = []
        
        for silhouettes, parsing_maps, track_id in zip(silhouettes_batch, parsing_maps_batch, track_ids):
            person_id, confidence = self.identify_person(silhouettes, parsing_maps, track_id, threshold)
            results.append((person_id, confidence))
        
        return results
    
    def get_gallery_stats(self) -> Dict:
        """Get statistics about the person gallery"""
        return {
            'total_identities': len(self.person_gallery),
            'identities': {
                person_id: {
                    'feature_count': metadata['feature_count'],
                    'first_seen_track': metadata['first_seen_track']
                }
                for person_id, metadata in self.person_metadata.items()
            }
        }
    
    def clear_gallery(self):
        """Clear the person gallery"""
        self.person_gallery.clear()
        self.person_metadata.clear()
    
    def save_gallery(self, filepath: str):
        """Save person gallery to file"""
        torch.save({
            'gallery': self.person_gallery,
            'metadata': self.person_metadata
        }, filepath)
        print(f"Gallery saved to {filepath}")
    
    def load_gallery(self, filepath: str):
        """Load person gallery from file"""
        if os.path.exists(filepath):
            data = torch.load(filepath, map_location=self.device)
            self.person_gallery = data['gallery']
            self.person_metadata = data['metadata']
            print(f"Gallery loaded from {filepath}")
        else:
            print(f"Gallery file not found: {filepath}")

def create_xgait_inference(device: str = "mps", 
                          model_path: str = "weights/Gait3D-XGait-120000.pt",
                          config: Optional[Dict] = None) -> XGaitInference:
    """Create an XGaitInference instance"""
    return XGaitInference(device=device, model_path=model_path, config=config)
