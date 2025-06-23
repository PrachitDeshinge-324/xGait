"""
XGait Model Implementation
Cross-view Gait Recognition using 3D CNNs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path
from typing import List, Union, Optional, Tuple, Dict
import logging
from config import xgaitConfig

logger = logging.getLogger(__name__)


class Temporal3DConv(nn.Module):
    """3D Convolution block for temporal modeling"""
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)):
        super(Temporal3DConv, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv3d(x)))


class SpatialTemporalBlock(nn.Module):
    """Spatial-Temporal block for XGait"""
    def __init__(self, in_channels, out_channels):
        super(SpatialTemporalBlock, self).__init__()
        
        self.conv1 = Temporal3DConv(in_channels, out_channels // 2, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv2 = Temporal3DConv(out_channels // 2, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        return out + identity


class TemporalPooling(nn.Module):
    """Temporal pooling module"""
    def __init__(self, pooling_type='max'):
        super(TemporalPooling, self).__init__()
        self.pooling_type = pooling_type
    
    def forward(self, x):
        # x: (N, C, T, H, W)
        if self.pooling_type == 'max':
            return torch.max(x, dim=2)[0]  # (N, C, H, W)
        elif self.pooling_type == 'avg':
            return torch.mean(x, dim=2)   # (N, C, H, W)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")


class XGaitBackbone(nn.Module):
    """XGait backbone network - enhanced for multi-channel input"""
    def __init__(self, in_channels=1, base_channels=32):
        super(XGaitBackbone, self).__init__()
        
        # Initial convolution - now supports both single channel (silhouettes) and multi-channel (sil+parsing)
        self.conv1 = Temporal3DConv(in_channels, base_channels, kernel_size=(1, 5, 5), padding=(0, 2, 2))
        
        # Spatial-temporal blocks
        self.st_block1 = SpatialTemporalBlock(base_channels, base_channels * 2)
        self.st_block2 = SpatialTemporalBlock(base_channels * 2, base_channels * 4)
        self.st_block3 = SpatialTemporalBlock(base_channels * 4, base_channels * 8)
        
        # Pooling layers
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # Temporal pooling
        self.temporal_pool = TemporalPooling('max')
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature dimension
        self.feature_dim = base_channels * 8
    
    def forward(self, x):
        # x: (N, C, T, H, W)
        
        # Initial convolution
        x = self.conv1(x)  # (N, 32, T, H, W)
        
        # ST blocks with pooling
        x = self.st_block1(x)   # (N, 64, T, H, W)
        x = self.pool1(x)       # (N, 64, T, H/2, W/2)
        
        x = self.st_block2(x)   # (N, 128, T, H/2, W/2)
        x = self.pool2(x)       # (N, 128, T, H/4, W/4)
        
        x = self.st_block3(x)   # (N, 256, T, H/4, W/4)
        x = self.pool3(x)       # (N, 256, T, H/8, W/8)
        
        # Temporal pooling
        x = self.temporal_pool(x)  # (N, 256, H/8, W/8)
        
        # Global pooling
        x = self.global_pool(x)    # (N, 256, 1, 1)
        
        return x.view(x.size(0), -1)  # (N, 256)


class XGaitModel(nn.Module):
    """Complete XGait model with classification head - supports multi-channel input"""
    def __init__(self, num_classes=100, backbone_channels=32, in_channels=1):
        super(XGaitModel, self).__init__()
        
        # Support both single-channel (silhouettes) and multi-channel (silhouettes + parsing)
        self.backbone = XGaitBackbone(in_channels=in_channels, base_channels=backbone_channels)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Feature extractor (without classification)
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.backbone.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)
        )
    
    def forward(self, x, return_features=False):
        # Extract backbone features
        features = self.backbone(x)
        
        if return_features:
            # Return normalized features for similarity computation
            normalized_features = F.normalize(self.feature_extractor(features), p=2, dim=1)
            return normalized_features
        else:
            # Return classification logits
            return self.classifier(features)


class XGaitInference:
    """
    XGait inference engine for gait recognition
    Enhanced for proper sequence handling and gait parsing integration
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu", num_classes: int = 100):
        self.device = device
        self.num_classes = num_classes
        
        # Support adaptive input channels - will be determined by input data
        self.model = XGaitModel(num_classes=num_classes, in_channels=2).to(self.device)  # Default to 2 channels for sil+parsing
        self.fallback_model = XGaitModel(num_classes=num_classes, in_channels=1).to(self.device)  # Fallback for silhouettes only
        
        # Sequence parameters for proper XGait input
        self.min_sequence_length = xgaitConfig.min_sequence_length  # Minimum frames for reliable gait analysis
        self.target_sequence_length = 30  # Target sequence length for XGait
        self.sequence_stride = 2  # Frame stride for sequence sampling
        
        # Load weights if available
        if model_path and os.path.exists(model_path):
            self.model_loaded = self.load_model_weights(self.model, model_path)
            if not self.model_loaded:
                logger.warning("⚠️ Fallback to random initialization for main model.")
        else:
            logger.warning("⚠️ No weights found, using random initialization.")
            self.model_loaded = False
        
        # Fallback model loading (optional, based on use case)
        if self.model_loaded and self.fallback_model:
            self.load_model_weights(self.fallback_model, model_path)
        else:
            logger.warning("⚠️ Fallback model not initialized or weights unavailable.")
        
        # Set both models to eval mode
        self.model.eval()
        self.fallback_model.eval()
        
        # Initialize gallery manager (will be set externally)
        self.gallery_manager = None
        
        # Similarity threshold for identification (fallback)
        self.similarity_threshold = xgaitConfig.similarity_threshold
    
    def load_model_weights(self, model, model_path: str):
        """
        Load weights into the given model from the specified path.

        Args:
            model: PyTorch model instance.
            model_path: Path to the checkpoint file.

        Returns:
            bool: True if weights are loaded successfully, False otherwise.
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            # Expect standardized checkpoint format
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))

            # Remove module prefix if present
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            model.load_state_dict(new_state_dict, strict=False)
            logger.info(f"✅ Loaded weights from {model_path}")
            return True
        except Exception as e:
            logger.warning(f"❌ Failed to load weights from {model_path}: {e}")
            return False

    def preprocess_gait_sequence_with_parsing(self, silhouettes: List[np.ndarray], 
                                            parsing_masks: List[np.ndarray] = None) -> torch.Tensor:
        """
        Preprocess silhouette sequence with optional parsing masks for XGait input
        
        Args:
            silhouettes: List of silhouette masks (temporal sequence)
            parsing_masks: Optional list of parsing masks for enhanced input
            
        Returns:
            Preprocessed tensor of shape (1, C, T, H, W) where C=1 for silhouettes or C=2 for sil+parsing
        """
        if not silhouettes:
            # Return dummy sequence
            channels = 2 if parsing_masks else 1
            return torch.zeros(1, channels, self.target_sequence_length, 64, 32, device=self.device)
        
        # Determine if we're using parsing-enhanced input
        use_parsing = parsing_masks is not None and len(parsing_masks) == len(silhouettes)
        channels = 2 if use_parsing else 1
        
        # Process silhouettes and parsing masks
        processed_frames = []
        
        for i, sil in enumerate(silhouettes):
            if len(sil.shape) == 3:
                sil = sil[:, :, 0]  # Take first channel if RGB
            
            # Resize silhouette to 64x32
            sil_resized = torch.from_numpy(sil).float()
            sil_resized = F.interpolate(sil_resized.unsqueeze(0).unsqueeze(0), 
                                      size=(64, 32), mode='bilinear', align_corners=False)
            sil_resized = sil_resized.squeeze()
            
            # Normalize to [0, 1]
            if sil_resized.max().item() > 1:
                sil_resized = sil_resized / 255.0
            
            if use_parsing:
                # Process parsing mask
                parsing = parsing_masks[i]
                if len(parsing.shape) == 3:
                    parsing = parsing[:, :, 0]
                
                parsing_resized = torch.from_numpy(parsing).float()
                parsing_resized = F.interpolate(parsing_resized.unsqueeze(0).unsqueeze(0),
                                              size=(64, 32), mode='nearest')
                parsing_resized = parsing_resized.squeeze()
                
                # Normalize parsing mask
                if parsing_resized.max().item() > 1:
                    parsing_resized = parsing_resized / 7.0  # 7 classes in gait parsing
                
                # Combine silhouette and parsing
                frame = torch.stack([sil_resized, parsing_resized], dim=0)  # (2, 64, 32)
            else:
                frame = sil_resized.unsqueeze(0)  # (1, 64, 32)
            
            processed_frames.append(frame)
        
        # Handle sequence length - ensure we have enough frames for gait analysis
        if len(processed_frames) >= self.target_sequence_length:
            # Sample frames uniformly with stride
            indices = np.arange(0, len(processed_frames), self.sequence_stride)
            if len(indices) > self.target_sequence_length:
                indices = np.linspace(0, len(processed_frames) - 1, self.target_sequence_length, dtype=int)
            processed_frames = [processed_frames[i] for i in indices[:self.target_sequence_length]]
        else:
            # If we don't have enough frames, repeat the sequence cyclically
            while len(processed_frames) < self.target_sequence_length:
                processed_frames.extend(processed_frames)
            processed_frames = processed_frames[:self.target_sequence_length]
        
        # Stack into tensor (C, T, H, W)
        sequence_tensor = torch.stack(processed_frames, dim=1)  # (C, T, H, W)
        sequence_tensor = sequence_tensor.unsqueeze(0)  # (1, C, T, H, W)
        
        return sequence_tensor.to(self.device)

    def preprocess_gait_sequence(self, silhouettes: List[np.ndarray], target_frames: int = 30) -> torch.Tensor:
        """
        Preprocess silhouette sequence for XGait input (legacy method)
        
        Args:
            silhouettes: List of silhouette masks
            target_frames: Target number of frames
            
        Returns:
            Preprocessed tensor of shape (1, 1, T, H, W)
        """
        # Use the enhanced method with no parsing masks
        return self.preprocess_gait_sequence_with_parsing(silhouettes, None)

    def extract_features_from_sequence(self, silhouettes: List[np.ndarray], 
                                     parsing_masks: List[np.ndarray] = None) -> np.ndarray:
        """
        Extract XGait features from a single silhouette sequence with optional parsing
        
        Args:
            silhouettes: List of silhouette masks (temporal sequence)
            parsing_masks: Optional list of parsing masks
            
        Returns:
            Feature vector of shape (feature_dim,)
        """
        if len(silhouettes) < self.min_sequence_length:
            logger.warning(f"Sequence too short: {len(silhouettes)} < {self.min_sequence_length}")
            # Return dummy features
            dummy_features = np.random.randn(256) * 0.1
            return dummy_features / np.linalg.norm(dummy_features)
        
        try:
            with torch.no_grad():
                # Preprocess sequence with parsing if available
                input_tensor = self.preprocess_gait_sequence_with_parsing(silhouettes, parsing_masks)
                
                # Choose appropriate model based on input channels
                use_parsing = parsing_masks is not None and len(parsing_masks) == len(silhouettes)
                model_to_use = self.model if use_parsing else self.fallback_model
                
                # Extract features
                feature_vector = model_to_use(input_tensor, return_features=True)
                return feature_vector.cpu().numpy().flatten()
                
        except Exception as e:
            logger.error(f"Error extracting XGait features: {e}")
            # Fallback to dummy features
            dummy_features = np.random.randn(256) * 0.1
            return dummy_features / np.linalg.norm(dummy_features)
    
    def extract_features(self, silhouette_sequences: List[List[np.ndarray]], 
                        parsing_sequences: List[List[np.ndarray]] = None) -> np.ndarray:
        """
        Extract XGait features from multiple silhouette sequences
        
        Args:
            silhouette_sequences: List of silhouette sequences
            parsing_sequences: Optional list of parsing sequences
            
        Returns:
            Feature array of shape (N, feature_dim)
        """
        if not silhouette_sequences:
            return np.array([]).reshape(0, 256)
        
        features = []
        
        for i, sequence in enumerate(silhouette_sequences):
            parsing_seq = parsing_sequences[i] if parsing_sequences and i < len(parsing_sequences) else None
            feature_vector = self.extract_features_from_sequence(sequence, parsing_seq)
            features.append(feature_vector)
        
        return np.array(features)
    
    def set_gallery_manager(self, gallery_manager):
        """Set the gallery manager for advanced gallery functionality"""
        self.gallery_manager = gallery_manager
    
    def add_to_gallery(self, person_id: str, features: np.ndarray, track_id: Optional[int] = None):
        """Add person features to gallery"""
        if self.gallery_manager:
            return self.gallery_manager.add_person(person_id, features, track_id)
        else:
            # Fallback to simple gallery
            if person_id not in getattr(self, 'gallery_features', {}):
                if not hasattr(self, 'gallery_features'):
                    self.gallery_features = {}
                self.gallery_features[person_id] = []
            self.gallery_features[person_id].append(features)
            return person_id
    
    def identify_person(self, query_features: np.ndarray, track_id: Optional[int] = None) -> Tuple[Optional[str], float]:
        """
        Identify person using gallery matching
        
        Args:
            query_features: Query feature vector
            track_id: Optional track ID for advanced matching
            
        Returns:
            (person_id, confidence) or (None, 0.0)
        """
        if self.gallery_manager:
            # Use advanced gallery manager
            person_id, confidence, metadata = self.gallery_manager.identify_person(
                query_features, track_id=track_id, auto_add=True
            )
            return person_id, confidence
        else:
            # Fallback to simple matching
            gallery_features = getattr(self, 'gallery_features', {})
            if not gallery_features:
                return None, 0.0
            
            best_match = None
            best_similarity = 0.0
            
            query_features = query_features / (np.linalg.norm(query_features) + 1e-8)
            
            for person_id, person_features_list in gallery_features.items():
                # Compare with all stored features for this person
                similarities = []
                for stored_features in person_features_list:
                    stored_features = stored_features / (np.linalg.norm(stored_features) + 1e-8)
                    similarity = np.dot(query_features, stored_features)
                    similarities.append(similarity)
                
                # Use maximum similarity
                max_similarity = max(similarities) if similarities else 0.0
                
                if max_similarity > best_similarity:
                    best_similarity = max_similarity
                    best_match = person_id
            
            # Check threshold
            if best_similarity >= self.similarity_threshold:
                return best_match, best_similarity
            else:
                return None, best_similarity
    
    def clear_gallery(self):
        """Clear the gallery"""
        if self.gallery_manager:
            self.gallery_manager._initialize_empty_gallery()
        else:
            if hasattr(self, 'gallery_features'):
                self.gallery_features.clear()
    
    def get_gallery_summary(self) -> Dict:
        """Get summary of gallery contents"""
        if self.gallery_manager:
            return self.gallery_manager.get_gallery_summary()
        else:
            gallery_features = getattr(self, 'gallery_features', {})
            return {
                'num_persons': len(gallery_features),
                'person_ids': list(gallery_features.keys()),
                'total_features': sum(len(features) for features in gallery_features.values())
            }
    
    def is_model_loaded(self) -> bool:
        """Check if real model weights are loaded"""
        return self.model_loaded


def create_xgait_inference(model_path: Optional[str] = None, device: str = "cpu", num_classes: int = 100) -> XGaitInference:
    """
    Create and return an XGait inference engine
    
    Args:
        model_path: Path to XGait model weights
        device: Device to use for inference
        num_classes: Number of identity classes
        
    Returns:
        XGaitInference instance
    """
    # Look for default model path in weights directory
    if model_path is None:
        current_dir = Path(__file__).parent.parent.parent
        potential_paths = [
            current_dir / "weights" / "Gait3D-XGait-120000.pt",
            current_dir / "weights" / "xgait_model.pth",
            current_dir / "weights" / "xgait.pt"
        ]
        
        for path in potential_paths:
            if path.exists():
                model_path = str(path)
                break
    
    return XGaitInference(model_path=model_path, device=device, num_classes=num_classes)


if __name__ == "__main__":
    # Test the XGait model
    print("🧪 Testing XGait Model")
    
    # Create test data
    test_silhouettes = [np.random.randint(0, 255, (64, 32), dtype=np.uint8) for _ in range(30)]
    
    # Create XGait inference
    xgait = create_xgait_inference(device="cpu")
    
    # Extract features
    features = xgait.extract_features([test_silhouettes])
    
    print(f"✅ Extracted features: shape {features.shape}")
    print(f"📊 Model loaded: {xgait.is_model_loaded()}")
    print(f"🎯 Feature dimension: {features.shape[1] if len(features) > 0 else 'N/A'}")
    
    # Test gallery functionality
    if len(features) > 0:
        xgait.add_to_gallery("person_1", features[0])
        person_id, confidence = xgait.identify_person(features[0])
        print(f"🔍 Identification test: {person_id} (confidence: {confidence:.3f})")
        print(f"📚 Gallery summary: {xgait.get_gallery_summary()}")
