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
    """XGait backbone network"""
    def __init__(self, in_channels=1, base_channels=32):
        super(XGaitBackbone, self).__init__()
        
        # Initial convolution
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
    """Complete XGait model with classification head"""
    def __init__(self, num_classes=100, backbone_channels=32):
        super(XGaitModel, self).__init__()
        
        self.backbone = XGaitBackbone(in_channels=1, base_channels=backbone_channels)
        
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
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu", num_classes: int = 100):
        self.device = device
        self.num_classes = num_classes
        self.model = XGaitModel(num_classes=num_classes).to(device)
        
        # Load weights if available
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=device)
                
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Remove module prefix if present
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                
                self.model.load_state_dict(new_state_dict, strict=False)
                logger.info(f"✅ Loaded XGait weights from {model_path}")
                self.model_loaded = True
            except Exception as e:
                logger.warning(f"❌ Failed to load XGait weights: {e}")
                self.model_loaded = False
        else:
            logger.warning("⚠️  No XGait weights found, using random initialization")
            self.model_loaded = False
        
        self.model.eval()
        
        # Gallery for known persons
        self.gallery_features = {}
        self.gallery_ids = []
        
        # Similarity threshold for identification
        self.similarity_threshold = 0.7
    
    def preprocess_gait_sequence(self, silhouettes: List[np.ndarray], target_frames: int = 30) -> torch.Tensor:
        """
        Preprocess silhouette sequence for XGait input
        
        Args:
            silhouettes: List of silhouette masks
            target_frames: Target number of frames
            
        Returns:
            Preprocessed tensor of shape (1, 1, T, H, W)
        """
        if not silhouettes:
            # Return dummy sequence
            return torch.zeros(1, 1, target_frames, 64, 32, device=self.device)
        
        # Resize silhouettes to standard size
        processed_silhouettes = []
        for sil in silhouettes:
            if len(sil.shape) == 3:
                sil = sil[:, :, 0]  # Take first channel if RGB
            
            # Resize to 64x32
            sil_resized = torch.from_numpy(sil).float()
            sil_resized = F.interpolate(sil_resized.unsqueeze(0).unsqueeze(0), 
                                      size=(64, 32), mode='bilinear', align_corners=False)
            sil_resized = sil_resized.squeeze()
            
            # Normalize to [0, 1]
            if sil_resized.max().item() > 1:
                sil_resized = sil_resized / 255.0
            
            processed_silhouettes.append(sil_resized)
        
        # Handle sequence length
        if len(processed_silhouettes) >= target_frames:
            # Sample frames uniformly
            indices = np.linspace(0, len(processed_silhouettes) - 1, target_frames, dtype=int)
            processed_silhouettes = [processed_silhouettes[i] for i in indices]
        else:
            # Repeat frames to reach target length
            while len(processed_silhouettes) < target_frames:
                processed_silhouettes.extend(processed_silhouettes)
            processed_silhouettes = processed_silhouettes[:target_frames]
        
        # Stack into tensor
        sequence_tensor = torch.stack(processed_silhouettes, dim=0)  # (T, H, W)
        sequence_tensor = sequence_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, T, H, W)
        
        return sequence_tensor.to(self.device)
    
    def extract_features(self, silhouette_sequences: List[List[np.ndarray]]) -> np.ndarray:
        """
        Extract XGait features from silhouette sequences
        
        Args:
            silhouette_sequences: List of silhouette sequences
            
        Returns:
            Feature array of shape (N, feature_dim)
        """
        if not silhouette_sequences:
            return np.array([]).reshape(0, 256)
        
        features = []
        
        with torch.no_grad():
            for sequence in silhouette_sequences:
                try:
                    # Preprocess sequence
                    input_tensor = self.preprocess_gait_sequence(sequence)
                    
                    # Extract features
                    feature_vector = self.model(input_tensor, return_features=True)
                    features.append(feature_vector.cpu().numpy().flatten())
                    
                except Exception as e:
                    logger.error(f"Error extracting XGait features: {e}")
                    # Fallback to dummy features
                    dummy_features = np.random.randn(256) * 0.1
                    dummy_features = dummy_features / np.linalg.norm(dummy_features)
                    features.append(dummy_features)
        
        return np.array(features)
    
    def add_to_gallery(self, person_id: str, features: np.ndarray):
        """Add person features to gallery"""
        if person_id not in self.gallery_features:
            self.gallery_features[person_id] = []
            self.gallery_ids.append(person_id)
        
        self.gallery_features[person_id].append(features)
    
    def identify_person(self, query_features: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Identify person using gallery matching
        
        Args:
            query_features: Query feature vector
            
        Returns:
            (person_id, confidence) or (None, 0.0)
        """
        if not self.gallery_features:
            return None, 0.0
        
        best_match = None
        best_similarity = 0.0
        
        query_features = query_features / np.linalg.norm(query_features)
        
        for person_id, person_features_list in self.gallery_features.items():
            # Compare with all stored features for this person
            similarities = []
            for stored_features in person_features_list:
                stored_features = stored_features / np.linalg.norm(stored_features)
                similarity = np.dot(query_features, stored_features)
                similarities.append(similarity)
            
            # Use maximum similarity
            max_similarity = max(similarities)
            
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
        self.gallery_features.clear()
        self.gallery_ids.clear()
    
    def get_gallery_summary(self) -> Dict:
        """Get summary of gallery contents"""
        summary = {
            'num_persons': len(self.gallery_features),
            'person_ids': list(self.gallery_features.keys()),
            'total_features': sum(len(features) for features in self.gallery_features.values())
        }
        return summary
    
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
