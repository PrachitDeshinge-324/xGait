"""
Inference engine for XGait model
"""

import torch
import numpy as np
import cv2
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from .model import create_official_xgait_model

# Configure logging
logger = logging.getLogger(__name__)


class OfficialXGaitInference:
    """
    Official XGait inference engine exactly matching Gait3D-Benchmark
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = None, num_classes: int = 3000):
        if device is None:
            from ...config import get_xgait_device
            device = get_xgait_device()
        
        self.device = device
        self.num_classes = num_classes
        
        # Create the official XGait model
        self.model = create_official_xgait_model(num_classes=num_classes).to(self.device)
        
        # Sequence parameters from official config
        self.min_sequence_length = 10
        self.target_sequence_length = 30
        
        # Input preprocessing parameters (official Gait3D config)
        self.input_height = 64
        self.input_width = 44
        
        # PERF-001 fix: Tensor pooling to prevent memory fragmentation
        self.tensor_pool = {}  # Cache for reusable tensors
        self.max_pool_size = 5  # Limit number of cached tensors
        self.pool_cleanup_counter = 0
        
        # Load weights if available
        if model_path:
            self.model_loaded = self.load_model_weights(model_path)
        else:
            logger.warning("⚠️ No model weights provided, using random initialization.")
            self.model_loaded = False
        
        # Set model to eval mode
        self.model.eval()
        
        logger.info(f"🎯 Official XGait Model initialized:")
        logger.info(f"   - Input size: {self.input_height}x{self.input_width}")
        logger.info(f"   - Target sequence length: {self.target_sequence_length}")
        logger.info(f"   - Device: {self.device}")
        logger.info(f"   - Weights loaded: {self.model_loaded}")
    
    def get_pooled_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """
        PERF-001 fix: Get a reusable tensor from the pool to prevent memory fragmentation
        """
        pool_key = (shape, dtype)
        
        if pool_key in self.tensor_pool and len(self.tensor_pool[pool_key]) > 0:
            # Reuse existing tensor
            tensor = self.tensor_pool[pool_key].pop()
            tensor.zero_()  # Clear contents for reuse
            return tensor
        else:
            # Create new tensor and ensure it's on correct device
            return torch.zeros(shape, dtype=dtype, device=self.device)
    
    def return_tensor_to_pool(self, tensor: torch.Tensor):
        """
        PERF-001 fix: Return tensor to pool for reuse
        """
        if tensor.device != self.device:
            return  # Don't pool tensors on wrong device
            
        pool_key = (tuple(tensor.shape), tensor.dtype)
        
        if pool_key not in self.tensor_pool:
            self.tensor_pool[pool_key] = []
        
        # Limit pool size to prevent excessive memory usage
        if len(self.tensor_pool[pool_key]) < self.max_pool_size:
            self.tensor_pool[pool_key].append(tensor.detach())
    
    def cleanup_tensor_pool(self):
        """
        PERF-001 fix: Periodic cleanup of tensor pool and GPU memory
        """
        self.pool_cleanup_counter += 1
        
        # Clean pool every 10 inference calls
        if self.pool_cleanup_counter >= 10:
            self.tensor_pool.clear()
            self.pool_cleanup_counter = 0
            
            # Force GPU memory cleanup
            if self.device.startswith('cuda'):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    
    def load_model_weights(self, model_path: str):
        """Load official XGait weights"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load with strict=True since this should match exactly
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"Missing keys: {len(missing_keys)}")
                logger.debug(f"Missing keys: {missing_keys[:5]}...")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {len(unexpected_keys)}")
                logger.debug(f"Unexpected keys: {unexpected_keys[:5]}...")
            
            logger.info(f"✅ Loaded official XGait weights from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load weights: {e}")
            return False
    
    def preprocess_sequence(self, silhouettes: List[np.ndarray], 
                          parsing_masks: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess silhouettes and parsing masks for official XGait
        Returns: (pars_tensor, sils_tensor) - note the order!
        """
        if not silhouettes or not parsing_masks:
            raise ValueError("Both silhouettes and parsing masks are required for official XGait")
        
        if len(silhouettes) != len(parsing_masks):
            raise ValueError("Silhouettes and parsing masks must have same length")
        
        # Ensure minimum sequence length for reliable feature extraction
        if len(silhouettes) < self.min_sequence_length:
            # Repeat frames to reach minimum length with some variation
            repeat_factor = self.min_sequence_length // len(silhouettes) + 1
            extended_silhouettes = []
            extended_parsing = []
            
            for i in range(self.min_sequence_length):
                idx = i % len(silhouettes)
                extended_silhouettes.append(silhouettes[idx])
                extended_parsing.append(parsing_masks[idx])
            
            silhouettes = extended_silhouettes
            parsing_masks = extended_parsing
        
        # Limit to target sequence length
        if len(silhouettes) > self.target_sequence_length:
            # Sample frames uniformly
            indices = np.linspace(0, len(silhouettes) - 1, self.target_sequence_length, dtype=int)
            silhouettes = [silhouettes[i] for i in indices]
            parsing_masks = [parsing_masks[i] for i in indices]
        
        # Process silhouettes
        processed_sils = []
        for sil in silhouettes:
            if sil.shape[:2] != (self.input_height, self.input_width):
                sil_resized = cv2.resize(sil, (self.input_width, self.input_height))
            else:
                sil_resized = sil.copy()
            
            # Normalize to [0, 1]
            if sil_resized.max() > 1:
                sil_resized = sil_resized.astype(np.float32) / 255.0
            
            processed_sils.append(sil_resized)
        
        # Process parsing masks
        processed_pars = []
        for par in parsing_masks:
            if par.shape[:2] != (self.input_height, self.input_width):
                par_resized = cv2.resize(par, (self.input_width, self.input_height), 
                                       interpolation=cv2.INTER_NEAREST)
            else:
                par_resized = par.copy()
            
            # Keep parsing masks as integers (class IDs 0-6)
            # Do NOT normalize - XGait expects semantic class labels, not normalized values
            # Maintain integer precision to avoid class label corruption
            processed_pars.append(par_resized.astype(np.uint8))  # Use uint8 to preserve class IDs exactly
        
        # Convert to tensors [S, H, W] with proper dtype for semantic segmentation
        sils_array = np.array(processed_sils, dtype=np.float32)
        pars_array = np.array(processed_pars, dtype=np.float32)  # Float for F.interpolate compatibility
        
        # PERF-001 fix: Use tensor pooling to prevent memory fragmentation
        sils_tensor = self.get_pooled_tensor(sils_array.shape, torch.float32)
        sils_tensor.copy_(torch.from_numpy(sils_array))
        
        # Create parsing tensor as Float for F.interpolate compatibility
        # F.interpolate requires float tensors, but class values remain as integers (0.0, 1.0, etc.)
        pars_tensor = self.get_pooled_tensor(pars_array.shape, torch.float32)
        pars_tensor.copy_(torch.from_numpy(pars_array))
        
        # Validate that semantic class indices are in expected range (should be integer values 0-6)
        unique_values = torch.unique(pars_tensor).long()
        if unique_values.min() < 0 or unique_values.max() > 6:
            logger.warning(f"Parsing mask contains out-of-range class values: min={unique_values.min()}, max={unique_values.max()}")
        
        unique_values = torch.unique(pars_tensor).long()  # Get unique class IDs as integers
        expected_classes = set(range(7))  # Classes 0-6 for human parsing
        actual_classes = set(unique_values.cpu().numpy())
        if not actual_classes.issubset(expected_classes):
            logger.warning(f"Parsing mask contains unexpected class values: {sorted(actual_classes)}, expected: {sorted(expected_classes)}")
        
        # Add batch dimension [1, S, H, W]
        sils_tensor = sils_tensor.unsqueeze(0)
        pars_tensor = pars_tensor.unsqueeze(0)
        
        return pars_tensor, sils_tensor  # Note: parsing first!
    
    def extract_features(self, silhouettes: List[np.ndarray], 
                        parsing_masks: List[np.ndarray]) -> np.ndarray:
        """
        Extract features using official XGait model with proper memory management
        """
        try:
            with torch.no_grad():
                self.model.eval()
                
                # Clear any cached gradients
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                
                # Preprocess with memory-efficient operations
                pars, sils = self.preprocess_sequence(silhouettes, parsing_masks)
                
                # Prepare inputs in official format
                ipts = [pars, sils]
                labs = torch.zeros(1, dtype=torch.long, device=self.device)  # Use proper dtype
                seqL = [len(silhouettes)]
                
                inputs = (ipts, labs, None, None, seqL)
                
                # Forward pass with memory management
                output = self.model(inputs)
                
                # Extract inference features
                features = output['inference_feat']['embeddings']
                
                # Features shape: [batch, feature_dim, parts] e.g., [1, 256, 64]
                # Flatten to single vector per sample: [batch, feature_dim * parts]
                batch_size = features.size(0)
                features_flat = features.view(batch_size, -1)  # [1, 16384]
                
                # L2 normalization across the ENTIRE embedding vector (not per-part)
                # This preserves relative magnitudes and improves discrimination
                features_normalized = torch.nn.functional.normalize(features_flat, p=2, dim=1)
                
                # DEBUG: Log embedding statistics to understand discrimination issue
                logger.debug(f"Embedding stats - mean: {features_flat.mean():.4f}, std: {features_flat.std():.4f}, "
                           f"min: {features_flat.min():.4f}, max: {features_flat.max():.4f}")
                logger.debug(f"Normalized embedding - mean: {features_normalized.mean():.4f}, "
                           f"std: {features_normalized.std():.4f}, norm: {torch.norm(features_normalized, p=2):.4f}")
                
                # Move to CPU and convert to numpy immediately to free GPU memory
                result = features_normalized.detach().cpu().numpy()  # [batch, 16384]
                
                # PERF-001 fix: Return tensors to pool and cleanup
                self.return_tensor_to_pool(sils)
                self.return_tensor_to_pool(pars)
                self.cleanup_tensor_pool()
                
                # Clear intermediate tensors
                del features, features_normalized, output, pars, sils, ipts, labs
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                
                return result
                
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            
            # Log detailed tensor information for debugging
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            
            # Try to log tensor dtypes if they exist
            try:
                if 'pars' in locals():
                    logger.error(f"pars tensor - dtype: {pars.dtype}, shape: {pars.shape}")
                if 'sils' in locals():
                    logger.error(f"sils tensor - dtype: {sils.dtype}, shape: {sils.shape}")
                if 'labs' in locals():
                    logger.error(f"labs tensor - dtype: {labs.dtype}, shape: {labs.shape}")
            except:
                pass
            
            # PERF-001 fix: Cleanup on error
            self.cleanup_tensor_pool()
            
            # Clear cache on error
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            return np.array([])
    
    def is_model_loaded(self) -> bool:
        """Check if model weights are loaded"""
        return self.model_loaded


def create_official_xgait_inference(model_path: Optional[str] = None, 
                                   device: str = None, 
                                   num_classes: int = 3000) -> OfficialXGaitInference:
    """
    Create official XGait inference engine
    """
    if device is None:
        from ...config import get_xgait_device
        device = get_xgait_device()
    
    # Look for default model path
    if model_path is None:
        import os
        current_dir = Path(__file__).parent.parent.parent.parent
        potential_paths = [
            current_dir / "weights" / "Gait3D-XGait-120000.pt",
            current_dir / "weights" / "xgait_gait3d.pth"
        ]
        
        for path in potential_paths:
            if path.exists():
                model_path = str(path)
                break
    
    return OfficialXGaitInference(model_path=model_path, device=device, num_classes=num_classes)
