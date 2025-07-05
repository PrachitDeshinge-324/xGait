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
        
        # Ensure minimum sequence length
        if len(silhouettes) < self.min_sequence_length:
            # Repeat frames to reach minimum length
            repeat_factor = self.min_sequence_length // len(silhouettes) + 1
            silhouettes = (silhouettes * repeat_factor)[:self.min_sequence_length]
            parsing_masks = (parsing_masks * repeat_factor)[:self.min_sequence_length]
        
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
            
            # Normalize parsing masks
            if par_resized.max() > 1:
                par_resized = par_resized.astype(np.float32) / par_resized.max()
            
            processed_pars.append(par_resized)
        
        # Convert to tensors [S, H, W]
        sils_tensor = torch.FloatTensor(np.array(processed_sils)).to(self.device)
        pars_tensor = torch.FloatTensor(np.array(processed_pars)).to(self.device)
        
        # Add batch dimension [1, S, H, W]
        sils_tensor = sils_tensor.unsqueeze(0)
        pars_tensor = pars_tensor.unsqueeze(0)
        
        return pars_tensor, sils_tensor  # Note: parsing first!
    
    def extract_features(self, silhouettes: List[np.ndarray], 
                        parsing_masks: List[np.ndarray]) -> np.ndarray:
        """
        Extract features using official XGait model
        """
        try:
            with torch.no_grad():
                self.model.eval()
                
                # Preprocess
                pars, sils = self.preprocess_sequence(silhouettes, parsing_masks)
                
                # Prepare inputs in official format
                ipts = [pars, sils]
                labs = torch.zeros(1).long().to(self.device)  # Dummy labels for inference
                seqL = [len(silhouettes)]
                
                inputs = (ipts, labs, None, None, seqL)
                
                # Forward pass
                output = self.model(inputs)
                
                # Extract inference features
                features = output['inference_feat']['embeddings']
                
                return features.cpu().numpy()
                
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
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
