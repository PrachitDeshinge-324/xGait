"""
Device-aware utilities for cross-platform compatibility
"""
import torch
import numpy as np
from typing import Union, Tuple, Any
import cv2

class DeviceManager:
    """Manages device-specific operations and optimizations"""
    
    def __init__(self, device: str, dtype: torch.dtype = None):
        self.device = device
        self.dtype = dtype or self._get_default_dtype()
        self.use_autocast = self._should_use_autocast()
        
    def _get_default_dtype(self) -> torch.dtype:
        """Get default dtype for device"""
        if self.device == "cuda":
            return torch.float16
        elif self.device == "mps":
            return torch.float32
        else:
            return torch.float32
    
    def _should_use_autocast(self) -> bool:
        """Determine if autocast should be used"""
        return self.device == "cuda"
    
    def prepare_tensor(self, tensor: Union[torch.Tensor, np.ndarray], 
                      requires_grad: bool = False) -> torch.Tensor:
        """Prepare tensor for the target device with appropriate dtype"""
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        
        tensor = tensor.to(device=self.device, dtype=self.dtype)
        
        if requires_grad:
            tensor.requires_grad_(True)
            
        return tensor
    
    def prepare_image(self, image: np.ndarray) -> torch.Tensor:
        """Prepare image for model input with device-specific optimizations"""
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and prepare for device
        tensor = torch.from_numpy(image)
        
        # Rearrange dimensions if needed (H, W, C) -> (C, H, W)
        if len(tensor.shape) == 3:
            tensor = tensor.permute(2, 0, 1)
        
        # Add batch dimension if needed
        if len(tensor.shape) == 3:
            tensor = tensor.unsqueeze(0)
        
        return self.prepare_tensor(tensor)
    
    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Prepare model for the target device with optimizations"""
        model = model.to(device=self.device, dtype=self.dtype)
        
        # Device-specific optimizations
        if self.device == "cuda":
            # Enable channels_last memory format for better performance
            model = model.to(memory_format=torch.channels_last)
            
            # Compile model for better performance (PyTorch 2.0+)
            try:
                if hasattr(torch, 'compile'):
                    model = torch.compile(model, mode="reduce-overhead")
            except Exception as e:
                print(f"Warning: Could not compile model: {e}")
        
        elif self.device == "mps":
            # MPS-specific optimizations
            model = model.to(memory_format=torch.contiguous_format)
        
        return model
    
    def autocast_context(self):
        """Get appropriate autocast context for device"""
        if self.use_autocast and self.device == "cuda":
            return torch.cuda.amp.autocast(dtype=torch.float16)
        else:
            # Return a no-op context manager
            from contextlib import nullcontext
            return nullcontext()
    
    def optimize_inference(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply inference-specific optimizations"""
        model.eval()
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad_(False)
        
        # Device-specific inference optimizations
        if self.device == "cuda":
            # Use FP16 for CUDA inference if available
            try:
                if hasattr(model, 'half'):
                    model = model.half()
            except Exception:
                pass
        
        return model
    
    def get_memory_info(self) -> dict:
        """Get device memory information"""
        if self.device == "cuda":
            return {
                "allocated": torch.cuda.memory_allocated(),
                "cached": torch.cuda.memory_reserved(),
                "max_allocated": torch.cuda.max_memory_allocated()
            }
        elif self.device == "mps":
            return {
                "allocated": torch.mps.current_allocated_memory(),
                "driver_allocated": torch.mps.driver_allocated_memory()
            }
        else:
            import psutil
            return {
                "system_memory": psutil.virtual_memory().percent
            }
    
    def clear_cache(self):
        """Clear device cache"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()
    
    def synchronize(self):
        """Synchronize device operations"""
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

def get_optimal_batch_size(device: str, base_batch_size: int = 8) -> int:
    """Get optimal batch size for device"""
    if device == "cuda":
        # Get GPU memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        if total_memory > 16 * 1024**3:  # > 16GB
            return base_batch_size * 2
        elif total_memory > 8 * 1024**3:  # > 8GB
            return base_batch_size
        else:
            return max(1, base_batch_size // 2)
    
    elif device == "mps":
        # MPS typically has shared memory with system
        return max(1, base_batch_size // 2)
    
    else:  # CPU
        return max(1, base_batch_size // 4)

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array, handling device transfers"""
    if tensor.is_cuda or (hasattr(tensor, 'is_mps') and tensor.is_mps):
        tensor = tensor.cpu()
    
    if tensor.requires_grad:
        tensor = tensor.detach()
    
    return tensor.numpy()

def ensure_tensor_device(tensor: torch.Tensor, target_device: str, 
                        target_dtype: torch.dtype = None) -> torch.Tensor:
    """Ensure tensor is on the correct device with correct dtype"""
    if target_dtype is None:
        if target_device == "cuda":
            target_dtype = torch.float16
        else:
            target_dtype = torch.float32
    
    return tensor.to(device=target_device, dtype=target_dtype)
