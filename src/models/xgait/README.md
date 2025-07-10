# XGait Model Package

This package contains the complete XGait implementation with modular architecture for better maintainability.

## Package Structure

```
xgait/
├── __init__.py           # Package entry point with all exports
├── adapter.py            # XGaitAdapter - main interface for the codebase
├── official.py           # Deprecated official model wrapper
├── backbone.py           # ResNet9 backbone implementation
├── alignment.py          # Cross-granularity alignment modules
├── pooling.py            # Temporal and spatial pooling modules
├── layers.py             # Custom layers (SetBlockWrapper, SeparateFCs, etc.)
├── model.py              # Core OfficialXGait model implementation
├── inference.py          # Inference engine for the official model
└── utils.py              # Utility functions
```

## Usage

### Main Entry Point (Recommended)
```python
# Use this import in your code
from src.models.xgait_model import create_xgait_inference

# This provides the XGaitAdapter with full backward compatibility
xgait = create_xgait_inference(model_path="weights/Gait3D-XGait-120000.pt")
```

### Direct Package Usage
```python
# For advanced usage, import directly from the package
from src.models.xgait import XGaitAdapter, OfficialXGaitInference
from src.models.xgait import create_xgait_adapter, create_official_xgait_inference

# Use the adapter (recommended for integration)
adapter = create_xgait_adapter(model_path="weights/Gait3D-XGait-120000.pt")

# Or use the core implementation directly
inference = create_official_xgait_inference(model_path="weights/Gait3D-XGait-120000.pt")
```

## Components

### Core Components
- **XGaitAdapter** (`adapter.py`): Main interface used by the tracking system
- **OfficialXGait** (`model.py`): Core model implementation matching Gait3D-Benchmark
- **OfficialXGaitInference** (`inference.py`): Inference engine for feature extraction

### Model Architecture
- **ResNet9** (`backbone.py`): Lightweight backbone network
- **CALayers/CALayersP** (`alignment.py`): Cross-granularity alignment modules
- **HorizontalPoolingPyramid** (`pooling.py`): Spatial pooling for part-based features
- **SeparateFCs/SeparateBNNecks** (`layers.py`): Part-specific classification layers

### Legacy Components
- **official.py**: Deprecated wrapper, kept for compatibility

## Integration Points

The XGait package integrates with:
- `main.py` - Main CLI entry point  
- `src.app.main_app` - Main tracking application
- `src.models.xgait_model` - Entry point module (outside the package)
- `src.utils.faiss_gallery` - FAISS-based person identification gallery
- `src.utils.embedding_visualization` - For feature visualization

## Migration Guide

If you're updating code that uses old XGait imports:

```python
# OLD (deprecated)
from src.models.official_xgait_model import create_official_xgait_inference

# NEW (recommended)
from src.models.xgait_model import create_xgait_inference

# OR (direct package usage)
from src.models.xgait import create_xgait_adapter
```

The `XGaitAdapter` provides full backward compatibility while using the official implementation underneath.
