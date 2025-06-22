# Simple XGait Inference Pipeline

A working, modular inference pipeline for person identification using a simplified XGait approach. This implementation provides a functional foundation that can be extended with actual model implementations.

## ✅ Current Status

**WORKING** - The simple inference pipeline is fully functional with placeholder implementations that demonstrate the complete workflow.

## 🏗️ Architecture

### Core Components

1. **SimpleInferencePipeline** - Main orchestration class
2. **Parallel Processing** - Silhouette extraction and parsing run in parallel
3. **Gallery System** - Store and match person features
4. **Device Optimization** - GPU/CPU optimization when available

### Features

- ✅ Modular design for easy extension
- ✅ Parallel processing for speed
- ✅ Thread-safe gallery management
- ✅ Device optimization (CPU/GPU)
- ✅ Configurable identification threshold
- ✅ Complete inference workflow
- ✅ Easy-to-use API

## 🚀 Quick Start

### Basic Usage

```python
from simple_inference_pipeline import create_simple_inference_pipeline
import numpy as np

# Create pipeline
pipeline = create_simple_inference_pipeline(
    device="cpu",  # or "cuda", "mps"
    identification_threshold=0.7,
    parallel_processing=True
)

# Prepare person crops (example)
person_crops = [
    np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8),  # Person crop 1
    np.random.randint(0, 255, (120, 60, 3), dtype=np.uint8),  # Person crop 2
]

# Organize by tracks
tracks_data = {
    1: person_crops[:1],  # Track 1
    2: person_crops[1:],  # Track 2
}

# Process tracks
results = pipeline.process_tracks(tracks_data)

# Check results
for track_id, result in results.items():
    print(f"Track {track_id}: {result['identified_person']} (confidence: {result['confidence']:.3f})")
```

### Gallery Management

```python
# Add known persons to gallery
person_features = [np.random.rand(3) for _ in range(5)]
pipeline.add_to_gallery("John_Doe", person_features)

# Check gallery stats
stats = pipeline.get_gallery_stats()
print(f"Gallery contains {stats['num_persons']} persons")

# Clear gallery
pipeline.clear_gallery()
```

## 🧪 Testing

Run the test suite to verify everything works:

```bash
# Test the simple pipeline
python test_simple_pipeline.py

# Run the demo
python simple_inference_pipeline.py
```

## 📋 API Reference

### SimpleInferencePipeline

Main class for person identification inference.

#### Constructor

```python
SimpleInferencePipeline(
    device: str = "cpu",
    identification_threshold: float = 0.7,
    parallel_processing: bool = True,
    max_workers: int = 4
)
```

#### Methods

- `process_tracks(tracks_data: Dict[int, List[np.ndarray]]) -> Dict[int, Dict]`
  - Process person tracks and perform identification
  - Returns identification results for each track

- `add_to_gallery(person_id: str, features: List[np.ndarray])`
  - Add person features to the identification gallery

- `clear_gallery()`
  - Clear all persons from the gallery

- `get_gallery_stats() -> Dict`
  - Get statistics about the gallery

### Factory Function

```python
create_simple_inference_pipeline(
    device: str = "cpu",
    identification_threshold: float = 0.7,
    parallel_processing: bool = True,
    max_workers: int = 4
) -> SimpleInferencePipeline
```

## 🔧 Configuration

### Device Options

- `"cpu"` - Use CPU for inference
- `"cuda"` - Use NVIDIA GPU (if available)
- `"mps"` - Use Apple Silicon GPU (if available)

### Performance Settings

- `identification_threshold` - Minimum confidence for positive identification (0.0-1.0)
- `parallel_processing` - Enable parallel processing for speed
- `max_workers` - Number of worker threads for parallel processing

## 🎯 Integration Guide

### Extending with Real Models

To replace placeholder implementations with real models:

1. **Silhouette Extraction**: Replace `extract_silhouettes()` method
2. **Human Parsing**: Replace `extract_parsing()` method  
3. **Feature Extraction**: Replace `extract_features()` method

Example:

```python
class RealInferencePipeline(SimpleInferencePipeline):
    def _initialize_models(self):
        # Load actual U²-Net model
        self.silhouette_extractor = load_unet_model()
        
        # Load actual SCHP model
        self.parsing_model = load_schp_model()
        
        # Load actual XGait model
        self.xgait_model = load_xgait_model()
    
    def extract_silhouettes(self, crops):
        # Implement actual silhouette extraction
        return self.silhouette_extractor.process(crops)
```

### Video Processing Integration

```python
import cv2
from simple_inference_pipeline import create_simple_inference_pipeline

# Initialize pipeline
pipeline = create_simple_inference_pipeline()

# Process video
cap = cv2.VideoCapture("video.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 1. Detect persons (use YOLO or similar)
    person_detections = detect_persons(frame)
    
    # 2. Track persons (use simple tracker or DeepSORT)
    tracks = track_persons(person_detections)
    
    # 3. Extract person crops
    tracks_data = {}
    for track_id, bbox in tracks.items():
        crop = extract_crop(frame, bbox)
        tracks_data[track_id] = [crop]
    
    # 4. Run identification
    results = pipeline.process_tracks(tracks_data)
    
    # 5. Visualize results
    display_results(frame, tracks, results)
```

## 🎨 Visualization

The pipeline provides identification results that can be easily visualized:

```python
def visualize_results(frame, tracks, results):
    for track_id, result in results.items():
        person_id = result.get('identified_person', 'Unknown')
        confidence = result.get('confidence', 0.0)
        
        # Draw on frame
        label = f"{person_id} ({confidence:.2f})"
        # ... drawing code ...
```

## 🔄 Workflow

1. **Input**: Person crop images organized by track ID
2. **Parallel Processing**: 
   - Extract silhouettes
   - Extract human parsing masks
3. **Feature Extraction**: Combine silhouettes and parsing into feature vectors
4. **Identification**: Match features against gallery using cosine similarity
5. **Output**: Identification results with confidence scores

## 📊 Performance

The simple pipeline demonstrates:
- **Fast Processing**: Parallel extraction of silhouettes and parsing
- **Efficient Threading**: Configurable worker threads
- **Low Latency**: Minimal overhead for coordination
- **Scalable**: Easy to extend with more sophisticated models

### Timing Example
```
Processing 2 tracks with 3 total crops: ~0.001 seconds
```

## 🚨 Known Limitations

1. **Placeholder Models**: Current implementation uses dummy models
2. **Simple Features**: Basic feature extraction for demonstration
3. **No Persistence**: Gallery is not saved between sessions
4. **Basic Similarity**: Uses cosine similarity for matching

## 🔮 Future Enhancements

1. **Real Model Integration**: Replace placeholders with actual trained models
2. **Advanced Features**: Implement sophisticated feature extraction
3. **Persistent Gallery**: Save/load gallery from disk
4. **Advanced Matching**: Use learned similarity metrics
5. **Batch Optimization**: Optimize for larger batch sizes
6. **Model Caching**: Cache model outputs for efficiency

## 📁 File Structure

```
/
├── simple_inference_pipeline.py    # Main pipeline implementation
├── test_simple_pipeline.py         # Test suite
├── src/                            # Original source modules
│   ├── utils/device_utils.py       # Device optimization utilities
│   └── config.py                   # Configuration utilities
└── README_SIMPLE.md                # This documentation
```

## 🤝 Contributing

To extend this pipeline:

1. Fork the repository
2. Create a feature branch
3. Implement your enhancement
4. Add tests
5. Submit a pull request

## 📄 License

This implementation is provided as-is for research and development purposes.

---

**Status**: ✅ Working and ready to use!  
**Last Updated**: June 2025  
**Version**: 1.0.0
