# XGait Person Identification System

A high-performance person identification system using XGait gait recognition, parallel silhouette extraction, and human parsing for real-time inference.

## 🚀 Features

- **XGait-based Person Identification**: Uses pre-trained XGait model for gait-based person identification
- **Parallel Processing**: Silhouette extraction and human parsing run in parallel for maximum speed
- **Device Optimization**: Automatic GPU acceleration with device-specific optimizations (CUDA, MPS, CPU)
- **Real-time Inference**: Optimized for real-time video processing with memory management
- **Modular Design**: Clean separation of components for easy maintenance and extension

## 📋 System Components

### Core Models
1. **Silhouette Extraction**: U²-Net architecture for fast and accurate person silhouette extraction
2. **Human Parsing**: SCHP ResNet101 model for detailed human body part segmentation
3. **XGait Model**: Pre-trained gait recognition model for person identification
4. **Person Detection**: YOLO11 for robust person detection and tracking

### Processing Pipeline
```
Input Video → Person Detection → Tracking → Crop Extraction
                                               ↓
Person Gallery ← XGait Features ← Silhouettes & Parsing Maps
                                               ↑
                                    [Parallel Processing]
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- CUDA (optional, for GPU acceleration)

### Required Dependencies
```bash
pip install torch torchvision ultralytics opencv-python pillow numpy
```

### Model Weights
Place the following pre-trained models in the `weights/` directory:
- `Gait3D-XGait-120000.pt` - Pre-trained XGait model
- `schp_resnet101.pth` - Human parsing model
- `yolo11m.pt` - YOLO detection model

## 🚀 Quick Start

### Basic Usage
```bash
python xgait_identify.py --video input/video.mp4 --device mps
```

### Advanced Usage
```bash
python xgait_identify.py \
    --video input/video.mp4 \
    --device cuda \
    --xgait-model weights/Gait3D-XGait-120000.pt \
    --parsing-model weights/schp_resnet101.pth \
    --id-threshold 0.6 \
    --confidence 0.5 \
    --verbose
```

### Command Line Options
- `--video`: Input video path (default: `input/3c.mp4`)
- `--device`: Processing device (`cpu`, `cuda`, `mps`)
- `--xgait-model`: Path to XGait model weights
- `--parsing-model`: Path to human parsing model weights
- `--id-threshold`: Identification confidence threshold (0.0-1.0)
- `--confidence`: Detection confidence threshold (0.0-1.0)
- `--no-display`: Disable video display window
- `--verbose`: Enable detailed logging
- `--load-gallery`: Load existing person gallery
- `--debug`: Enable debug mode

## 📊 Performance Optimization

### Device-Specific Optimizations
- **CUDA**: Float16 precision, autocast, memory optimization
- **MPS (Apple Silicon)**: Optimized for Apple M1/M2 processors
- **CPU**: Multi-threading with optimized batch processing

### Memory Management
- Automatic memory cleanup every 500 frames
- Configurable sequence length for identification
- Efficient tensor operations with device synchronization

### Parallel Processing
- Silhouette extraction and human parsing run in parallel
- ThreadPoolExecutor for concurrent model inference
- Batch processing for multiple tracks

## 🎯 Identification Workflow

### 1. Person Detection and Tracking
- YOLO11 detects persons in each frame
- IoU-based tracking maintains person tracks
- Configurable tracking parameters for stability

### 2. Feature Extraction (Parallel)
- **Silhouette Branch**: U²-Net extracts person silhouettes
- **Parsing Branch**: SCHP model segments body parts
- Both processes run concurrently for speed

### 3. XGait Identification
- Combined features fed to XGait model
- Gallery-based person identification
- Confidence-based identity assignment

### 4. Results
- Consistent person IDs across video
- Real-time visualization with track histories
- Comprehensive performance statistics

## 📈 Performance Metrics

The system tracks detailed performance metrics:
- **Processing FPS**: Real-time processing speed
- **Component Timing**: Individual model inference times
- **Memory Usage**: GPU/CPU memory utilization
- **Identification Accuracy**: Gallery statistics and confidence scores

## 🔧 Configuration

### Model Configuration
```python
@dataclass
class ModelConfig:
    yolo_model_path: str = "weights/yolo11m.pt"
    xgait_model_path: str = "weights/Gait3D-XGait-120000.pt"
    parsing_model_path: str = "weights/schp_resnet101.pth"
    device: str = "mps"  # or "cuda", "cpu"
```

### Tracker Configuration
```python
@dataclass
class TrackerConfig:
    confidence_threshold: float = 0.5
    identification_threshold: float = 0.6
    sequence_length: int = 10
    max_missing_frames: int = 75
```

## 📁 Project Structure

```
├── src/
│   ├── models/
│   │   ├── silhouette_model.py    # U²-Net silhouette extraction
│   │   ├── parsing_model.py       # SCHP human parsing
│   │   ├── xgait_model.py         # XGait inference wrapper
│   │   └── identification_processor.py  # Main coordination
│   ├── trackers/
│   │   └── xgait_tracker.py       # Enhanced tracking with XGait
│   ├── utils/
│   │   ├── device_utils.py        # Device optimization utilities
│   │   └── visualization.py       # Visualization tools
│   └── config.py                  # System configuration
├── weights/                       # Model weights directory
├── input/                         # Input videos
├── xgait_identify.py             # Main application
└── README_XGAIT.md              # This file
```

## 🎨 Visualization Features

- **Real-time Display**: Live video with tracking overlays
- **Color-coded IDs**: Consistent colors for each person identity
- **Track Histories**: Visual trails showing person movements
- **Performance HUD**: FPS, memory usage, and statistics
- **Interactive Controls**: Pause, resume, save gallery

## 📊 Output and Results

### Console Output
- Real-time progress updates
- Performance statistics
- Device utilization metrics
- Identification accuracy reports

### Final Statistics
- Total tracks and identities detected
- Processing performance breakdown
- Memory usage summary
- Identification confidence analysis

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in configuration
   - Enable memory cleanup more frequently
   - Use CPU fallback if needed

2. **Slow Processing**
   - Verify GPU acceleration is enabled
   - Check model weights are properly loaded
   - Consider reducing video resolution

3. **Poor Identification**
   - Adjust identification threshold
   - Increase sequence length for more stable features
   - Ensure quality person crops

### Performance Tips
- Use GPU acceleration when available
- Enable parallel processing for best speed
- Adjust confidence thresholds based on video quality
- Monitor memory usage for long videos

## 🔮 Future Enhancements

- [ ] Support for custom XGait model training
- [ ] Integration with additional gait recognition models
- [ ] Web interface for remote processing
- [ ] Batch video processing capabilities
- [ ] Export results to various formats

## 📄 License

This project uses pre-trained models that may have their own licenses:
- XGait: Research use only
- SCHP: Academic use
- YOLO: AGPL-3.0

Please ensure compliance with all model licenses for your use case.
