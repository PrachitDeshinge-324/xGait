# xGait - Person Tracking and Gait Recognition System

A comprehensive person tracking and identification system using YOLO detection, TransReID for appearance-based re-identification, and XGait for gait-based feature extraction.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)

## 🌟 Features

- **Real-time Person Detection & Tracking**: YOLO11-based segmentation for accurate person detection with instance segmentation
- **Appearance-based Re-identification**: TransReID model for robust person re-identification across frames
- **Gait Recognition Pipeline**: 
  - Silhouette extraction using YOLO segmentation masks
  - Human parsing for detailed body part segmentation
  - XGait feature extraction for gait-based person identification
- **Multi-device Support**: Automatic optimization for CUDA, MPS (Apple Silicon), and CPU
- **Advanced Identity Management**: FAISS-based gallery system for efficient person matching
- **Interactive Review System**: Manual verification and correction of track assignments
- **Comprehensive Visualization**: Debug outputs, track visualization, and embedding analysis
- **Performance Monitoring**: Real-time statistics and device utilization tracking

## 📋 Table of Contents

- [System Requirements](#-system-requirements)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Advanced Features](#-advanced-features)
- [Troubleshooting](#-troubleshooting)
- [Performance Optimization](#-performance-optimization)

## 💻 System Requirements

### Hardware
- **Minimum**: 8GB RAM, modern CPU
- **Recommended**: 16GB+ RAM, NVIDIA GPU with 6GB+ VRAM
- **Supported**: NVIDIA CUDA GPUs, Apple Silicon (M1/M2), CPU-only mode

### Software
- **OS**: Linux, macOS, Windows
- **Python**: 3.8 or higher
- **CUDA**: 11.0+ (for NVIDIA GPUs)
- **Conda**: Miniconda or Anaconda (recommended)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/PrachitDeshinge-324/xGait.git
cd xgait_yolo
```

### 2. Create Conda Environment
```bash
# Create a new conda environment
conda create -n tracker python=3.10
conda activate tracker
```

### 3. Install Dependencies
We provide a setup script that handles dependency conflicts:

```bash
# Run the setup script (handles numpy/torch compatibility)
bash setup_environment.sh
```

**Manual installation (alternative):**
```bash
# Install numpy first via conda
conda install numpy=1.26.4 -c conda-forge -y

# Install scipy
pip install "scipy>=1.11.0,<1.14.0"

# Install remaining dependencies
pip install -r requirements.txt
```

### 4. Download Model Weights
Create a `Weights` directory at the project root level and download the following models:

```bash
mkdir -p ../Weights
cd ../Weights
```

Required models:
- **YOLO11**: `yolo11s-seg.pt` - [Download from Ultralytics](https://github.com/ultralytics/ultralytics)
- **TransReID**: `transreid.pth` - Person re-identification model
- **XGait**: `Gait3D-XGait-120000.pt` - Gait recognition model
- **Human Parsing**: `human_parsing.pth` - Body part segmentation model

**Note**: Model weights are not included in the repository. Please contact the authors or refer to the original model repositories.

## ⚡ Quick Start

### Basic Usage
```bash
# Process a video with default settings
python main.py --input input/video.mp4
```

### Common Use Cases
```bash
# Headless mode (no display window)
python main.py --input video.mp4 --no-display

# Save annotated output video
python main.py --input video.mp4 --save-video --output-video output/result.mp4

# Process first 500 frames only (testing)
python main.py --input video.mp4 --max-frames 500

# Enable debug mode with visualization
python main.py --input video.mp4 --debug --interactive
```

### Using the Run Script
For convenience, you can use the provided shell script:

```bash
# Edit run.sh to set your video path and parameters
bash run.sh
```

## 📖 Usage

### Command Line Arguments

```
Required Arguments:
  --input, -i          Input video file path

Optional Arguments:
  --output, -o         Output directory for results (default: ./output)
  --output-video       Path for annotated video output (e.g., output.mp4)
  --save-video         Save annotated video (default: True)
  --no-display         Run in headless mode without display window
  --max-frames         Maximum number of frames to process (for testing)
  --debug              Enable debug mode with detailed logging
  --interactive        Enable interactive track review after processing
```

### Example Workflows

#### 1. Basic Video Processing
```bash
python main.py --input input/sample.mp4
```
- Processes entire video
- Displays real-time tracking window
- Saves results to `output/` directory

#### 2. Production Mode (Headless)
```bash
python main.py \
    --input video.mp4 \
    --no-display \
    --save-video \
    --output-video results/tracked_video.mp4
```
- No display window (suitable for servers)
- Saves annotated video
- Generates analysis reports

#### 3. Development Mode (Debug + Interactive)
```bash
python main.py \
    --input video.mp4 \
    --debug \
    --interactive \
    --max-frames 500
```
- Processes first 500 frames
- Saves debug visualizations
- Launches interactive review after processing

## ⚙️ Configuration

The system uses a comprehensive configuration system defined in `src/config.py`. Key configuration classes:

### ModelConfig
- Model paths and device settings
- Data types and optimization flags
- Device-specific configurations (CUDA/MPS/CPU)

### TrackerConfig
- YOLO detection thresholds
- Track history and stability settings
- ReID similarity thresholds
- Gallery management parameters

### xgaitConfig
- Sequence buffer sizes
- Similarity thresholds for gait matching
- Processing intervals
- Debug output settings

### VideoConfig
- Input/output paths
- Frame processing limits
- Display settings

### Modifying Configuration
Edit `src/config.py` to adjust default parameters, or pass configurations programmatically:

```python
from src.config import SystemConfig, TrackerConfig

# Custom configuration
custom_config = SystemConfig()
custom_config.tracker.similarity_threshold = 0.4
custom_config.video.input_path = "my_video.mp4"
custom_config.debug_mode = True

# Initialize app with custom config
app = PersonTrackingApp(config=custom_config)
```

## 📁 Project Structure

```
xgait/
├── main.py                      # Main entry point
├── requirements.txt             # Python dependencies
├── setup_environment.sh         # Setup script
├── run.sh                       # Quick run script
├── clear_cache.sh              # Cache cleanup utility
│
├── input/                       # Input videos directory
├── output/                      # Output results directory
│
├── src/                         # Source code
│   ├── config.py               # Configuration system
│   ├── app/                    # Application layer
│   │   └── main_app.py        # Main application class
│   │
│   ├── models/                 # Model implementations
│   │   ├── reid_model.py      # TransReID model
│   │   ├── xgait_model.py     # XGait wrapper
│   │   ├── parsing_model.py   # Human parsing model
│   │   └── xgait/             # XGait implementation
│   │       ├── inference.py   # Inference engine
│   │       ├── model.py       # Model architecture
│   │       └── ...            # Supporting modules
│   │
│   ├── trackers/              # Tracking system
│   │   └── person_tracker.py # Main tracker class
│   │
│   ├── processing/            # Processing modules
│   │   ├── video_processor.py        # Video I/O
│   │   ├── gait_processor.py         # Gait processing
│   │   ├── statistics_manager.py     # Statistics
│   │   └── enhanced_identity_manager.py # Identity management
│   │
│   └── utils/                 # Utility modules
│       ├── visualization.py            # Visualization tools
│       ├── faiss_gallery.py           # FAISS gallery
│       ├── embedding_visualization.py  # Embedding analysis
│       └── visual_track_reviewer.py   # Interactive review
│
└── archive/                   # Archived/deprecated code
```

## 🎯 Advanced Features

### 1. Interactive Track Review
After processing, manually verify and correct track assignments:

```bash
python main.py --input video.mp4 --interactive
```

Features:
- Review each track's representative frames
- Assign person names to tracks
- Merge tracks belonging to same person
- View similarity statistics

### 2. Embedding Visualization
The system generates UMAP/t-SNE visualizations of gait embeddings:
- Located in `visualization_analysis/`
- Shows embedding clusters by track ID
- Helps analyze model performance

### 3. Debug Mode
Enable comprehensive debugging:

```bash
python main.py --input video.mp4 --debug
```

Generates:
- Frame-by-frame debug images in `debug_gait_parsing/`
- Silhouette visualizations
- Parsing results
- XGait feature activations

### 4. Gallery Management
The FAISS-based gallery system provides:
- Efficient similarity search
- Quality-based embedding storage
- Person-level statistics
- Automatic identity assignment

### 5. Performance Monitoring
Real-time statistics include:
- FPS and processing time
- GPU/memory utilization
- Track statistics
- Gait processing metrics

## 🔧 Troubleshooting

### Common Issues

#### 1. NumPy/PyTorch Compatibility Error
```bash
RuntimeError: Numpy is not available
```
**Solution**: Use the setup script which handles version compatibility
```bash
bash setup_environment.sh
```

#### 2. CUDA Out of Memory
```
CUDA out of memory
```
**Solutions**:
- Reduce batch size in `src/config.py`
- Process fewer frames at once
- Use CPU mode: Set `device = "cpu"` in config

#### 3. MPS (Apple Silicon) Issues
```bash
# If MPS errors occur, force CPU mode
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

#### 4. Model Weights Not Found
```
FileNotFoundError: [Errno 2] No such file or directory: '../Weights/...'
```
**Solution**: Ensure all model weights are in the `../Weights/` directory

#### 5. Slow Processing
- Enable GPU: Check CUDA installation
- Reduce `parsing_skip_interval` in config
- Lower video resolution
- Decrease `max_frames` for testing

### Memory Management

Clear GPU cache:
```bash
bash clear_cache.sh
```

Or programmatically:
```python
torch.cuda.empty_cache()  # CUDA
torch.mps.empty_cache()   # MPS
```

## 🚀 Performance Optimization

### Device-Specific Settings

#### CUDA (NVIDIA GPU)
- Uses float16 for memory efficiency
- Autocast enabled
- Model compilation supported
- CUDA streams for async operations

#### MPS (Apple Silicon)
- Uses float32 (MPS limitation)
- Optimized batch sizes for M1/M2
- Contiguous memory format

#### CPU
- Standard float32 precision
- Conservative batch sizes
- No compilation overhead

### Configuration Tuning

**For Speed:**
```python
config.identity.parsing_skip_interval = 5  # Process every 5th frame
config.tracker.max_missing_frames = 20     # Faster track cleanup
config.video.max_frames = 1000             # Limit frames
```

**For Accuracy:**
```python
config.identity.parsing_skip_interval = 1   # Process every frame
config.identity.similarity_threshold = 0.93 # Higher confidence
config.tracker.track_history_length = 60    # Longer history
```

### Batch Processing
The system automatically adjusts batch sizes based on device:
- CUDA: Up to 8 frames per batch
- MPS: Up to 6 frames per batch
- CPU: Up to 4 frames per batch

## 📊 Output Files

After processing, the following files are generated:

```
output/
├── tracked_video.mp4              # Annotated video (if --save-video)
├── visualization_analysis/        # Embedding visualizations
│   ├── track_metadata.json       # Track information
│   └── embedding_*.png           # UMAP/t-SNE plots
├── debug_gait_parsing/           # Debug outputs (if --debug)
│   ├── frame_XXXX_track_YY/     # Per-frame debug images
│   └── ...
└── logs/                         # Application logs
```

### Track Metadata JSON
```json
{
  "track_1": {
    "person_id": "Person_A",
    "first_seen": 15,
    "last_seen": 450,
    "total_frames": 435,
    "average_confidence": 0.95,
    "xgait_features_count": 87
  }
}
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Contact

For questions or issues:
- GitHub: [@PrachitDeshinge-324](https://github.com/PrachitDeshinge-324)
- Repository: [xGait](https://github.com/PrachitDeshinge-324/xGait)

## 🙏 Acknowledgments

This project builds upon:
- [YOLO](https://github.com/ultralytics/ultralytics) - Object detection
- [TransReID](https://github.com/damo-cv/TransReID) - Person re-identification
- [XGait](https://github.com/Gait3D/Gait3D-Benchmark) - Gait recognition
- [FAISS](https://github.com/facebookresearch/faiss) - Similarity search

## 📚 Citation

If you use this work in your research, please cite:
```bibtex
@software{xgait2024,
  author = {Prachit Deshinge},
  title = {xGait: Person Tracking and Gait Recognition System},
  year = {2024},
  url = {https://github.com/PrachitDeshinge-324/xGait}
}
```

---

**Version**: 1.0.0  
**Last Updated**: November 2025
