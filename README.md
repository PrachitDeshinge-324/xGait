# Person Identification with Simplified XGait System

A simplified person identification system using XGait embeddings for gait-based recognition.

## Overview

This system has been refactored to implement a clean, minimal approach to person identification:

- **30-frame sequence feature extraction**: Collects silhouettes over 30 frames to compute a single XGait embedding
- **Simple gallery system**: Maps person names (e.g., `'person_1'`, `'person_2'`) to their embeddings
- **Automatic person assignment**: New tracks are automatically assigned unique person IDs
- **No duplicate assignments**: Ensures no two tracks in a single frame get the same person name
- **Persistent storage**: Gallery is saved to disk for use across sessions

## Key Changes from Previous System

- Removed complex `GalleryManager` with clustering and visualization features
- Removed `identification_processor.py` with multiple model integrations
- Removed `manage_gallery.py` CLI tool
- Simplified to core identification logic only

## Architecture

### Core Components

1. **SimpleGaitIdentification** (`src/models/simple_gait_identification.py`)
   - Manages the gallery as a simple dictionary
   - Handles 30-frame sequence processing
   - Provides person identification logic
   - Saves/loads gallery data

2. **PersonTrackingApp** (`track_persons.py`)
   - Integrated with simplified identification system
   - Processes silhouettes and extracts XGait features
   - Runs frame-level identification avoiding duplicates

3. **Gallery Storage** (`gallery_data/simple_gallery.json`)
   - Simple JSON format storing person_id -> embedding mappings
   - Persistent across sessions
   - Human-readable format

## Usage

### Basic Tracking with Identification

```bash
python main.py --input video.mp4 --enable-identification --enable-gait
```

### Key Features

- **Automatic Person Detection**: New persons automatically get IDs like `person_1`, `person_2`, etc.
- **Similarity Matching**: Uses cosine similarity with 0.7 threshold for identification
- **Sequence Processing**: Extracts features every 30 frames for robust identification
- **No Duplicates**: Frame-level processing ensures unique person assignments

### Gallery Management

The gallery is automatically managed:

- New persons are added when similarity < 0.7
- Gallery is saved after each update
- Persists across application runs
- Can be manually cleared by deleting `gallery_data/simple_gallery.json`

## Configuration

Key parameters in `SimpleGaitIdentification`:

```python
similarity_threshold: float = 0.7      # Minimum similarity for identification
sequence_length: int = 30              # Frames needed for feature extraction
gallery_file: str = "gallery_data/simple_gallery.json"
```

## File Structure

```
.
├── main.py                           # Application entry point
├── track_persons.py                  # Main tracking application
├── src/
│   ├── models/
│   │   ├── simple_gait_identification.py  # Simplified gait identification
│   │   ├── xgait_model.py                 # XGait model interface
│   │   └── ...                            # Other models
│   └── ...
├── gallery_data/
│   └── simple_gallery.json          # Persistent gallery storage
└── weights/                          # Model weights
```

## Dependencies

- numpy
- scikit-learn (for cosine similarity)
- opencv-python
- pytorch (for XGait model)
- matplotlib (for visualizations)
- seaborn (for enhanced plotting)
- sklearn (for clustering and dimensionality reduction)

## Advanced Embedding Analysis & Clustering

The system now includes comprehensive clustering analysis and visualization capabilities for XGait embeddings:

### Features

- **Multiple Clustering Algorithms**: K-means, DBSCAN, Hierarchical clustering
- **Dimensionality Reduction**: PCA, t-SNE, UMAP (if available)
- **Quality Assessment**: Inter/intra-class distance analysis, separability scoring
- **Comprehensive Visualizations**: Similarity heatmaps, cluster plots, quality dashboards
- **Automated Analysis**: Runs automatically after processing

### Clustering Outputs

The system generates the following analysis files in `clustering_analysis/`:

1. **`dimensionality_reduction_*.png`** - PCA, t-SNE, UMAP visualizations
2. **`clustering_results_*.png`** - K-means elbow curves, silhouette analysis
3. **`similarity_heatmaps_*.png`** - Cosine similarity and distance matrices
4. **`quality_assessment_*.png`** - Embedding quality metrics and distributions
5. **`analysis_dashboard_*.png`** - Comprehensive overview dashboard
6. **`clustering_analysis_report_*.json`** - Detailed numerical results

### Example Clustering Output

```
📊 Analysis Summary:
   • Gallery Persons: 5
   • Track Features: 18
   • Embedding Dimension: 16384
   • Overall Quality: Moderate (0.60)

🎯 Clustering Results:
   • K-means optimal clusters: 4
   • K-means silhouette score: 0.192
   • DBSCAN clusters: 3
   • Hierarchical clusters: 5

🔍 Dimensionality Reduction:
   • PCA: 73.7% variance in 10 components
   • t-SNE: KL divergence = 0.437

🎨 Visualizations saved to: clustering_analysis/
```

### Manual Clustering Analysis

You can also run clustering analysis independently:

```python
from src.utils.embedding_clustering import EmbeddingClusterAnalyzer

# Create analyzer
analyzer = EmbeddingClusterAnalyzer(output_dir="my_analysis")

# Run analysis
results = analyzer.analyze_gallery_embeddings(
    gallery=gallery_dict,           # {person_id: embedding}
    track_features=track_dict,      # {track_id: [embeddings]}
    save_results=True
)
```

### Testing Clustering System

Test the clustering system with sample data:

```bash
python test_clustering.py
```

This generates synthetic embeddings and demonstrates all clustering capabilities.

## Removed Components

The following complex components were removed to simplify the system:

- `src/gallery/gallery_manager.py` - Complex gallery with clustering
- `manage_gallery.py` - CLI gallery management tool  
- `src/models/identification_processor.py` - Multi-model processor
- Gallery clustering and visualization features
- Advanced gallery analytics and reporting
- Multi-feature aggregation per person

## Benefits of Simplified Approach

1. **Easier to Understand**: Clear, linear flow from silhouettes to identification
2. **Easier to Maintain**: Fewer dependencies and components
3. **Faster Setup**: No complex gallery initialization
4. **Reliable**: Simpler logic reduces potential failure points
5. **Sufficient**: Meets core requirement of person identification

## Example Output

```
🔍 Track 1 identified as 'person_1' (confidence: 0.856)
🔍 Track 2 identified as 'person_2' (confidence: 1.000)
🔍 Track 3 identified as 'person_1' (confidence: 0.723)

📊 Gallery Summary:
   • Total Persons: 2
   • Person IDs: ['person_1', 'person_2']
   • Next Person ID: 3
```

This simplified system provides robust person identification while maintaining clean, maintainable code.
