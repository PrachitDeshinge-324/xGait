# Clustering-Based Embedding Visualization

## Overview
The person identification system now uses advanced clustering methods instead of PCA for embedding visualization. This provides better insights into feature separability and person identification quality.

## Available Clustering Methods

### 1. K-means Clustering (Default)
- **Purpose**: Groups features into a predetermined number of clusters
- **Best for**: When you know the expected number of persons
- **Parameters**: `n_clusters` (default: 3)
- **Visualization**: Shows cluster centers and boundaries

### 2. DBSCAN
- **Purpose**: Density-based clustering that automatically finds clusters
- **Best for**: Discovering natural groupings without pre-specifying cluster count
- **Parameters**: `eps=0.5`, `min_samples=2`
- **Advantages**: Can identify outliers and noise points

### 3. t-SNE (t-Distributed Stochastic Neighbor Embedding)
- **Purpose**: Non-linear dimensionality reduction for 2D visualization
- **Best for**: Preserving local structure and revealing cluster patterns
- **Parameters**: `perplexity` (auto-adjusted based on data size)
- **Advantages**: Excellent for visualization, reveals complex patterns

### 4. UMAP (Uniform Manifold Approximation and Projection)
- **Purpose**: Advanced dimensionality reduction with global structure preservation
- **Best for**: Large datasets, preserving both local and global structure
- **Parameters**: `n_neighbors`, `min_dist` (auto-tuned)
- **Advantages**: Faster than t-SNE, better global structure preservation

## Usage Examples

### Basic Clustering Analysis
```python
# Create gallery manager with K-means clustering
gallery = GalleryManager(
    clustering_method="kmeans",
    n_clusters=3
)

# Run clustering analysis
clustering_path = app.run_clustering_analysis(
    save_path="clustering_analysis.png",
    show_plot=False
)
```

### Using Different Clustering Methods
```python
# DBSCAN for automatic cluster discovery
gallery = GalleryManager(clustering_method="dbscan")

# t-SNE for visualization
gallery = GalleryManager(clustering_method="tsne")

# UMAP for large datasets (if available)
gallery = GalleryManager(clustering_method="umap")
```

## Visualization Features

### Enhanced Plots Include:
1. **Person Clusters**: Each person's features shown in different colors
2. **Convex Hulls**: Boundaries around each person's feature space
3. **Cluster Centers**: For K-means and DBSCAN methods
4. **Query Points**: Current tracking targets overlaid as stars
5. **Statistics Panel**: Shows clustering quality metrics

### Quality Metrics:
- **Silhouette Score**: Measures clustering quality (higher = better)
- **Separability Score**: Measures feature distinction between persons
- **Intra/Inter-person Similarity**: Quality assessment metrics

## Migration from PCA

### Backward Compatibility
The system maintains backward compatibility with existing code:
```python
# Old PCA method still works (alias to clustering method)
pca_path = app.run_pca_analysis(save_path="analysis.png")

# New clustering method (recommended)
clustering_path = app.run_clustering_analysis(save_path="analysis.png")
```

### Advantages over PCA
1. **Multiple Methods**: Choose the best method for your data
2. **Better Visualization**: Non-linear methods reveal complex patterns
3. **Automatic Tuning**: Parameters auto-adjust based on data size
4. **Quality Metrics**: Built-in clustering quality assessment
5. **Outlier Detection**: DBSCAN can identify problematic features

## Configuration Options

### Gallery Manager Parameters
```python
GalleryManager(
    gallery_dir="gallery_data",
    similarity_threshold=0.7,           # Feature matching threshold
    auto_add_threshold=0.5,             # Auto-add new persons threshold
    max_features_per_person=20,         # Feature buffer size
    n_clusters=3,                       # Number of clusters (for K-means)
    clustering_method="kmeans"          # Clustering method to use
)
```

### Supported Methods
- `"kmeans"` - K-means clustering (default)
- `"dbscan"` - DBSCAN density-based clustering
- `"tsne"` - t-SNE dimensionality reduction
- `"umap"` - UMAP dimensionality reduction (if available)

## Performance Recommendations

### For Small Datasets (< 100 features):
- Use **t-SNE** for best visualization quality
- Set `clustering_method="tsne"`

### For Medium Datasets (100-1000 features):
- Use **K-means** for speed and interpretability
- Set `clustering_method="kmeans"` with appropriate `n_clusters`

### For Large Datasets (> 1000 features):
- Use **UMAP** if available, otherwise **K-means**
- Set `clustering_method="umap"` or `clustering_method="kmeans"`

### For Unknown Number of Persons:
- Use **DBSCAN** for automatic cluster discovery
- Set `clustering_method="dbscan"`

## Output Files

The system now generates clustering-based visualizations:
- `final_clustering_analysis.png` - Main analysis output
- `clustering_visualization_YYYYMMDD_HHMMSS.png` - Timestamped outputs
- Gallery reports include clustering quality metrics

## Benefits

1. **Better Feature Understanding**: Clustering reveals natural groupings
2. **Quality Assessment**: Built-in metrics for feature quality
3. **Flexible Methods**: Choose the best approach for your data
4. **Automatic Tuning**: Parameters adjust based on data characteristics
5. **Enhanced Visualization**: Multiple visualization options available
