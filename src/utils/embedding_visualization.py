"""
Embedding Visualization System for XGait Person Identification
Provides comprehensive visualization of gait feature embeddings and identity clusters
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Tuple, Dict, Optional
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

# Optional imports with fallbacks
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️  UMAP not available, using PCA/t-SNE only")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  Plotly not available, using matplotlib only")

logger = logging.getLogger(__name__)

class EmbeddingVisualizer:
    """
    Advanced visualization system for gait embeddings and identity analysis
    
    Features:
    - 2D/3D visualization with PCA, t-SNE, and UMAP
    - Track-based clustering analysis
    - Gallery identity separation analysis
    - Interactive visualizations with Plotly
    - Quality and consistency metrics
    """
    
    def __init__(self, 
                 random_state: int = 42,
                 figure_size: Tuple[int, int] = (12, 8),
                 style: str = "whitegrid"):
        """
        Initialize the embedding visualizer
        
        Args:
            random_state: Random seed for reproducible results
            figure_size: Default figure size for matplotlib plots
            style: Seaborn style for matplotlib plots
        """
        self.random_state = random_state
        self.figure_size = figure_size
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_style(style)
        sns.set_palette("husl")
        
        # Dimensionality reduction models
        self.pca_model = None
        self.tsne_model = None
        self.umap_model = None
        
        # Color schemes
        self.track_colors = None
        self.identity_colors = None
        
        logger.info("✅ Embedding Visualizer initialized")
    
    def _prepare_color_schemes(self, track_ids: List[int], identity_ids: List[str]):
        """Prepare consistent color schemes for tracks and identities"""
        # Generate colors for tracks
        n_tracks = len(set(track_ids))
        if n_tracks <= 10:
            track_palette = sns.color_palette("tab10", n_tracks)
        else:
            track_palette = sns.color_palette("husl", n_tracks)
        
        unique_tracks = sorted(list(set(track_ids)))
        self.track_colors = {track_id: track_palette[i % len(track_palette)] 
                           for i, track_id in enumerate(unique_tracks)}
        
        # Generate colors for identities
        n_identities = len(set(identity_ids))
        if n_identities <= 10:
            identity_palette = sns.color_palette("Set3", n_identities)
        else:
            identity_palette = sns.color_palette("husl", n_identities)
        
        unique_identities = sorted(list(set(identity_ids)))
        self.identity_colors = {identity_id: identity_palette[i % len(identity_palette)] 
                              for i, identity_id in enumerate(unique_identities)}
    
    def _reduce_dimensions(self, embeddings: np.ndarray, method: str = "pca", n_components: int = 2) -> np.ndarray:
        """
        Reduce embedding dimensions using specified method
        
        Args:
            embeddings: High-dimensional embeddings
            method: Reduction method ("pca", "tsne", "umap")
            n_components: Number of output dimensions (2 or 3)
            
        Returns:
            Reduced embeddings
        """
        if embeddings.shape[0] < 2:
            logger.warning("Not enough embeddings for dimensionality reduction")
            return embeddings
        
        if method == "pca":
            if self.pca_model is None or self.pca_model.n_components != n_components:
                self.pca_model = PCA(n_components=n_components, random_state=self.random_state)
            reduced = self.pca_model.fit_transform(embeddings)
            
        elif method == "tsne":
            # t-SNE parameters adjusted for gait embeddings
            perplexity = min(30, max(2, embeddings.shape[0] // 3))
            if self.tsne_model is None or self.tsne_model.n_components != n_components:
                self.tsne_model = TSNE(
                    n_components=n_components,
                    perplexity=perplexity,
                    random_state=self.random_state,
                    n_iter=1000,
                    learning_rate='auto'
                )
            reduced = self.tsne_model.fit_transform(embeddings)
            
        elif method == "umap":
            if not UMAP_AVAILABLE:
                print("⚠️  UMAP not available, falling back to PCA")
                return self._reduce_dimensions(embeddings, "pca", n_components)
            
            # UMAP parameters optimized for clustering
            n_neighbors = min(15, max(2, embeddings.shape[0] // 5))
            if self.umap_model is None or self.umap_model.n_components != n_components:
                self.umap_model = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=n_neighbors,
                    min_dist=0.1,
                    random_state=self.random_state
                )
            reduced = self.umap_model.fit_transform(embeddings)
            
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        return reduced
    
    def visualize_track_embeddings(self, 
                                 embeddings_by_track: Dict[int, List[Tuple[np.ndarray, str]]],
                                 method: str = "umap",
                                 save_path: Optional[str] = None,
                                 show_labels: bool = True,
                                 plot_type: str = "2d") -> plt.Figure:
        """
        Visualize embeddings colored by track ID
        
        Args:
            embeddings_by_track: Dict mapping track_id to list of (embedding, identity) tuples
            method: Dimensionality reduction method
            save_path: Path to save the plot
            show_labels: Whether to show track ID labels
            plot_type: "2d" or "3d"
            
        Returns:
            Matplotlib figure
        """
        # Prepare data
        all_embeddings = []
        track_ids = []
        identity_labels = []
        
        for track_id, embedding_list in embeddings_by_track.items():
            for embedding, identity in embedding_list:
                all_embeddings.append(embedding)
                track_ids.append(track_id)
                identity_labels.append(identity)
        
        if len(all_embeddings) == 0:
            logger.warning("No embeddings to visualize")
            return None
        
        embeddings_array = np.array(all_embeddings)
        self._prepare_color_schemes(track_ids, identity_labels)
        
        # Reduce dimensions
        n_components = 3 if plot_type == "3d" else 2
        reduced_embeddings = self._reduce_dimensions(embeddings_array, method, n_components)
        
        # Create plot
        if plot_type == "3d":
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            for track_id in set(track_ids):
                mask = np.array(track_ids) == track_id
                track_points = reduced_embeddings[mask]
                if len(track_points) > 0:
                    ax.scatter(track_points[:, 0], track_points[:, 1], track_points[:, 2],
                             c=[self.track_colors[track_id]], label=f'Track {track_id}',
                             alpha=0.7, s=50)
            
            ax.set_xlabel(f'{method.upper()} Component 1')
            ax.set_ylabel(f'{method.upper()} Component 2')
            ax.set_zlabel(f'{method.upper()} Component 3')
            
        else:
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            for track_id in set(track_ids):
                mask = np.array(track_ids) == track_id
                track_points = reduced_embeddings[mask]
                if len(track_points) > 0:
                    ax.scatter(track_points[:, 0], track_points[:, 1],
                             c=[self.track_colors[track_id]], label=f'Track {track_id}',
                             alpha=0.7, s=50)
                    
                    # Add track ID labels
                    if show_labels and len(track_points) > 0:
                        center = np.mean(track_points, axis=0)
                        ax.annotate(f'T{track_id}', center, fontsize=8, ha='center',
                                  bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
            
            ax.set_xlabel(f'{method.upper()} Component 1')
            ax.set_ylabel(f'{method.upper()} Component 2')
        
        plt.title(f'Track Embeddings Visualization ({method.upper()})\n'
                 f'{len(set(track_ids))} tracks, {len(all_embeddings)} embeddings')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Track visualization saved to {save_path}")
        
        return fig
    
    def visualize_identity_gallery(self,
                                 all_embeddings: List[Tuple[np.ndarray, str, int, str]],
                                 method: str = "umap",
                                 save_path: Optional[str] = None,
                                 show_labels: bool = True,
                                 plot_type: str = "2d") -> plt.Figure:
        """
        Visualize all embeddings with gallery identities highlighted
        
        Args:
            all_embeddings: List of (embedding, identity, track_id, type) tuples
            method: Dimensionality reduction method
            save_path: Path to save the plot
            show_labels: Whether to show identity labels
            plot_type: "2d" or "3d"
            
        Returns:
            Matplotlib figure
        """
        if len(all_embeddings) == 0:
            logger.warning("No embeddings to visualize")
            return None
        
        # Separate data
        embeddings = np.array([emb[0] for emb in all_embeddings])
        identities = [emb[1] for emb in all_embeddings]
        track_ids = [emb[2] for emb in all_embeddings]
        types = [emb[3] for emb in all_embeddings]
        
        self._prepare_color_schemes(track_ids, identities)
        
        # Reduce dimensions
        n_components = 3 if plot_type == "3d" else 2
        reduced_embeddings = self._reduce_dimensions(embeddings, method, n_components)
        
        # Create plot
        if plot_type == "3d":
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot track embeddings first
            track_mask = np.array(types) == "track_embedding"
            if np.any(track_mask):
                track_points = reduced_embeddings[track_mask]
                track_identities = np.array(identities)[track_mask]
                
                for identity in set(track_identities):
                    identity_mask = track_identities == identity
                    points = track_points[identity_mask]
                    if len(points) > 0:
                        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                                 c=[self.identity_colors.get(identity, 'gray')],
                                 alpha=0.6, s=30, label=f'{identity} (track)')
            
            # Plot gallery embeddings on top
            gallery_mask = np.array(types) == "gallery_embedding"
            if np.any(gallery_mask):
                gallery_points = reduced_embeddings[gallery_mask]
                gallery_identities = np.array(identities)[gallery_mask]
                
                ax.scatter(gallery_points[:, 0], gallery_points[:, 1], gallery_points[:, 2],
                         c='red', s=200, marker='*', alpha=0.9,
                         edgecolors='black', linewidths=2, label='Gallery embeddings')
                
                if show_labels:
                    for i, identity in enumerate(gallery_identities):
                        ax.text(gallery_points[i, 0], gallery_points[i, 1], gallery_points[i, 2],
                               identity, fontsize=8)
            
            ax.set_xlabel(f'{method.upper()} Component 1')
            ax.set_ylabel(f'{method.upper()} Component 2')
            ax.set_zlabel(f'{method.upper()} Component 3')
            
        else:
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Plot track embeddings first
            track_mask = np.array(types) == "track_embedding"
            if np.any(track_mask):
                track_points = reduced_embeddings[track_mask]
                track_identities = np.array(identities)[track_mask]
                track_track_ids = np.array(track_ids)[track_mask]
                
                for identity in set(track_identities):
                    identity_mask = track_identities == identity
                    points = track_points[identity_mask]
                    if len(points) > 0:
                        ax.scatter(points[:, 0], points[:, 1],
                                 c=[self.identity_colors.get(identity, 'gray')],
                                 alpha=0.6, s=30, label=f'{identity}')
            
            # Plot gallery embeddings on top
            gallery_mask = np.array(types) == "gallery_embedding"
            if np.any(gallery_mask):
                gallery_points = reduced_embeddings[gallery_mask]
                gallery_identities = np.array(identities)[gallery_mask]
                
                ax.scatter(gallery_points[:, 0], gallery_points[:, 1],
                         c='red', s=200, marker='*', alpha=0.9,
                         edgecolors='black', linewidths=2, label='Gallery embeddings')
                
                if show_labels:
                    for i, identity in enumerate(gallery_identities):
                        ax.annotate(identity, gallery_points[i], fontsize=8, ha='center',
                                  bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
            
            ax.set_xlabel(f'{method.upper()} Component 1')
            ax.set_ylabel(f'{method.upper()} Component 2')
        
        plt.title(f'Identity Gallery Visualization ({method.upper()})\n'
                 f'{len(set(identities))} identities, {len(embeddings)} total embeddings')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gallery visualization saved to {save_path}")
        
        return fig
    
    def create_interactive_visualization(self,
                                       all_embeddings: List[Tuple[np.ndarray, str, int, str]],
                                       method: str = "umap",
                                       save_path: Optional[str] = None) -> Optional[object]:
        """
        Create an interactive 3D visualization using Plotly
        
        Args:
            all_embeddings: List of (embedding, identity, track_id, type) tuples
            method: Dimensionality reduction method
            save_path: Path to save the HTML file
            
        Returns:
            Plotly figure or None if Plotly not available
        """
        if not PLOTLY_AVAILABLE:
            print("⚠️  Plotly not available, skipping interactive visualization")
            return None
        
        if len(all_embeddings) == 0:
            logger.warning("No embeddings to visualize")
            return None
        
        # Prepare data
        embeddings = np.array([emb[0] for emb in all_embeddings])
        identities = [emb[1] for emb in all_embeddings]
        track_ids = [emb[2] for emb in all_embeddings]
        types = [emb[3] for emb in all_embeddings]
        
        # Reduce to 3D
        reduced_embeddings = self._reduce_dimensions(embeddings, method, 3)
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'z': reduced_embeddings[:, 2],
            'identity': identities,
            'track_id': track_ids,
            'type': types,
            'hover_text': [f"Identity: {identities[i]}<br>Track: {track_ids[i]}<br>Type: {types[i]}" 
                          for i in range(len(identities))]
        })
        
        # Create figure
        fig = go.Figure()
        
        # Add track embeddings
        track_df = df[df['type'] == 'track_embedding']
        if not track_df.empty:
            fig.add_trace(go.Scatter3d(
                x=track_df['x'],
                y=track_df['y'],
                z=track_df['z'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=track_df['track_id'],
                    colorscale='Viridis',
                    opacity=0.7,
                    showscale=True,
                    colorbar=dict(title="Track ID")
                ),
                text=track_df['hover_text'],
                hovertemplate='%{text}<extra></extra>',
                name='Track Embeddings'
            ))
        
        # Add gallery embeddings
        gallery_df = df[df['type'] == 'gallery_embedding']
        if not gallery_df.empty:
            fig.add_trace(go.Scatter3d(
                x=gallery_df['x'],
                y=gallery_df['y'],
                z=gallery_df['z'],
                mode='markers+text',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='diamond',
                    line=dict(width=2, color='black'),
                    opacity=0.9
                ),
                text=gallery_df['identity'],
                textposition='top center',
                hovertemplate='%{text}<extra></extra>',
                name='Gallery Embeddings'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'Interactive Gait Embedding Visualization ({method.upper()})',
            scene=dict(
                xaxis_title=f'{method.upper()} Component 1',
                yaxis_title=f'{method.upper()} Component 2',
                zaxis_title=f'{method.upper()} Component 3',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.2, y=1.2, z=1.2)
                )
            ),
            width=1000,
            height=800,
            margin=dict(l=0, r=0, b=0, t=50)
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive visualization saved to {save_path}")
        
        return fig
    
    def analyze_embedding_consistency(self,
                                    embeddings_by_track: Dict[int, List[Tuple[np.ndarray, str]]],
                                    save_path: Optional[str] = None) -> Dict:
        """
        Analyze consistency of embeddings within each track
        
        Args:
            embeddings_by_track: Dict mapping track_id to list of (embedding, identity) tuples
            save_path: Path to save the analysis plot
            
        Returns:
            Dictionary with consistency metrics
        """
        consistency_metrics = {}
        track_consistencies = []
        track_labels = []
        
        for track_id, embedding_list in embeddings_by_track.items():
            if len(embedding_list) < 2:
                continue
            
            embeddings = np.array([emb[0] for emb in embedding_list])
            
            # Calculate pairwise similarities within track
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    similarities.append(sim)
            
            if similarities:
                track_consistency = np.mean(similarities)
                consistency_metrics[track_id] = {
                    'mean_similarity': track_consistency,
                    'std_similarity': np.std(similarities),
                    'min_similarity': np.min(similarities),
                    'max_similarity': np.max(similarities),
                    'num_embeddings': len(embeddings)
                }
                track_consistencies.append(track_consistency)
                track_labels.append(f'Track {track_id}')
        
        # Create visualization
        if track_consistencies and save_path:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Bar plot of consistency per track
            ax1.bar(range(len(track_consistencies)), track_consistencies)
            ax1.set_xlabel('Track')
            ax1.set_ylabel('Mean Intra-Track Similarity')
            ax1.set_title('Embedding Consistency by Track')
            ax1.set_xticks(range(len(track_labels)))
            ax1.set_xticklabels(track_labels, rotation=45)
            
            # Histogram of consistency values
            ax2.hist(track_consistencies, bins=min(10, len(track_consistencies)), alpha=0.7)
            ax2.axvline(np.mean(track_consistencies), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(track_consistencies):.3f}')
            ax2.set_xlabel('Mean Intra-Track Similarity')
            ax2.set_ylabel('Number of Tracks')
            ax2.set_title('Distribution of Track Consistency')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Consistency analysis saved to {save_path}")
        
        # Overall metrics
        if track_consistencies:
            consistency_metrics['overall'] = {
                'mean_track_consistency': np.mean(track_consistencies),
                'std_track_consistency': np.std(track_consistencies),
                'min_track_consistency': np.min(track_consistencies),
                'max_track_consistency': np.max(track_consistencies),
                'num_tracks_analyzed': len(track_consistencies)
            }
        
        return consistency_metrics
    
    def create_comprehensive_report(self,
                                  gallery_manager,
                                  output_dir: str,
                                  methods: List[str] = ["pca", "tsne", "umap"]):
        """
        Create a comprehensive visualization report
        
        Args:
            gallery_manager: IdentityGalleryManager instance
            output_dir: Directory to save all visualizations
            methods: List of dimensionality reduction methods to use
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get data from gallery manager
        all_embeddings = gallery_manager.get_all_embeddings()
        embeddings_by_track = gallery_manager.get_track_embeddings_by_track()
        
        if not all_embeddings:
            logger.warning("No embeddings available for visualization")
            return
        
        logger.info(f"📊 Creating comprehensive visualization report")
        logger.info(f"   Output directory: {output_path}")
        logger.info(f"   Total embeddings: {len(all_embeddings)}")
        logger.info(f"   Tracks: {len(embeddings_by_track)}")
        
        # Create visualizations for each method
        available_methods = []
        for method in methods:
            if method == "umap" and not UMAP_AVAILABLE:
                print(f"⚠️  Skipping {method} - UMAP not available")
                continue
            available_methods.append(method)
            
            method_dir = output_path / method
            method_dir.mkdir(exist_ok=True)
            
            # Track embeddings visualization
            fig_track = self.visualize_track_embeddings(
                embeddings_by_track,
                method=method,
                save_path=str(method_dir / f"track_embeddings_{method}.png")
            )
            if fig_track:
                plt.close(fig_track)
            
            # Gallery visualization
            fig_gallery = self.visualize_identity_gallery(
                all_embeddings,
                method=method,
                save_path=str(method_dir / f"identity_gallery_{method}.png")
            )
            if fig_gallery:
                plt.close(fig_gallery)
            
            # 3D visualizations
            fig_track_3d = self.visualize_track_embeddings(
                embeddings_by_track,
                method=method,
                plot_type="3d",
                save_path=str(method_dir / f"track_embeddings_{method}_3d.png")
            )
            if fig_track_3d:
                plt.close(fig_track_3d)
            
            fig_gallery_3d = self.visualize_identity_gallery(
                all_embeddings,
                method=method,
                plot_type="3d",
                save_path=str(method_dir / f"identity_gallery_{method}_3d.png")
            )
            if fig_gallery_3d:
                plt.close(fig_gallery_3d)
        
        # Interactive visualization (if Plotly available)
        if PLOTLY_AVAILABLE:
            interactive_fig = self.create_interactive_visualization(
                all_embeddings,
                method="umap" if UMAP_AVAILABLE else "pca",
                save_path=str(output_path / "interactive_visualization.html")
            )
        else:
            print("⚠️  Skipping interactive visualization - Plotly not available")
        
        # Consistency analysis
        consistency_metrics = self.analyze_embedding_consistency(
            embeddings_by_track,
            save_path=str(output_path / "embedding_consistency.png")
        )
        
        # Save metrics
        gallery_summary = gallery_manager.get_gallery_summary()
        full_report = {
            'generation_time': datetime.now().isoformat(),
            'gallery_summary': gallery_summary,
            'consistency_metrics': consistency_metrics,
            'visualization_methods': available_methods,
            'total_visualizations_created': len(available_methods) * 4 + (1 if PLOTLY_AVAILABLE else 0) + 1,  # 4 per method + interactive + consistency
            'dependencies_available': {
                'umap': UMAP_AVAILABLE,
                'plotly': PLOTLY_AVAILABLE
            }
        }
        
        with open(output_path / "visualization_report.json", 'w') as f:
            import json
            json.dump(full_report, f, indent=2, default=str)
        
        logger.info(f"✅ Comprehensive report created in {output_path}")
        logger.info(f"   📈 {len(available_methods)} reduction methods analyzed")
        logger.info(f"   🎯 {gallery_summary['num_identities']} identities visualized")
        logger.info(f"   📊 {gallery_summary['total_tracks']} tracks analyzed")
        if not UMAP_AVAILABLE:
            logger.info(f"   ⚠️  UMAP not available - install with: pip install umap-learn")
        if not PLOTLY_AVAILABLE:
            logger.info(f"   ⚠️  Plotly not available - install with: pip install plotly")
        
        return full_report
