"""
Advanced Embedding Clustering and Visualization for Gait Identification
Provides comprehensive clustering analysis and visualization of XGait embeddings
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.cluster.hierarchy import dendrogram, linkage
from datetime import datetime
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class EmbeddingClusterAnalyzer:
    """
    Advanced clustering analysis for XGait embeddings with comprehensive visualization
    """
    
    def __init__(self, output_dir: str = "clustering_analysis"):
        """
        Initialize clustering analyzer
        
        Args:
            output_dir: Directory to save analysis results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        logger.info(f"📊 Clustering analyzer initialized - output: {self.output_dir}")
    
    def analyze_gallery_embeddings(self, gallery: Dict[str, np.ndarray], 
                                 track_features: Dict[int, List[np.ndarray]] = None,
                                 save_results: bool = True) -> Dict[str, Any]:
        """
        Comprehensive analysis of gallery embeddings with multiple clustering methods
        
        Args:
            gallery: Gallery embeddings {person_id: embedding}
            track_features: Track features {track_id: [embeddings]}
            save_results: Whether to save analysis results
            
        Returns:
            Analysis results dictionary
        """
        if not gallery:
            logger.warning("📊 Empty gallery - no embeddings to analyze")
            return {}
        
        logger.info(f"📊 Starting clustering analysis - {len(gallery)} gallery persons")
        
        # Prepare data
        person_ids = list(gallery.keys())
        embeddings = np.array([gallery[pid] for pid in person_ids])
        
        # Add track features if provided
        track_embeddings = []
        track_labels = []
        track_info = []
        
        if track_features:
            for track_id, features_list in track_features.items():
                for i, features in enumerate(features_list):
                    track_embeddings.append(features)
                    track_labels.append(f"track_{track_id}")
                    track_info.append({'track_id': track_id, 'feature_idx': i})
        
        # Combine gallery and track embeddings
        all_embeddings = embeddings
        all_labels = person_ids
        all_info = [{'type': 'gallery', 'person_id': pid} for pid in person_ids]
        
        if track_embeddings:
            all_embeddings = np.vstack([embeddings, np.array(track_embeddings)])
            all_labels = person_ids + track_labels
            all_info = all_info + [{'type': 'track', **info} for info in track_info]
        
        logger.info(f"📊 Analyzing {len(all_embeddings)} total embeddings")
        
        # Perform comprehensive analysis
        results = {
            'timestamp': datetime.now().isoformat(),
            'num_gallery_persons': len(gallery),
            'num_track_features': len(track_embeddings) if track_embeddings else 0,
            'embedding_dimension': embeddings.shape[1],
            'analysis_methods': []
        }
        
        # 1. Dimensionality Reduction Analysis
        dim_reduction_results = self._analyze_dimensionality_reduction(
            all_embeddings, all_labels, all_info
        )
        results['dimensionality_reduction'] = dim_reduction_results
        
        # 2. Clustering Analysis
        clustering_results = self._analyze_clustering(
            all_embeddings, all_labels, all_info
        )
        results['clustering'] = clustering_results
        
        # 3. Similarity Analysis
        similarity_results = self._analyze_similarity_patterns(
            all_embeddings, all_labels, all_info
        )
        results['similarity'] = similarity_results
        
        # 4. Quality Assessment
        quality_results = self._assess_embedding_quality(
            embeddings, person_ids, track_embeddings if track_embeddings else None
        )
        results['quality'] = quality_results
        
        # 5. Create comprehensive visualizations
        if save_results:
            viz_results = self._create_comprehensive_visualizations(
                all_embeddings, all_labels, all_info, results
            )
            results['visualizations'] = viz_results
            
            # Save analysis report
            report_path = self._save_analysis_report(results)
            results['report_path'] = str(report_path)
        
        logger.info(f"✅ Clustering analysis completed - {len(results)} analysis components")
        return results
    
    def _analyze_dimensionality_reduction(self, embeddings: np.ndarray, 
                                       labels: List[str], 
                                       info: List[Dict]) -> Dict[str, Any]:
        """Analyze embeddings using various dimensionality reduction techniques"""
        logger.info("🔍 Performing dimensionality reduction analysis...")
        
        results = {}
        
        # PCA Analysis
        try:
            pca = PCA(n_components=min(50, embeddings.shape[1], embeddings.shape[0]-1))
            pca_embeddings = pca.fit_transform(embeddings)
            
            # Calculate explained variance
            explained_variance = np.cumsum(pca.explained_variance_ratio_)
            
            results['pca'] = {
                'embeddings_2d': PCA(n_components=2).fit_transform(embeddings),
                'embeddings_3d': PCA(n_components=3).fit_transform(embeddings),
                'explained_variance': explained_variance[:10].tolist(),
                'total_variance_10_components': float(explained_variance[9] if len(explained_variance) > 9 else explained_variance[-1])
            }
            logger.info(f"✅ PCA: {results['pca']['total_variance_10_components']:.3f} variance in 10 components")
            
        except Exception as e:
            logger.warning(f"⚠️ PCA analysis failed: {e}")
            results['pca'] = {'error': str(e)}
        
        # t-SNE Analysis
        try:
            if len(embeddings) > 3:  # t-SNE needs at least 4 samples
                tsne_2d = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
                tsne_embeddings_2d = tsne_2d.fit_transform(embeddings)
                
                results['tsne'] = {
                    'embeddings_2d': tsne_embeddings_2d,
                    'kl_divergence': float(tsne_2d.kl_divergence_)
                }
                logger.info(f"✅ t-SNE: KL divergence = {results['tsne']['kl_divergence']:.3f}")
            else:
                results['tsne'] = {'error': 'Insufficient samples for t-SNE'}
                
        except Exception as e:
            logger.warning(f"⚠️ t-SNE analysis failed: {e}")
            results['tsne'] = {'error': str(e)}
        
        # UMAP Analysis (if available)
        try:
            if len(embeddings) > 3 and UMAP_AVAILABLE:
                umap_reducer = UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(embeddings)-1))
                umap_embeddings_2d = umap_reducer.fit_transform(embeddings)
                
                results['umap'] = {
                    'embeddings_2d': umap_embeddings_2d
                }
                logger.info("✅ UMAP completed successfully")
            else:
                if not UMAP_AVAILABLE:
                    results['umap'] = {'error': 'UMAP not installed'}
                else:
                    results['umap'] = {'error': 'Insufficient samples for UMAP'}
                
        except Exception as e:
            logger.warning(f"⚠️ UMAP analysis failed: {e}")
            results['umap'] = {'error': str(e)}
        
        return results
    
    def _analyze_clustering(self, embeddings: np.ndarray, 
                          labels: List[str], 
                          info: List[Dict]) -> Dict[str, Any]:
        """Perform clustering analysis with multiple algorithms"""
        logger.info("🎯 Performing clustering analysis...")
        
        results = {}
        
        # Determine optimal number of clusters using elbow method and silhouette
        max_clusters = min(10, len(embeddings) - 1)
        if max_clusters < 2:
            logger.warning("⚠️ Too few samples for clustering analysis")
            return {'error': 'Insufficient samples for clustering'}
        
        # K-means clustering analysis
        try:
            inertias = []
            silhouette_scores = []
            k_range = range(2, max_clusters + 1)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)
                
                inertias.append(kmeans.inertia_)
                if len(set(cluster_labels)) > 1:  # Need at least 2 clusters for silhouette
                    sil_score = silhouette_score(embeddings, cluster_labels)
                    silhouette_scores.append(sil_score)
                else:
                    silhouette_scores.append(0)
            
            # Find optimal k using silhouette score
            optimal_k = k_range[np.argmax(silhouette_scores)]
            
            # Perform final clustering with optimal k
            optimal_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            optimal_labels = optimal_kmeans.fit_predict(embeddings)
            
            results['kmeans'] = {
                'optimal_k': int(optimal_k),
                'inertias': inertias,
                'silhouette_scores': silhouette_scores,
                'cluster_labels': optimal_labels.tolist(),
                'cluster_centers': optimal_kmeans.cluster_centers_,
                'best_silhouette_score': float(max(silhouette_scores))
            }
            
            logger.info(f"✅ K-means: optimal k={optimal_k}, silhouette={max(silhouette_scores):.3f}")
            
        except Exception as e:
            logger.warning(f"⚠️ K-means clustering failed: {e}")
            results['kmeans'] = {'error': str(e)}
        
        # DBSCAN clustering
        try:
            # Try multiple eps values
            eps_values = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
            best_eps = 0.5
            best_score = -1
            best_labels = None
            
            for eps in eps_values:
                dbscan = DBSCAN(eps=eps, min_samples=2)
                cluster_labels = dbscan.fit_predict(embeddings)
                
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                if n_clusters > 1:
                    score = silhouette_score(embeddings, cluster_labels)
                    if score > best_score:
                        best_score = score
                        best_eps = eps
                        best_labels = cluster_labels
            
            if best_labels is not None:
                n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
                n_noise = list(best_labels).count(-1)
                
                results['dbscan'] = {
                    'optimal_eps': float(best_eps),
                    'n_clusters': int(n_clusters),
                    'n_noise_points': int(n_noise),
                    'cluster_labels': best_labels.tolist(),
                    'silhouette_score': float(best_score)
                }
                
                logger.info(f"✅ DBSCAN: {n_clusters} clusters, {n_noise} noise points, eps={best_eps}")
            else:
                results['dbscan'] = {'error': 'No valid clustering found'}
                
        except Exception as e:
            logger.warning(f"⚠️ DBSCAN clustering failed: {e}")
            results['dbscan'] = {'error': str(e)}
        
        # Hierarchical clustering
        try:
            if len(embeddings) > 2:
                linkage_matrix = linkage(embeddings, method='ward')
                
                # Find optimal number of clusters using distance threshold
                distances = linkage_matrix[:, 2]
                distance_diffs = np.diff(distances)
                optimal_clusters = len(distances) - np.argmax(distance_diffs[::-1])
                optimal_clusters = max(2, min(optimal_clusters, max_clusters))
                
                hierarchical = AgglomerativeClustering(n_clusters=optimal_clusters)
                cluster_labels = hierarchical.fit_predict(embeddings)
                
                sil_score = silhouette_score(embeddings, cluster_labels) if len(set(cluster_labels)) > 1 else 0
                
                results['hierarchical'] = {
                    'optimal_clusters': int(optimal_clusters),
                    'cluster_labels': cluster_labels.tolist(),
                    'linkage_matrix': linkage_matrix.tolist(),
                    'silhouette_score': float(sil_score)
                }
                
                logger.info(f"✅ Hierarchical: {optimal_clusters} clusters, silhouette={sil_score:.3f}")
            else:
                results['hierarchical'] = {'error': 'Insufficient samples'}
                
        except Exception as e:
            logger.warning(f"⚠️ Hierarchical clustering failed: {e}")
            results['hierarchical'] = {'error': str(e)}
        
        return results
    
    def _analyze_similarity_patterns(self, embeddings: np.ndarray, 
                                   labels: List[str], 
                                   info: List[Dict]) -> Dict[str, Any]:
        """Analyze similarity patterns in embeddings"""
        logger.info("🔍 Analyzing similarity patterns...")
        
        results = {}
        
        try:
            # Cosine similarity matrix
            cosine_sim_matrix = cosine_similarity(embeddings)
            
            # Euclidean distance matrix
            euclidean_dist_matrix = euclidean_distances(embeddings)
            
            # Normalize euclidean distances to [0, 1] for comparison
            euclidean_sim_matrix = 1 - (euclidean_dist_matrix / euclidean_dist_matrix.max())
            
            # Statistics
            cosine_stats = {
                'mean': float(np.mean(cosine_sim_matrix)),
                'std': float(np.std(cosine_sim_matrix)),
                'min': float(np.min(cosine_sim_matrix)),
                'max': float(np.max(cosine_sim_matrix))
            }
            
            euclidean_stats = {
                'mean': float(np.mean(euclidean_dist_matrix)),
                'std': float(np.std(euclidean_dist_matrix)),
                'min': float(np.min(euclidean_dist_matrix)),
                'max': float(np.max(euclidean_dist_matrix))
            }
            
            # Find most similar and dissimilar pairs
            np.fill_diagonal(cosine_sim_matrix, -1)  # Ignore self-similarity
            most_similar_idx = np.unravel_index(np.argmax(cosine_sim_matrix), cosine_sim_matrix.shape)
            most_dissimilar_idx = np.unravel_index(np.argmin(cosine_sim_matrix), cosine_sim_matrix.shape)
            
            results = {
                'cosine_similarity': {
                    'matrix': cosine_sim_matrix,
                    'statistics': cosine_stats,
                    'most_similar_pair': {
                        'labels': [labels[most_similar_idx[0]], labels[most_similar_idx[1]]],
                        'similarity': float(cosine_sim_matrix[most_similar_idx])
                    },
                    'most_dissimilar_pair': {
                        'labels': [labels[most_dissimilar_idx[0]], labels[most_dissimilar_idx[1]]],
                        'similarity': float(cosine_sim_matrix[most_dissimilar_idx])
                    }
                },
                'euclidean_distance': {
                    'matrix': euclidean_dist_matrix,
                    'statistics': euclidean_stats
                }
            }
            
            logger.info(f"✅ Similarity analysis: cosine mean={cosine_stats['mean']:.3f}")
            
        except Exception as e:
            logger.warning(f"⚠️ Similarity analysis failed: {e}")
            results = {'error': str(e)}
        
        return results
    
    def _assess_embedding_quality(self, gallery_embeddings: np.ndarray, 
                                person_ids: List[str],
                                track_embeddings: List[np.ndarray] = None) -> Dict[str, Any]:
        """Assess the quality of embeddings for identification"""
        logger.info("📈 Assessing embedding quality...")
        
        results = {}
        
        try:
            # Inter-class vs Intra-class distances
            if len(gallery_embeddings) > 1:
                # Calculate pairwise distances between different persons (inter-class)
                inter_class_distances = []
                for i in range(len(gallery_embeddings)):
                    for j in range(i + 1, len(gallery_embeddings)):
                        dist = np.linalg.norm(gallery_embeddings[i] - gallery_embeddings[j])
                        inter_class_distances.append(dist)
                
                # Calculate intra-class distances (if track embeddings available)
                intra_class_distances = []
                if track_embeddings:
                    # For now, assume all track embeddings are from different persons
                    # In a real scenario, you'd need ground truth person assignments
                    for track_emb in track_embeddings:
                        for gallery_emb in gallery_embeddings:
                            dist = np.linalg.norm(track_emb - gallery_emb)
                            intra_class_distances.append(dist)
                
                # Quality metrics
                inter_class_mean = np.mean(inter_class_distances)
                inter_class_std = np.std(inter_class_distances)
                
                results['distance_analysis'] = {
                    'inter_class_mean': float(inter_class_mean),
                    'inter_class_std': float(inter_class_std),
                    'inter_class_distances': inter_class_distances
                }
                
                if intra_class_distances:
                    intra_class_mean = np.mean(intra_class_distances)
                    separability_ratio = inter_class_mean / intra_class_mean if intra_class_mean > 0 else float('inf')
                    
                    results['distance_analysis'].update({
                        'intra_class_mean': float(intra_class_mean),
                        'intra_class_std': float(np.std(intra_class_distances)),
                        'separability_ratio': float(separability_ratio),
                        'intra_class_distances': intra_class_distances
                    })
            
            # Embedding statistics
            embedding_stats = {
                'dimension': int(gallery_embeddings.shape[1]),
                'mean_norm': float(np.mean([np.linalg.norm(emb) for emb in gallery_embeddings])),
                'std_norm': float(np.std([np.linalg.norm(emb) for emb in gallery_embeddings])),
                'mean_value': float(np.mean(gallery_embeddings)),
                'std_value': float(np.std(gallery_embeddings))
            }
            
            results['embedding_statistics'] = embedding_stats
            
            # Overall quality assessment
            quality_score = 0.0
            quality_factors = []
            
            if 'distance_analysis' in results and 'separability_ratio' in results['distance_analysis']:
                sep_ratio = results['distance_analysis']['separability_ratio']
                if sep_ratio > 2.0:
                    quality_score += 0.4
                    quality_factors.append("Good separability")
                elif sep_ratio > 1.5:
                    quality_score += 0.2
                    quality_factors.append("Moderate separability")
                else:
                    quality_factors.append("Poor separability")
            
            if embedding_stats['mean_norm'] > 0.1:
                quality_score += 0.3
                quality_factors.append("Good embedding magnitude")
            
            if embedding_stats['std_value'] < 1.0:
                quality_score += 0.3
                quality_factors.append("Stable embedding values")
            
            results['quality_assessment'] = {
                'overall_score': float(quality_score),
                'quality_factors': quality_factors,
                'assessment': 'Excellent' if quality_score > 0.8 else 
                           'Good' if quality_score > 0.6 else
                           'Moderate' if quality_score > 0.4 else 'Poor'
            }
            
            logger.info(f"✅ Quality assessment: {results['quality_assessment']['assessment']}")
            
        except Exception as e:
            logger.warning(f"⚠️ Quality assessment failed: {e}")
            results = {'error': str(e)}
        
        return results
    
    def _create_comprehensive_visualizations(self, embeddings: np.ndarray, 
                                           labels: List[str], 
                                           info: List[Dict],
                                           analysis_results: Dict) -> Dict[str, str]:
        """Create comprehensive visualization plots"""
        logger.info("🎨 Creating comprehensive visualizations...")
        
        viz_paths = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # 1. Dimensionality Reduction Visualization
            if 'dimensionality_reduction' in analysis_results:
                viz_paths['dimensionality_reduction'] = self._plot_dimensionality_reduction(
                    analysis_results['dimensionality_reduction'], labels, info, timestamp
                )
            
            # 2. Clustering Results Visualization
            if 'clustering' in analysis_results:
                viz_paths['clustering'] = self._plot_clustering_results(
                    embeddings, analysis_results['clustering'], labels, info, timestamp
                )
            
            # 3. Similarity Heatmaps
            if 'similarity' in analysis_results:
                viz_paths['similarity'] = self._plot_similarity_heatmaps(
                    analysis_results['similarity'], labels, timestamp
                )
            
            # 4. Quality Assessment Plots
            if 'quality' in analysis_results:
                viz_paths['quality'] = self._plot_quality_assessment(
                    analysis_results['quality'], timestamp
                )
            
            # 5. Comprehensive Dashboard
            viz_paths['dashboard'] = self._create_analysis_dashboard(
                analysis_results, labels, info, timestamp
            )
            
        except Exception as e:
            logger.warning(f"⚠️ Visualization creation failed: {e}")
            viz_paths['error'] = str(e)
        
        return viz_paths
    
    def _plot_dimensionality_reduction(self, dim_results: Dict, labels: List[str], 
                                     info: List[Dict], timestamp: str) -> str:
        """Plot dimensionality reduction results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dimensionality Reduction Analysis', fontsize=16, fontweight='bold')
        
        # Color mapping for different types
        colors = []
        for item in info:
            if item['type'] == 'gallery':
                colors.append('red')
            else:
                colors.append('blue')
        
        # PCA 2D
        if 'pca' in dim_results and 'embeddings_2d' in dim_results['pca']:
            ax = axes[0, 0]
            pca_2d = dim_results['pca']['embeddings_2d']
            scatter = ax.scatter(pca_2d[:, 0], pca_2d[:, 1], c=colors, alpha=0.7, s=60)
            ax.set_title('PCA (2D)')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            
            # Add labels for gallery embeddings
            for i, (label, item) in enumerate(zip(labels, info)):
                if item['type'] == 'gallery':
                    ax.annotate(label, (pca_2d[i, 0], pca_2d[i, 1]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # PCA Explained Variance
        if 'pca' in dim_results and 'explained_variance' in dim_results['pca']:
            ax = axes[0, 1]
            explained_var = dim_results['pca']['explained_variance']
            ax.plot(range(1, len(explained_var) + 1), explained_var, 'bo-')
            ax.set_title('PCA Explained Variance')
            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Cumulative Explained Variance')
            ax.grid(True)
        
        # t-SNE 2D
        if 'tsne' in dim_results and 'embeddings_2d' in dim_results['tsne']:
            ax = axes[1, 0]
            tsne_2d = dim_results['tsne']['embeddings_2d']
            scatter = ax.scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=colors, alpha=0.7, s=60)
            ax.set_title('t-SNE (2D)')
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            
            # Add labels for gallery embeddings
            for i, (label, item) in enumerate(zip(labels, info)):
                if item['type'] == 'gallery':
                    ax.annotate(label, (tsne_2d[i, 0], tsne_2d[i, 1]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # UMAP 2D (if available)
        if 'umap' in dim_results and 'embeddings_2d' in dim_results['umap']:
            ax = axes[1, 1]
            umap_2d = dim_results['umap']['embeddings_2d']
            scatter = ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c=colors, alpha=0.7, s=60)
            ax.set_title('UMAP (2D)')
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            
            # Add labels for gallery embeddings
            for i, (label, item) in enumerate(zip(labels, info)):
                if item['type'] == 'gallery':
                    ax.annotate(label, (umap_2d[i, 0], umap_2d[i, 1]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        else:
            # Hide the subplot if UMAP is not available
            axes[1, 1].axis('off')
            axes[1, 1].text(0.5, 0.5, 'UMAP not available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', label='Gallery'),
                          Patch(facecolor='blue', label='Track Features')]
        fig.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"dimensionality_reduction_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_clustering_results(self, embeddings: np.ndarray, clustering_results: Dict,
                               labels: List[str], info: List[Dict], timestamp: str) -> str:
        """Plot clustering analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Clustering Analysis Results', fontsize=16, fontweight='bold')
        
        # K-means results
        if 'kmeans' in clustering_results and 'cluster_labels' in clustering_results['kmeans']:
            ax = axes[0, 0]
            
            # Plot elbow curve
            if 'inertias' in clustering_results['kmeans']:
                inertias = clustering_results['kmeans']['inertias']
                k_range = range(2, len(inertias) + 2)
                ax.plot(k_range, inertias, 'bo-')
                ax.set_title('K-means Elbow Curve')
                ax.set_xlabel('Number of Clusters (k)')
                ax.set_ylabel('Inertia')
                ax.grid(True)
                
                # Mark optimal k
                optimal_k = clustering_results['kmeans']['optimal_k']
                optimal_idx = optimal_k - 2
                if 0 <= optimal_idx < len(inertias):
                    ax.plot(optimal_k, inertias[optimal_idx], 'ro', markersize=10, 
                           label=f'Optimal k={optimal_k}')
                    ax.legend()
        
        # Silhouette scores
        if 'kmeans' in clustering_results and 'silhouette_scores' in clustering_results['kmeans']:
            ax = axes[0, 1]
            sil_scores = clustering_results['kmeans']['silhouette_scores']
            k_range = range(2, len(sil_scores) + 2)
            ax.plot(k_range, sil_scores, 'go-')
            ax.set_title('Silhouette Score vs Number of Clusters')
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Silhouette Score')
            ax.grid(True)
            
            # Mark optimal k
            optimal_k = clustering_results['kmeans']['optimal_k']
            optimal_idx = optimal_k - 2
            if 0 <= optimal_idx < len(sil_scores):
                ax.plot(optimal_k, sil_scores[optimal_idx], 'ro', markersize=10)
        
        # Hierarchical clustering dendrogram
        if 'hierarchical' in clustering_results and 'linkage_matrix' in clustering_results['hierarchical']:
            ax = axes[1, 0]
            linkage_matrix = np.array(clustering_results['hierarchical']['linkage_matrix'])
            dendrogram(linkage_matrix, ax=ax, labels=labels, leaf_rotation=90)
            ax.set_title('Hierarchical Clustering Dendrogram')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Distance')
        
        # Clustering comparison
        ax = axes[1, 1]
        methods = []
        scores = []
        
        for method in ['kmeans', 'dbscan', 'hierarchical']:
            if method in clustering_results and 'silhouette_score' in clustering_results[method]:
                methods.append(method.upper())
                scores.append(clustering_results[method]['silhouette_score'])
        
        if methods:
            bars = ax.bar(methods, scores, color=['blue', 'green', 'orange'][:len(methods)])
            ax.set_title('Clustering Method Comparison')
            ax.set_ylabel('Silhouette Score')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"clustering_results_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_similarity_heatmaps(self, similarity_results: Dict, labels: List[str], timestamp: str) -> str:
        """Plot similarity heatmaps"""
        if 'error' in similarity_results:
            return ""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Similarity Analysis', fontsize=16, fontweight='bold')
        
        # Cosine similarity heatmap
        if 'cosine_similarity' in similarity_results:
            ax = axes[0]
            cosine_matrix = similarity_results['cosine_similarity']['matrix']
            
            # Fill diagonal with 1 for display
            np.fill_diagonal(cosine_matrix, 1)
            
            im = ax.imshow(cosine_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
            ax.set_title('Cosine Similarity Matrix')
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticklabels(labels)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Cosine Similarity')
            
            # Add text annotations
            for i in range(len(labels)):
                for j in range(len(labels)):
                    text = ax.text(j, i, f'{cosine_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8)
        
        # Euclidean distance heatmap
        if 'euclidean_distance' in similarity_results:
            ax = axes[1]
            euclidean_matrix = similarity_results['euclidean_distance']['matrix']
            
            im = ax.imshow(euclidean_matrix, cmap='YlOrRd')
            ax.set_title('Euclidean Distance Matrix')
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticklabels(labels)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Euclidean Distance')
            
            # Add text annotations
            for i in range(len(labels)):
                for j in range(len(labels)):
                    text = ax.text(j, i, f'{euclidean_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="white", fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"similarity_heatmaps_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_quality_assessment(self, quality_results: Dict, timestamp: str) -> str:
        """Plot quality assessment results"""
        if 'error' in quality_results:
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Embedding Quality Assessment', fontsize=16, fontweight='bold')
        
        # Distance distribution comparison
        if 'distance_analysis' in quality_results:
            ax = axes[0, 0]
            dist_data = quality_results['distance_analysis']
            
            if 'inter_class_distances' in dist_data:
                ax.hist(dist_data['inter_class_distances'], bins=20, alpha=0.7, 
                       label='Inter-class', color='blue')
            
            if 'intra_class_distances' in dist_data:
                ax.hist(dist_data['intra_class_distances'], bins=20, alpha=0.7, 
                       label='Intra-class', color='red')
            
            ax.set_title('Distance Distribution')
            ax.set_xlabel('Distance')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Embedding statistics
        if 'embedding_statistics' in quality_results:
            ax = axes[0, 1]
            stats = quality_results['embedding_statistics']
            
            metrics = ['Mean Norm', 'Std Norm', 'Mean Value', 'Std Value']
            values = [stats.get('mean_norm', 0), stats.get('std_norm', 0),
                     stats.get('mean_value', 0), stats.get('std_value', 0)]
            
            bars = ax.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
            ax.set_title('Embedding Statistics')
            ax.set_ylabel('Value')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.3f}', ha='center', va='bottom', rotation=45)
        
        # Quality score visualization
        if 'quality_assessment' in quality_results:
            ax = axes[1, 0]
            assessment = quality_results['quality_assessment']
            
            score = assessment.get('overall_score', 0)
            
            # Create a gauge-like visualization
            theta = np.linspace(0, np.pi, 100)
            r = np.ones_like(theta)
            
            ax = plt.subplot(2, 2, 3, projection='polar')
            ax.set_theta_zero_location('W')
            ax.set_theta_direction(1)
            ax.set_ylim(0, 1)
            
            # Color segments
            colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
            for i, color in enumerate(colors):
                start_angle = i * np.pi / 5
                end_angle = (i + 1) * np.pi / 5
                theta_seg = np.linspace(start_angle, end_angle, 20)
                r_seg = np.ones_like(theta_seg)
                ax.fill_between(theta_seg, 0, r_seg, color=color, alpha=0.7)
            
            # Add score indicator
            score_angle = score * np.pi
            ax.plot([score_angle, score_angle], [0, 1], 'black', linewidth=3)
            
            ax.set_title(f'Quality Score: {score:.2f}\n({assessment.get("assessment", "Unknown")})')
            ax.set_ylim(0, 1)
            ax.set_rticks([])
        
        # Quality factors
        if 'quality_assessment' in quality_results:
            ax = axes[1, 1]
            factors = quality_results['quality_assessment'].get('quality_factors', [])
            
            if factors:
                y_pos = np.arange(len(factors))
                ax.barh(y_pos, [1] * len(factors), color='lightblue')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(factors)
                ax.set_xlabel('Factor Present')
                ax.set_title('Quality Factors')
                ax.set_xlim(0, 1.2)
            else:
                ax.text(0.5, 0.5, 'No quality factors identified', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Quality Factors')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"quality_assessment_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _create_analysis_dashboard(self, analysis_results: Dict, labels: List[str], 
                                 info: List[Dict], timestamp: str) -> str:
        """Create a comprehensive analysis dashboard"""
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle('XGait Embedding Analysis Dashboard', fontsize=20, fontweight='bold')
        
        # Add summary text
        summary_ax = fig.add_subplot(gs[0, :2])
        summary_ax.axis('off')
        
        summary_text = f"""
Analysis Summary:
• Total Embeddings: {len(labels)}
• Gallery Persons: {analysis_results.get('num_gallery_persons', 0)}
• Track Features: {analysis_results.get('num_track_features', 0)}
• Embedding Dimension: {analysis_results.get('embedding_dimension', 'Unknown')}
• Analysis Timestamp: {analysis_results.get('timestamp', 'Unknown')}
        """
        
        if 'quality' in analysis_results and 'quality_assessment' in analysis_results['quality']:
            quality = analysis_results['quality']['quality_assessment']
            summary_text += f"\n• Overall Quality: {quality.get('assessment', 'Unknown')} ({quality.get('overall_score', 0):.2f})"
        
        summary_ax.text(0.05, 0.95, summary_text, transform=summary_ax.transAxes, 
                       fontsize=12, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # Quick stats
        stats_ax = fig.add_subplot(gs[0, 2:])
        stats_ax.axis('off')
        
        if 'clustering' in analysis_results:
            clustering = analysis_results['clustering']
            stats_text = "Clustering Results:\n"
            
            if 'kmeans' in clustering and 'optimal_k' in clustering['kmeans']:
                stats_text += f"• K-means optimal k: {clustering['kmeans']['optimal_k']}\n"
                stats_text += f"• K-means silhouette: {clustering['kmeans'].get('best_silhouette_score', 0):.3f}\n"
            
            if 'dbscan' in clustering and 'n_clusters' in clustering['dbscan']:
                stats_text += f"• DBSCAN clusters: {clustering['dbscan']['n_clusters']}\n"
                stats_text += f"• DBSCAN silhouette: {clustering['dbscan'].get('silhouette_score', 0):.3f}\n"
            
            stats_ax.text(0.05, 0.95, stats_text, transform=stats_ax.transAxes,
                         fontsize=11, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # Add key visualizations to the dashboard
        # (Implementation would include smaller versions of the main plots)
        
        plt.tight_layout()
        
        # Save dashboard
        dashboard_path = self.output_dir / f"analysis_dashboard_{timestamp}.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(dashboard_path)
    
    def _save_analysis_report(self, results: Dict) -> Path:
        """Save detailed analysis report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"clustering_analysis_report_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(results)
        
        with open(report_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"📄 Analysis report saved to {report_path}")
        return report_path
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj


def create_clustering_analyzer(**kwargs) -> EmbeddingClusterAnalyzer:
    """Factory function to create EmbeddingClusterAnalyzer"""
    return EmbeddingClusterAnalyzer(**kwargs)


if __name__ == "__main__":
    # Test the clustering analyzer
    print("🧪 Testing Embedding Clustering Analyzer")
    
    # Create analyzer
    analyzer = EmbeddingClusterAnalyzer()
    
    # Generate test data
    np.random.seed(42)
    test_gallery = {
        f"person_{i}": np.random.randn(256) + i * 0.5 
        for i in range(5)
    }
    
    test_track_features = {
        1: [np.random.randn(256) + 0.1],
        2: [np.random.randn(256) + 1.1],
        3: [np.random.randn(256) + 2.1]
    }
    
    # Run analysis
    results = analyzer.analyze_gallery_embeddings(
        gallery=test_gallery,
        track_features=test_track_features,
        save_results=True
    )
    
    print(f"✅ Analysis completed with {len(results)} components")
    if 'visualizations' in results:
        print(f"📊 Visualizations saved: {list(results['visualizations'].keys())}")
    if 'report_path' in results:
        print(f"📄 Report saved to: {results['report_path']}")
