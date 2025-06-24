#!/usr/bin/env python3
"""
Test script for the comprehensive embedding clustering system
Demonstrates the clustering analysis capabilities with sample data
"""
import sys
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.embedding_clustering import EmbeddingClusterAnalyzer


def generate_test_embeddings():
    """Generate realistic test embeddings for demonstration"""
    np.random.seed(42)
    
    # Create 3 distinct groups of embeddings (simulating different persons)
    group_centers = [
        np.random.randn(256) * 0.1,      # Person 1 center
        np.random.randn(256) * 0.1 + 2,  # Person 2 center  
        np.random.randn(256) * 0.1 - 1.5 # Person 3 center
    ]
    
    # Gallery embeddings (one per person)
    gallery = {}
    for i, center in enumerate(group_centers):
        person_id = f"person_{i+1}"
        # Add some noise to the center
        gallery[person_id] = center + np.random.randn(256) * 0.05
        
    # Track features (multiple per track, should cluster near gallery persons)
    track_features = {}
    
    # Track 1 - similar to person 1
    track_features[1] = [
        group_centers[0] + np.random.randn(256) * 0.1 + np.random.randn(256) * 0.02
        for _ in range(3)
    ]
    
    # Track 2 - similar to person 2  
    track_features[2] = [
        group_centers[1] + np.random.randn(256) * 0.1 + np.random.randn(256) * 0.02
        for _ in range(4)
    ]
    
    # Track 3 - similar to person 3
    track_features[3] = [
        group_centers[2] + np.random.randn(256) * 0.1 + np.random.randn(256) * 0.02
        for _ in range(2)
    ]
    
    # Track 4 - a new person (should cluster separately)
    new_center = np.random.randn(256) * 0.1 + 3
    track_features[4] = [
        new_center + np.random.randn(256) * 0.02
        for _ in range(3)
    ]
    
    return gallery, track_features


def main():
    """Test the clustering analysis system"""
    print("🧪 Testing Comprehensive Embedding Clustering System")
    print("=" * 60)
    
    # Generate test data
    print("📊 Generating test embeddings...")
    gallery, track_features = generate_test_embeddings()
    
    print(f"   • Gallery persons: {len(gallery)}")
    print(f"   • Track features: {sum(len(features) for features in track_features.values())}")
    print(f"   • Embedding dimension: {len(list(gallery.values())[0])}")
    
    # Create clustering analyzer
    print("\n📊 Initializing clustering analyzer...")
    analyzer = EmbeddingClusterAnalyzer(output_dir="test_clustering_analysis")
    
    # Run comprehensive analysis
    print("\n📊 Running comprehensive clustering analysis...")
    results = analyzer.analyze_gallery_embeddings(
        gallery=gallery,
        track_features=track_features,
        save_results=True
    )
    
    if results:
        print("\n✅ Analysis completed successfully!")
        
        # Print detailed results
        print(f"\n📊 Analysis Summary:")
        print(f"   • Gallery Persons: {results.get('num_gallery_persons', 0)}")
        print(f"   • Track Features: {results.get('num_track_features', 0)}")
        print(f"   • Embedding Dimension: {results.get('embedding_dimension', 'Unknown')}")
        
        # Print quality assessment
        if 'quality' in results and 'quality_assessment' in results['quality']:
            quality = results['quality']['quality_assessment']
            print(f"   • Overall Quality: {quality.get('assessment', 'Unknown')} ({quality.get('overall_score', 0):.2f})")
            if 'quality_factors' in quality:
                print(f"   • Quality Factors: {', '.join(quality['quality_factors'])}")
        
        # Print clustering results
        if 'clustering' in results:
            clustering = results['clustering']
            print(f"\n🎯 Clustering Results:")
            
            if 'kmeans' in clustering and 'optimal_k' in clustering['kmeans']:
                print(f"   • K-means optimal clusters: {clustering['kmeans']['optimal_k']}")
                print(f"   • K-means silhouette score: {clustering['kmeans'].get('best_silhouette_score', 0):.3f}")
            
            if 'dbscan' in clustering and 'n_clusters' in clustering['dbscan']:
                print(f"   • DBSCAN clusters: {clustering['dbscan']['n_clusters']}")
                print(f"   • DBSCAN noise points: {clustering['dbscan'].get('n_noise_points', 0)}")
                print(f"   • DBSCAN silhouette score: {clustering['dbscan'].get('silhouette_score', 0):.3f}")
        
        # Print dimensionality reduction results
        if 'dimensionality_reduction' in results:
            dim_red = results['dimensionality_reduction']
            print(f"\n🔍 Dimensionality Reduction:")
            
            if 'pca' in dim_red and 'total_variance_10_components' in dim_red['pca']:
                print(f"   • PCA: {dim_red['pca']['total_variance_10_components']:.1%} variance in 10 components")
            
            if 'tsne' in dim_red and 'kl_divergence' in dim_red['tsne']:
                print(f"   • t-SNE: KL divergence = {dim_red['tsne']['kl_divergence']:.3f}")
        
        # Print visualization paths
        if 'visualizations' in results:
            print(f"\n🎨 Visualizations saved:")
            for viz_type, path in results['visualizations'].items():
                if path and viz_type != 'error':
                    print(f"   • {viz_type}: {path}")
        
        # Print analysis directory
        if 'report_path' in results:
            analysis_dir = str(Path(results['report_path']).parent)
            print(f"\n📁 Full analysis available in: {analysis_dir}")
        
        print(f"\n🎯 Key Insights:")
        print(f"   • Expected to find 4 clusters (3 gallery + 1 new person)")
        print(f"   • Track features should cluster near corresponding gallery persons")
        print(f"   • Quality should be good with clear separability")
        
    else:
        print("❌ Analysis failed!")
        return 1
    
    print(f"\n✅ Test completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
