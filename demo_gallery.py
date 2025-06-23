#!/usr/bin/env python3
"""
Gallery Demo Script
Demonstrates the gallery management system with synthetic data
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.gallery.gallery_manager import GalleryManager


def create_synthetic_person_features(person_id: int, num_features: int = 5, feature_dim: int = 256) -> np.ndarray:
    """Create synthetic features for a person with realistic variation"""
    # Base feature vector for this person
    base_feature = np.random.randn(feature_dim) * 0.1 + person_id * 0.5
    
    features = []
    for i in range(num_features):
        # Add some variation around the base feature
        variation = np.random.randn(feature_dim) * 0.05  # Small variation
        feature = base_feature + variation
        # Normalize
        feature = feature / (np.linalg.norm(feature) + 1e-8)
        features.append(feature)
    
    return np.array(features)


def demo_gallery_functionality():
    """Demonstrate gallery functionality"""
    print("🧪 Gallery Management System Demo")
    print("=" * 50)
    
    # Create demo gallery
    gallery = GalleryManager(
        gallery_dir="demo_gallery",
        similarity_threshold=0.7,
        auto_add_threshold=0.5,
        max_features_per_person=10
    )
    
    # Clear any existing data
    gallery._initialize_empty_gallery()
    
    print("📝 Step 1: Adding synthetic persons to gallery")
    print("-" * 40)
    
    # Add 5 persons with different numbers of features
    person_data = {}
    for person_idx in range(5):
        person_id = f"demo_person_{person_idx + 1:02d}"
        num_features = np.random.randint(3, 8)  # Random number of features
        
        features_array = create_synthetic_person_features(person_idx, num_features)
        person_data[person_id] = features_array
        
        # Add features one by one (simulating tracking over time)
        for i, feature in enumerate(features_array):
            track_id = person_idx * 100 + i
            assigned_id = gallery.add_person(person_id, feature, track_id=track_id)
            
        print(f"✅ Added {person_id}: {num_features} features")
    
    print(f"\n📊 Gallery Summary:")
    summary = gallery.get_gallery_summary()
    print(f"   • Persons: {summary['persons']}")
    print(f"   • Total features: {summary['total_features']}")
    print(f"   • Avg features per person: {summary['avg_features_per_person']:.1f}")
    
    print("\n🔍 Step 2: Testing identification")
    print("-" * 40)
    
    # Test identification with known features
    for person_id, features_array in person_data.items():
        # Use a slightly modified version of the first feature
        query_feature = features_array[0] + np.random.randn(256) * 0.02
        query_feature = query_feature / (np.linalg.norm(query_feature) + 1e-8)
        
        identified_person, confidence, metadata = gallery.identify_person(
            query_feature, track_id=999, auto_add=False
        )
        
        status = "✅" if identified_person == person_id else "❌"
        print(f"{status} Query from {person_id}: identified as '{identified_person}' (confidence: {confidence:.3f})")
    
    # Test with unknown person
    print("\n🆕 Testing with unknown person:")
    unknown_features = create_synthetic_person_features(10, 1)[0]  # Very different features
    identified_person, confidence, metadata = gallery.identify_person(
        unknown_features, track_id=888, auto_add=True
    )
    print(f"   Unknown person identified as: '{identified_person}' (confidence: {confidence:.3f})")
    print(f"   Action taken: {metadata['action']}")
    
    print("\n🎯 Step 3: Feature separability analysis")
    print("-" * 40)
    
    separability = gallery.analyze_separability()
    if 'error' not in separability:
        print(f"   Separability Score: {separability['separability_score']:.3f}")
        print(f"   Overall Quality: {separability['quality_assessment']['overall']}")
        print(f"   Intra-person Similarity: {separability['intra_person_similarity']['mean']:.3f}")
        print(f"   Inter-person Similarity: {separability['inter_person_similarity']['mean']:.3f}")
        
        if separability['quality_assessment']['recommendations']:
            print(f"   Recommendations:")
            for rec in separability['quality_assessment']['recommendations']:
                print(f"     • {rec}")
    
    print("\n📊 Step 4: PCA visualization")
    print("-" * 40)
    
    # Create PCA visualization
    pca_path = gallery.visualize_feature_space(
        save_path="demo_pca_visualization.png",
        show_plot=False
    )
    
    if pca_path:
        print(f"   📈 PCA visualization saved to: {pca_path}")
        
        if gallery.pca_fitted:
            explained_var = gallery.pca.explained_variance_ratio_
            total_var = sum(explained_var)
            print(f"   📐 Explained variance: {total_var:.1%}")
            print(f"   📊 PC1: {explained_var[0]:.1%}, PC2: {explained_var[1]:.1%}")
    
    print("\n📄 Step 5: Generate detailed report")
    print("-" * 40)
    
    report = gallery.create_detailed_report("demo_gallery_report.txt")
    print(f"   📋 Report saved to: demo_gallery_report.txt")
    
    # Print key sections of the report
    print("\n   Key findings:")
    lines = report.split('\n')
    for line in lines:
        if any(keyword in line for keyword in ['Separability Score:', 'Overall Quality:', 'Total Persons:']):
            print(f"     {line.strip()}")
    
    print("\n💾 Step 6: Save and reload gallery")
    print("-" * 40)
    
    # Save gallery
    gallery.save_gallery()
    print("   ✅ Gallery saved to persistent storage")
    
    # Create new gallery manager and load data
    gallery2 = GalleryManager(gallery_dir="demo_gallery")
    summary2 = gallery2.get_gallery_summary()
    print(f"   📥 Reloaded gallery: {summary2['persons']} persons, {summary2['total_features']} features")
    
    print("\n🧹 Cleanup")
    print("-" * 40)
    gallery.cleanup()
    print("   ✅ Demo completed")
    
    print(f"\n📁 Generated files:")
    print(f"   • demo_pca_visualization.png - PCA visualization")
    print(f"   • demo_gallery_report.txt - Detailed analysis report")
    print(f"   • demo_gallery/ - Persistent gallery data")
    
    return True


if __name__ == "__main__":
    try:
        success = demo_gallery_functionality()
        if success:
            print("\n🎉 Demo completed successfully!")
            print("\nNext steps:")
            print("   • Run: python manage_gallery.py --gallery-dir demo_gallery list")
            print("   • Run: python analyze_gallery.py --gallery-dir demo_gallery --pca --report")
        else:
            print("\n❌ Demo failed")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
