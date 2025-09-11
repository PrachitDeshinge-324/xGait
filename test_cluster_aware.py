#!/usr/bin/env python3
"""
Test the enhanced cluster-aware FAISS gallery system
"""

import sys
import os
import numpy as np
from pathlib import Path

import os
# Fix OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.faiss_gallery import FAISSPersonGallery

def test_cluster_awareness():
    """Test the cluster-aware features"""
    
    print("=== TESTING CLUSTER-AWARE FAISS GALLERY ===")
    
    # Create gallery with smaller limit to test removal logic
    gallery = FAISSPersonGallery(embedding_dim=16384, max_embeddings_per_person=3)
    
    # Create test embeddings with varying similarity
    # Person 1: tight cluster
    person1_center = np.random.randn(16384)
    person1_center = person1_center / np.linalg.norm(person1_center)
    
    person1_embeddings = []
    for i in range(5):  # Add 5 embeddings (will test removal)
        # Add small noise to create a tight cluster
        noise = np.random.randn(16384) * 0.1
        embedding = person1_center + noise
        embedding = embedding / np.linalg.norm(embedding)
        person1_embeddings.append(embedding)
    
    # Person 2: loose cluster with outlier
    person2_center = np.random.randn(16384)
    person2_center = person2_center / np.linalg.norm(person2_center)
    
    person2_embeddings = []
    for i in range(4):
        if i == 3:  # Make last one an outlier
            noise = np.random.randn(16384) * 0.8  # Large noise
        else:
            noise = np.random.randn(16384) * 0.2  # Small noise
        embedding = person2_center + noise
        embedding = embedding / np.linalg.norm(embedding)
        person2_embeddings.append(embedding)
    
    # Add embeddings to gallery
    print("\n=== Adding Person 1 (tight cluster) ===")
    for i, emb in enumerate(person1_embeddings):
        quality = 0.8 + i * 0.05  # Increasing quality
        success = gallery.add_person_embedding("Prachit", 1, emb, quality, i)
        print(f"Added embedding {i+1}/5: {'✅' if success else '❌'}")
    
    print("\n=== Adding Person 2 (loose cluster with outlier) ===")
    for i, emb in enumerate(person2_embeddings):
        quality = 0.7 + i * 0.05
        success = gallery.add_person_embedding("Nayan", 2, emb, quality, i)
        print(f"Added embedding {i+1}/4: {'✅' if success else '❌'}")
    
    # Print cluster analysis
    print("\n" + "="*50)
    gallery.print_cluster_report()
    
    # Test cluster consistency scoring
    print("\n=== Testing Cluster Consistency Scoring ===")
    
    # Test with embedding similar to Person 1
    test_embedding_similar = person1_center + np.random.randn(16384) * 0.05
    test_embedding_similar = test_embedding_similar / np.linalg.norm(test_embedding_similar)
    
    consistency_p1 = gallery._calculate_cluster_consistency_score("Prachit", test_embedding_similar)
    consistency_p2 = gallery._calculate_cluster_consistency_score("Nayan", test_embedding_similar)
    
    print(f"Test embedding (similar to Prachit):")
    print(f"  Consistency with Prachit: {consistency_p1:.3f}")
    print(f"  Consistency with Nayan: {consistency_p2:.3f}")
    
    # Test identification with cluster-aware scoring
    print("\n=== Testing Cluster-Aware Identification ===")
    
    # Query similar to Person 1
    query1 = person1_center + np.random.randn(16384) * 0.1
    query1 = query1 / np.linalg.norm(query1)
    
    person, confidence = gallery.identify_person(query1, track_id=10, frame_number=50)
    print(f"Query similar to Prachit: Identified as '{person}' (confidence: {confidence:.3f})")
    
    # Query similar to Person 2 but not the outlier
    query2 = person2_center + np.random.randn(16384) * 0.15
    query2 = query2 / np.linalg.norm(query2)
    
    person, confidence = gallery.identify_person(query2, track_id=11, frame_number=51)
    print(f"Query similar to Nayan: Identified as '{person}' (confidence: {confidence:.3f})")
    
    # Final gallery report
    print("\n" + "="*50)
    gallery.print_gallery_report()
    
    return gallery

def test_with_real_data():
    """Test with real data from the visualization analysis"""
    
    print("\n=== TESTING WITH REAL DATA ===")
    
    gallery_path = "visualization_analysis/faiss_gallery.pkl"
    if not Path(gallery_path).exists():
        print("❌ Real gallery data not found")
        return
    
    # Load real gallery
    gallery = FAISSPersonGallery()
    success = gallery.load_gallery(gallery_path)
    
    if not success:
        print("❌ Failed to load real gallery")
        return
    
    print("✅ Loaded real gallery data")
    
    # Print cluster analysis for real data
    gallery.print_cluster_report()

if __name__ == "__main__":
    os.chdir('/Users/prachitdeshinge/IIIT/Project/Person Identification/xgait_yolo')
    
    # Test with synthetic data
    test_gallery = test_cluster_awareness()
    
    # Test with real data
    test_with_real_data()
    
    print("\n=== CLUSTER-AWARE TESTING COMPLETE ===")
