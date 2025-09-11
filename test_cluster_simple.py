import os
# Fix OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.faiss_gallery import FAISSPersonGallery

def create_realistic_embeddings(center, num_embeddings=5, cluster_std=0.05):
    """Create embeddings clustered around a center point with more realistic similarity"""
    embeddings = []
    
    # Create tight cluster (most embeddings)
    for i in range(num_embeddings-1):
        # Add small random variation to center
        noise = np.random.normal(0, cluster_std, center.shape)
        embedding = center + noise
        # Normalize to unit vector
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding)
    
    # Add one outlier (further from center but still somewhat similar)
    if num_embeddings > 1:
        outlier_noise = np.random.normal(0, cluster_std * 2, center.shape)
        outlier = center + outlier_noise
        outlier = outlier / np.linalg.norm(outlier)
        embeddings.append(outlier)
    
    return embeddings

def test_cluster_methods():
    """Test cluster analysis methods directly"""
    print("=== TESTING CLUSTER ANALYSIS METHODS ===\n")
    
    # Create gallery
    gallery = FAISSPersonGallery(embedding_dim=512)  # Smaller dim for testing
    
    # Create realistic centers (more separated)
    center1 = np.random.randn(512)
    center1 = center1 / np.linalg.norm(center1)
    
    center2 = np.random.randn(512) 
    center2 = center2 / np.linalg.norm(center2)
    # Ensure centers are different enough
    while np.dot(center1, center2) > 0.5:  # Less than 50% similarity
        center2 = np.random.randn(512)
        center2 = center2 / np.linalg.norm(center2)
    
    print(f"Center similarity: {np.dot(center1, center2):.3f}")
    
    # Add embeddings for person 1
    embeddings1 = create_realistic_embeddings(center1, 4, 0.02)  # Very tight cluster
    for i, emb in enumerate(embeddings1):
        gallery.add_person_embedding("TestPerson1", f"track_{i}", emb, 0.9, i)
    
    # Add embeddings for person 2
    embeddings2 = create_realistic_embeddings(center2, 3, 0.03)  # Slightly looser cluster
    for i, emb in enumerate(embeddings2):
        gallery.add_person_embedding("TestPerson2", f"track_{i+10}", emb, 0.85, i+10)
    
    # Test cluster density calculation
    print("=== CLUSTER DENSITY ANALYSIS ===")
    for person_name in ["TestPerson1", "TestPerson2"]:
        person_embeddings = [
            emb.embedding for emb in gallery.person_embeddings 
            if emb is not None and emb.person_name == person_name
        ]
        
        if len(person_embeddings) > 1:
            density = gallery._calculate_cluster_density(person_name)
            print(f"{person_name}: {len(person_embeddings)} embeddings, density: {density:.3f}")
            
            # Calculate intra-cluster similarities manually for verification
            similarities = []
            for i in range(len(person_embeddings)):
                for j in range(i+1, len(person_embeddings)):
                    sim = np.dot(person_embeddings[i], person_embeddings[j])
                    similarities.append(sim)
            
            print(f"  Manual verification - similarities: {[f'{s:.3f}' for s in similarities]}")
            print(f"  Average similarity: {np.mean(similarities):.3f}")
            print()
    
    # Test outlier detection
    print("=== OUTLIER DETECTION ===")
    for person_name in ["TestPerson1", "TestPerson2"]:
        person_indices = [
            i for i, emb in enumerate(gallery.person_embeddings) 
            if emb is not None and emb.person_name == person_name
        ]
        
        if len(person_indices) > 2:
            outlier_idx = gallery._find_cluster_outlier(person_name)
            if outlier_idx is not None:
                outlier_emb = gallery.person_embeddings[outlier_idx]
                print(f"{person_name}: Outlier detected at index {outlier_idx}")
                print(f"  Track ID: {outlier_emb.track_id}")
                print(f"  Quality: {outlier_emb.quality}")
            else:
                print(f"{person_name}: No outlier detected")
        print()
    
    # Test consistency scoring
    print("=== CONSISTENCY SCORING ===")
    test_embedding = center1 + np.random.normal(0, 0.01, center1.shape)  # Very similar to person1
    test_embedding = test_embedding / np.linalg.norm(test_embedding)
    
    for person_name in ["TestPerson1", "TestPerson2"]:
        score = gallery._calculate_cluster_consistency_score(person_name, test_embedding)
        print(f"Test embedding consistency with {person_name}: {score:.3f}")
    
    print(f"\nExpected: Higher score for TestPerson1")

if __name__ == "__main__":
    test_cluster_methods()
