import os
# Fix OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.faiss_gallery import FAISSPersonGallery
import pickle

def load_and_analyze_real_gallery():
    """Load real gallery data and analyze cluster properties"""
    print("=== ANALYZING REAL GALLERY DATA ===\n")
    
    # Try to load existing gallery
    gallery_path = "visualization_analysis/faiss_gallery.pkl"
    
    if not os.path.exists(gallery_path):
        print(f"❌ Gallery file not found: {gallery_path}")
        return
    
    # Load the gallery
    with open(gallery_path, 'rb') as f:
        gallery_data = pickle.load(f)
    
    # Extract embeddings list
    embeddings_list = gallery_data["embeddings"]
    print(f"✅ Loaded gallery with {len(embeddings_list)} embeddings")
    
    # Recreate gallery and add embeddings
    gallery = FAISSPersonGallery()
    
    for emb_obj in embeddings_list:
        gallery.add_person_embedding(
            emb_obj.person_name, 
            emb_obj.track_id, 
            emb_obj.embedding, 
            emb_obj.quality, 
            emb_obj.frame_number
        )
    
    print(f"✅ Recreated gallery with {len(gallery.person_embeddings)} embeddings")
    
    # Analyze each person's cluster
    print("\n" + "="*60)
    print("🧮 REAL GALLERY CLUSTER ANALYSIS")
    print("="*60)
    
    person_names = set()
    for emb in gallery.person_embeddings:
        if emb is not None:
            person_names.add(emb.person_name)
    
    total_outliers = 0
    
    for person_name in sorted(person_names):
        person_embeddings = [
            emb for emb in gallery.person_embeddings 
            if emb is not None and emb.person_name == person_name
        ]
        
        if len(person_embeddings) == 0:
            continue
            
        print(f"\n👤 {person_name}:")
        print(f"   Embeddings: {len(person_embeddings)}")
        
        if len(person_embeddings) > 1:
            density = gallery._calculate_cluster_density(person_name)
            print(f"   Cluster Density: {density:.3f}")
            
            # Quality analysis
            qualities = [emb.quality for emb in person_embeddings]
            print(f"   Average Quality: {np.mean(qualities):.3f} (±{np.std(qualities):.3f})")
            print(f"   Quality Range: {np.min(qualities):.3f} - {np.max(qualities):.3f}")
            
            # Check for outliers
            outlier_idx = gallery._find_cluster_outlier(person_name)
            if outlier_idx is not None and outlier_idx < len(gallery.person_embeddings):
                outlier_emb = gallery.person_embeddings[outlier_idx]
                if outlier_emb is not None:
                    print(f"   ⚠️  Potential outlier detected (idx: {outlier_idx})")
                    print(f"       Track ID: {outlier_emb.track_id}, Quality: {outlier_emb.quality:.3f}")
                    total_outliers += 1
            
            # Classify cluster quality
            if density > 0.7:
                cluster_quality = "Excellent"
            elif density > 0.5:
                cluster_quality = "Good"
            elif density > 0.3:
                cluster_quality = "Fair"
            else:
                cluster_quality = "Poor"
            print(f"   Cluster Quality: {cluster_quality}")
        else:
            print(f"   Single embedding - no cluster analysis")
    
    print(f"\n📊 OVERALL CLUSTER STATISTICS:")
    print(f"   Total Persons: {len(person_names)}")
    print(f"   Total Embeddings: {len([e for e in gallery.person_embeddings if e is not None])}")
    print(f"   Persons with Outliers: {total_outliers}")
    print("="*60)
    
    # Test cluster-aware identification with a synthetic query
    print(f"\n🔍 TESTING CLUSTER-AWARE IDENTIFICATION:")
    
    # Use first person's first embedding as a base for creating a test query
    first_person = None
    first_embedding = None
    
    for emb in gallery.person_embeddings:
        if emb is not None:
            first_person = emb.person_name
            first_embedding = emb.embedding
            break
    
    if first_person and first_embedding is not None:
        # Create a slightly modified version of the first embedding
        test_embedding = first_embedding + np.random.normal(0, 0.01, first_embedding.shape)
        test_embedding = test_embedding / np.linalg.norm(test_embedding)
        
        print(f"Test query (similar to {first_person}):")
        
        # Test consistency with each person
        consistencies = []
        for person_name in sorted(person_names):
            person_count = len([e for e in gallery.person_embeddings 
                              if e is not None and e.person_name == person_name])
            if person_count > 1:  # Only test persons with multiple embeddings
                consistency = gallery._calculate_cluster_consistency_score(person_name, test_embedding)
                consistencies.append((consistency, person_name))
                print(f"  Consistency with {person_name}: {consistency:.3f}")
        
        # Show best matches
        consistencies.sort(reverse=True)
        print(f"\nBest cluster matches:")
        for i, (score, name) in enumerate(consistencies[:3]):
            print(f"  {i+1}. {name}: {score:.3f}")

if __name__ == "__main__":
    load_and_analyze_real_gallery()
