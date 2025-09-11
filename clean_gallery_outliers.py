import os
# Fix OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import sys
import pickle
import shutil
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.faiss_gallery import FAISSPersonGallery

def clean_gallery_outliers(gallery_path="visualization_analysis/faiss_gallery.pkl", 
                          backup=True, min_embeddings_per_person=3):
    """
    Clean outliers from the saved gallery and save the cleaned version
    
    Args:
        gallery_path: Path to the gallery file
        backup: Whether to create a backup of the original
        min_embeddings_per_person: Minimum embeddings needed to detect outliers
    """
    print("=== CLEANING GALLERY OUTLIERS ===\n")
    
    if not os.path.exists(gallery_path):
        print(f"❌ Gallery file not found: {gallery_path}")
        return
    
    # Create backup if requested
    if backup:
        backup_path = gallery_path.replace('.pkl', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
        shutil.copy2(gallery_path, backup_path)
        print(f"✅ Backup created: {backup_path}")
    
    # Load the gallery
    with open(gallery_path, 'rb') as f:
        gallery_data = pickle.load(f)
    
    embeddings_list = gallery_data["embeddings"]
    print(f"📁 Original gallery: {len(embeddings_list)} embeddings")
    
    # Recreate gallery 
    gallery = FAISSPersonGallery()
    
    # Add all embeddings first
    for emb_obj in embeddings_list:
        gallery.add_person_embedding(
            emb_obj.person_name, 
            emb_obj.track_id, 
            emb_obj.embedding, 
            emb_obj.quality, 
            emb_obj.frame_number
        )
    
    print(f"✅ Recreated gallery with {len(gallery.person_embeddings)} embeddings")
    
    # Analyze and remove outliers for each person
    person_names = set()
    for emb in gallery.person_embeddings:
        if emb is not None:
            person_names.add(emb.person_name)
    
    outliers_removed = 0
    print(f"\n🔍 ANALYZING OUTLIERS FOR {len(person_names)} PERSONS:")
    
    for person_name in sorted(person_names):
        person_embeddings = [
            emb for emb in gallery.person_embeddings 
            if emb is not None and emb.person_name == person_name
        ]
        
        print(f"\n👤 {person_name}:")
        print(f"   Initial embeddings: {len(person_embeddings)}")
        
        if len(person_embeddings) >= min_embeddings_per_person:
            # Calculate initial cluster density
            initial_density = gallery._calculate_cluster_density(person_name)
            print(f"   Initial cluster density: {initial_density:.3f}")
            
            # Find and remove outliers iteratively
            removed_for_person = 0
            max_iterations = 3  # Prevent infinite loops
            
            for iteration in range(max_iterations):
                outlier_idx = gallery._find_cluster_outlier(person_name)
                
                if outlier_idx is not None and outlier_idx < len(gallery.person_embeddings):
                    outlier_emb = gallery.person_embeddings[outlier_idx]
                    if outlier_emb is not None and outlier_emb.person_name == person_name:
                        # Calculate density improvement
                        current_embeddings = [
                            emb for emb in gallery.person_embeddings 
                            if emb is not None and emb.person_name == person_name
                        ]
                        
                        if len(current_embeddings) > min_embeddings_per_person:
                            # Remove the outlier
                            gallery._mark_for_removal(outlier_idx)
                            removed_for_person += 1
                            outliers_removed += 1
                            print(f"   🗑️  Removed outlier {iteration+1}: track {outlier_emb.track_id}, quality {outlier_emb.quality:.3f}")
                        else:
                            print(f"   ⚠️  Stopping removal - would go below minimum embeddings")
                            break
                    else:
                        break
                else:
                    print(f"   ✅ No more outliers detected")
                    break
            
            # Calculate final density
            if removed_for_person > 0:
                final_density = gallery._calculate_cluster_density(person_name)
                improvement = final_density - initial_density
                print(f"   Final cluster density: {final_density:.3f} (improvement: +{improvement:.3f})")
                print(f"   Removed {removed_for_person} outliers")
            
            final_count = len([
                emb for emb in gallery.person_embeddings 
                if emb is not None and emb.person_name == person_name
            ])
            print(f"   Final embeddings: {final_count}")
        else:
            print(f"   ⏭️  Skipping (less than {min_embeddings_per_person} embeddings)")
    
    print(f"\n📊 CLEANING SUMMARY:")
    print(f"   Total outliers removed: {outliers_removed}")
    
    final_embedding_count = len([emb for emb in gallery.person_embeddings if emb is not None])
    print(f"   Final embedding count: {final_embedding_count}")
    print(f"   Embeddings reduced by: {len(embeddings_list) - final_embedding_count}")
    
    # Save the cleaned gallery
    gallery.save_gallery(gallery_path.replace('.pkl', '_cleaned.pkl'))
    print(f"\n✅ Cleaned gallery saved to: {gallery_path.replace('.pkl', '_cleaned.pkl')}")
    
    # Optionally replace the original
    replace_original = input("\n🔄 Replace original gallery with cleaned version? (y/N): ").lower().strip()
    if replace_original == 'y':
        gallery.save_gallery(gallery_path)
        print(f"✅ Original gallery updated with cleaned version")
    
    return gallery

def analyze_cleaning_results(original_path, cleaned_path):
    """Compare original vs cleaned gallery"""
    print("\n" + "="*60)
    print("📈 CLEANING RESULTS COMPARISON")
    print("="*60)
    
    # Load both galleries
    with open(original_path, 'rb') as f:
        original_data = pickle.load(f)
    
    with open(cleaned_path, 'rb') as f:
        cleaned_data = pickle.load(f)
    
    original_embeddings = original_data["embeddings"]
    cleaned_embeddings = cleaned_data["embeddings"]
    
    print(f"Original: {len(original_embeddings)} embeddings")
    print(f"Cleaned:  {len(cleaned_embeddings)} embeddings")
    print(f"Removed:  {len(original_embeddings) - len(cleaned_embeddings)} embeddings")
    
    # Analyze by person
    for gallery_type, embeddings_list in [("Original", original_embeddings), ("Cleaned", cleaned_embeddings)]:
        print(f"\n{gallery_type} Gallery:")
        
        # Recreate gallery for analysis
        gallery = FAISSPersonGallery()
        for emb_obj in embeddings_list:
            gallery.add_person_embedding(
                emb_obj.person_name, emb_obj.track_id, 
                emb_obj.embedding, emb_obj.quality, emb_obj.frame_number
            )
        
        person_names = set(emb.person_name for emb in gallery.person_embeddings if emb is not None)
        
        for person_name in sorted(person_names):
            person_count = len([e for e in gallery.person_embeddings 
                              if e is not None and e.person_name == person_name])
            if person_count > 1:
                density = gallery._calculate_cluster_density(person_name)
                qualities = [e.quality for e in gallery.person_embeddings 
                           if e is not None and e.person_name == person_name]
                print(f"  {person_name}: {person_count} embeddings, density: {density:.3f}, avg quality: {np.mean(qualities):.3f}")

if __name__ == "__main__":
    # Clean the gallery
    gallery = clean_gallery_outliers()
    
    # Analyze results if cleaning was performed
    original_path = "visualization_analysis/faiss_gallery.pkl"
    cleaned_path = "visualization_analysis/faiss_gallery_cleaned.pkl"
    
    if os.path.exists(cleaned_path):
        analyze_cleaning_results(original_path, cleaned_path)
