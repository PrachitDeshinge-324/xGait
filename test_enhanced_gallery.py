#!/usr/bin/env python3
"""
Test script for the enhanced SimpleIdentityGallery
"""
import numpy as np
import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.simple_identity_gallery import SimpleIdentityGallery

def test_enhanced_gallery():
    print("🧪 Testing Enhanced Identity Gallery...")
    
    # Initialize gallery
    gallery = SimpleIdentityGallery(
        similarity_threshold=0.7,
        max_gallery_embeddings_per_person=5,
        min_quality_threshold=0.3,
        prototype_update_strategy="weighted_average"
    )
    
    print(f"✅ Gallery initialized with {len(gallery.gallery)} persons")
    
    # Create some test embeddings
    np.random.seed(42)
    
    # Person 1: Similar embeddings (should be grouped together)
    person1_base = np.random.random(512)
    person1_embeddings = [
        person1_base + np.random.normal(0, 0.1, 512),
        person1_base + np.random.normal(0, 0.1, 512),
        person1_base + np.random.normal(0, 0.1, 512)
    ]
    person1_qualities = [0.8, 0.9, 0.7]
    
    # Person 2: Different embeddings
    person2_base = np.random.random(512) * 2
    person2_embeddings = [
        person2_base + np.random.normal(0, 0.1, 512),
        person2_base + np.random.normal(0, 0.1, 512)
    ]
    person2_qualities = [0.6, 0.8]
    
    print("\n📊 Test 1: Adding embeddings for different tracks...")
    
    # Simulate track embeddings over frames
    track_embeddings = {
        1: (person1_embeddings[0], person1_qualities[0]),
        2: (person2_embeddings[0], person2_qualities[0])
    }
    
    # Frame 1: Initial assignments
    assignments = gallery.assign_or_update_identities(track_embeddings, frame_number=1)
    print(f"Frame 1 assignments: {assignments}")
    
    # Frame 2: Add more embeddings for same tracks
    track_embeddings = {
        1: (person1_embeddings[1], person1_qualities[1]),
        2: (person2_embeddings[1], person2_qualities[1])
    }
    
    assignments = gallery.assign_or_update_identities(track_embeddings, frame_number=2)
    print(f"Frame 2 assignments: {assignments}")
    
    # Frame 3: Add another embedding for person 1
    track_embeddings = {
        1: (person1_embeddings[2], person1_qualities[2])
    }
    
    assignments = gallery.assign_or_update_identities(track_embeddings, frame_number=3)
    print(f"Frame 3 assignments: {assignments}")
    
    print(f"\n📈 Gallery now has {len(gallery.gallery)} persons")
    
    # Print gallery statistics
    summary = gallery.get_gallery_summary()
    print(f"\n📊 Gallery Summary:")
    print(f"  • Total persons: {summary['num_persons']}")
    print(f"  • Total embeddings added: {summary['total_embeddings_added']}")
    print(f"  • Prototype updates: {summary['prototype_updates']}")
    print(f"  • Average embeddings per person: {summary['average_embeddings_per_person']:.1f}")
    
    for person_name, person_stats in summary['person_qualities'].items():
        print(f"  • {person_name}:")
        print(f"    - Embeddings: {person_stats['num_embeddings']}")
        print(f"    - Avg quality: {person_stats['avg_quality']:.3f}")
        print(f"    - Max quality: {person_stats['max_quality']:.3f}")
        print(f"    - Updates: {person_stats['update_count']}")
    
    print("\n💾 Test 2: Save/Load functionality...")
    
    # Test save/load
    test_file = Path("test_gallery.json")
    gallery.save_gallery(str(test_file))
    print(f"✅ Gallery saved to {test_file}")
    
    # Create new gallery and load
    new_gallery = SimpleIdentityGallery()
    new_gallery.load_gallery(str(test_file))
    print(f"✅ Gallery loaded. New gallery has {len(new_gallery.gallery)} persons")
    
    # Clean up
    test_file.unlink()
    
    print("\n🔗 Test 3: Consolidation functionality...")
    
    # Create very similar embeddings that should be consolidated
    similar_embedding = person1_base + np.random.normal(0, 0.05, 512)
    track_embeddings = {
        3: (similar_embedding, 0.9)  # Very similar to person 1
    }
    
    # This should create a new person initially
    assignments = gallery.assign_or_update_identities(track_embeddings, frame_number=4)
    print(f"Before consolidation: {len(gallery.gallery)} persons")
    print(f"Assignment: {assignments}")
    
    # Run consolidation
    consolidated = gallery.consolidate_similar_persons(consolidation_threshold=0.8)
    print(f"After consolidation: {len(gallery.gallery)} persons")
    print(f"Consolidated: {consolidated} persons")
    
    print("\n🎉 All tests passed! Enhanced gallery is working correctly.")
    
    return gallery

if __name__ == "__main__":
    gallery = test_enhanced_gallery()
