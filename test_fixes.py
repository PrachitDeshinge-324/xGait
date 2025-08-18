#!/usr/bin/env python3
"""
Test script to verify the fixed assignment logic and silhouette extraction
"""

import sys
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import SystemConfig
from src.processing.enhanced_identity_manager import IdentityManager


def test_assignment_logic():
    """Test the assignment logic with simulated embeddings"""
    print("🧪 Testing Fixed Assignment Logic")
    print("=" * 60)
    
    # Create config and identity manager
    config = SystemConfig()
    identity_manager = IdentityManager(config)
    
    # Simulate some embeddings (XGait dimension)
    embedding1 = np.random.rand(16384).astype(np.float32)
    embedding2 = np.random.rand(16384).astype(np.float32)  
    embedding3 = np.random.rand(16384).astype(np.float32)
    
    print("📝 Testing assignment with empty gallery (should create new persons)...")
    
    # Test 1: First assignment - should create new person
    frame_embeddings = {1: (embedding1, 0.8)}
    assignments = identity_manager.assign_or_update_identities(frame_embeddings, 10)
    print(f"Frame 10 assignments: {assignments}")
    
    # Test 2: Second track - should create another new person
    frame_embeddings = {1: (embedding1, 0.8), 2: (embedding2, 0.7)}
    assignments = identity_manager.assign_or_update_identities(frame_embeddings, 20)
    print(f"Frame 20 assignments: {assignments}")
    
    # Test 3: Similar embedding - should match existing person
    similar_embedding = embedding1 + np.random.normal(0, 0.01, embedding1.shape)  # Very similar
    frame_embeddings = {3: (similar_embedding, 0.9)}
    assignments = identity_manager.assign_or_update_identities(frame_embeddings, 30)
    print(f"Frame 30 assignments: {assignments}")
    
    # Test 4: New different embedding - should create new person
    frame_embeddings = {4: (embedding3, 0.85)}
    assignments = identity_manager.assign_or_update_identities(frame_embeddings, 40)
    print(f"Frame 40 assignments: {assignments}")
    
    print("\n📊 Gallery Statistics:")
    gallery_stats = identity_manager.faiss_gallery.get_gallery_statistics()
    print(f"Total persons: {gallery_stats['total_persons']}")
    print(f"Total embeddings: {gallery_stats['total_embeddings']}")
    print(f"Persons: {gallery_stats['persons']}")
    
    print("\n🔍 Identification Statistics:")
    id_stats = identity_manager.get_identification_statistics()
    print(f"Total identifications: {id_stats['total_identifications']}")
    print(f"Unique persons identified: {id_stats['unique_persons_identified']}")
    print(f"Person counts: {dict(id_stats['person_identification_counts'])}")
    
    print("\n📋 Conclusion Matrix:")
    identity_manager.print_identification_conclusion_matrix()
    
    return len(gallery_stats['persons']) > 0


def test_silhouette_and_processing():
    """Test silhouette extraction and processing pipeline"""
    print("\n🧪 Testing Silhouette Extraction Pipeline")
    print("=" * 60)
    
    # Create config and components
    config = SystemConfig()
    identity_manager = IdentityManager(config)
    
    try:
        from src.processing.gait_processor import GaitProcessor
        gait_processor = GaitProcessor(config, identity_manager)
        
        print("✅ GaitProcessor initialized successfully")
        
        # Create a dummy crop image
        dummy_crop = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
        
        # Test single track processing
        print("📝 Testing single track processing...")
        result = gait_processor._process_single_track_parsing(
            track_id=1, 
            crop=dummy_crop, 
            frame_count=10
        )
        
        print(f"Processing result success: {result.get('success', False)}")
        print(f"Silhouette shape: {result.get('silhouette', np.array([])).shape}")
        print(f"Parsing mask shape: {result.get('parsing_mask', np.array([])).shape}")
        print(f"Feature vector shape: {result.get('feature_vector', np.array([])).shape}")
        
        if result.get('success'):
            print("✅ Silhouette extraction pipeline working correctly")
            return True
        else:
            print("❌ Silhouette extraction pipeline failed")
            return False
        
    except Exception as e:
        print(f"❌ Failed to test silhouette pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Testing Fixed Assignment Logic and Silhouette Extraction")
    print("=" * 80)
    
    # Test assignment logic
    assignment_success = test_assignment_logic()
    
    # Test silhouette processing
    silhouette_success = test_silhouette_and_processing()
    
    print("\n" + "=" * 80)
    print("📊 OVERALL TEST RESULTS")
    print("=" * 80)
    print(f"Assignment logic: {'✅ PASSED' if assignment_success else '❌ FAILED'}")
    print(f"Silhouette pipeline: {'✅ PASSED' if silhouette_success else '❌ FAILED'}")
    
    if assignment_success and silhouette_success:
        print("\n🎉 All tests passed! The issues should be resolved.")
        print("   - New persons will be created automatically")
        print("   - Silhouette extraction should work properly")
        print("   - Identification conclusion matrix will show results")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    print("=" * 80)
