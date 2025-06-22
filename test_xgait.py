#!/usr/bin/env python3
"""
Test script for XGait Person Identification System
Validates model loading and basic functionality
"""

import sys
import os
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_device_detection():
    """Test device detection and configuration"""
    print("🔍 Testing device detection...")
    
    from src.config import get_device, get_device_config
    
    device = get_device()
    config = get_device_config(device)
    
    print(f"   ✅ Detected device: {device}")
    print(f"   ✅ Device config: {config}")
    
    return device

def test_silhouette_model(device):
    """Test silhouette extraction model"""
    print("\n🎭 Testing silhouette model...")
    
    try:
        from src.models.silhouette_model import create_silhouette_extractor
        
        model = create_silhouette_extractor(device=device)
        
        # Test with dummy data
        dummy_crop = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)
        crops = [dummy_crop]
        
        silhouettes = model.extract_silhouettes(crops, target_size=(64, 44))
        
        assert len(silhouettes) == 1
        assert silhouettes[0].shape == (64, 44)
        
        print("   ✅ Silhouette model working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Silhouette model error: {e}")
        return False

def test_parsing_model(device):
    """Test human parsing model"""
    print("\n👤 Testing human parsing model...")
    
    try:
        from src.models.parsing_model import create_human_parsing_model
        
        model = create_human_parsing_model(device=device, model_path="weights/schp_resnet101.pth")
        
        # Test with dummy data
        dummy_crop = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)
        crops = [dummy_crop]
        
        parsing_maps = model.parse_humans(crops, target_size=(64, 44))
        
        assert len(parsing_maps) == 1
        assert parsing_maps[0].shape == (64, 44)
        
        print("   ✅ Human parsing model working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Human parsing model error: {e}")
        return False

def test_xgait_model(device):
    """Test XGait inference model"""
    print("\n🚶 Testing XGait model...")
    
    try:
        from src.models.xgait_model import create_xgait_inference
        
        model = create_xgait_inference(device=device, model_path="weights/Gait3D-XGait-120000.pt")
        
        # Test with dummy data
        dummy_silhouettes = [np.random.randint(0, 255, (64, 44), dtype=np.uint8)]
        dummy_parsing = [np.random.randint(0, 19, (64, 44), dtype=np.uint8)]
        
        features = model.extract_features(dummy_silhouettes, dummy_parsing)
        
        # Test identification
        person_id, confidence = model.identify_person(dummy_silhouettes, dummy_parsing, track_id=1)
        
        assert person_id is not None
        assert 0.0 <= confidence <= 1.0
        
        print(f"   ✅ XGait model working correctly (Person ID: {person_id}, Confidence: {confidence:.3f})")
        return True
        
    except Exception as e:
        print(f"   ❌ XGait model error: {e}")
        return False

def test_identification_processor(device):
    """Test the main identification processor"""
    print("\n🧠 Testing identification processor...")
    
    try:
        from src.models.identification_processor import create_identification_processor
        
        processor = create_identification_processor(
            device=device,
            xgait_model_path="weights/Gait3D-XGait-120000.pt",
            parsing_model_path="weights/schp_resnet101.pth",
            parallel_processing=True
        )
        
        # Test with dummy data
        dummy_crops = [np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8) for _ in range(3)]
        
        person_id, confidence = processor.process_track_crops(
            track_id=1, 
            crops=dummy_crops, 
            frame_number=1
        )
        
        print(f"   ✅ Identification processor working correctly")
        print(f"   📊 Stats: {processor.get_identification_stats()}")
        return True
        
    except Exception as e:
        print(f"   ❌ Identification processor error: {e}")
        return False

def test_file_structure():
    """Test if required files exist"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        "weights/yolo11m.pt",
        "weights/Gait3D-XGait-120000.pt", 
        "weights/schp_resnet101.pth",
        "input/3c.mp4"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ Found: {file_path}")
        else:
            print(f"   ⚠️  Missing: {file_path}")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("🧪 XGait Person Identification System - Test Suite")
    print("=" * 60)
    
    # Test 1: File structure
    files_ok = test_file_structure()
    
    # Test 2: Device detection
    device = test_device_detection()
    
    # Test 3: Individual models (only if files exist)
    tests_passed = 0
    total_tests = 4
    
    if files_ok:
        if test_silhouette_model(device):
            tests_passed += 1
        
        if test_parsing_model(device):
            tests_passed += 1
        
        if test_xgait_model(device):
            tests_passed += 1
        
        if test_identification_processor(device):
            tests_passed += 1
    else:
        print("\n⚠️  Skipping model tests due to missing files")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"   Device: {device}")
    print(f"   Files present: {'✅ Yes' if files_ok else '❌ No'}")
    
    if files_ok:
        print(f"   Model tests passed: {tests_passed}/{total_tests}")
        
        if tests_passed == total_tests:
            print("   🎉 All tests passed! System ready for use.")
            return 0
        else:
            print("   ⚠️  Some tests failed. Check error messages above.")
            return 1
    else:
        print("   📝 Download required model weights to complete testing.")
        return 1

if __name__ == "__main__":
    exit(main())
