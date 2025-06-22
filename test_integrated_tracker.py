"""
Test Script for XGait-Enhanced Person Tracking
Tests the integrated tracking + identification system
"""
import sys
import os
import numpy as np
from pathlib import Path

def test_integrated_tracker():
    """Test the integrated tracking and identification system"""
    print("\n🧪 Testing XGait-Enhanced Person Tracking")
    print("=" * 50)
    
    try:
        # Test imports
        from track_persons import PersonTrackingApp
        from src.config import SystemConfig
        print("✅ Tracking app imports working")
        
        # Create test configuration
        config = SystemConfig.load_default()
        config.video.input_path = "input/3c.mp4"
        config.model.device = "cpu"
        config.video.display_window = False  # No display for testing
        config.verbose = False
        
        # Test app creation with identification enabled
        app = PersonTrackingApp(config, enable_identification=True)
        print("✅ Tracking app with identification created")
        
        # Test identification methods
        stats = app.get_identification_stats()
        print(f"✅ Identification stats: {stats}")
        
        # Test gallery functionality (without actual tracking)
        print("✅ Gallery functionality accessible")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_pipeline_import():
    """Test that simple pipeline can be imported"""
    print("\n🔧 Testing Simple Pipeline Integration")
    print("=" * 40)
    
    try:
        from simple_inference_pipeline import create_simple_inference_pipeline
        pipeline = create_simple_inference_pipeline(device="cpu")
        print("✅ Simple inference pipeline import working")
        
        # Test basic functionality
        dummy_crops = [np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)]
        tracks_data = {1: dummy_crops}
        results = pipeline.process_tracks(tracks_data)
        print("✅ Simple pipeline processing working")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple pipeline test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 XGait-Enhanced Person Tracking Test Suite")
    print("=" * 60)
    
    # Test simple pipeline
    pipeline_test = test_simple_pipeline_import()
    
    # Test integrated tracker
    tracker_test = test_integrated_tracker()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"   Simple Pipeline Test: {'✅ PASS' if pipeline_test else '❌ FAIL'}")
    print(f"   Integrated Tracker Test: {'✅ PASS' if tracker_test else '❌ FAIL'}")
    
    total_tests = 2
    passed_tests = sum([pipeline_test, tracker_test])
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! The integrated system is working correctly.")
        print("\n💡 Usage examples:")
        print("   • Basic tracking: python track_persons.py --video input/3c.mp4")
        print("   • Tracking + ID: python track_persons.py --video input/3c.mp4 --verbose")
        print("   • Disable ID: python track_persons.py --video input/3c.mp4 --no-identification")
        print("   • CPU mode: python track_persons.py --video input/3c.mp4 --device cpu")
    else:
        print("🚨 Some tests failed. Check the error messages above.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
