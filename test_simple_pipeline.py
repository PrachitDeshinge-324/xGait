"""
Test Script for Simple XGait Inference Pipeline
Tests the working simplified version
"""
import sys
import os
import numpy as np
from pathlib import Path

def test_simple_pipeline():
    """Test the simple inference pipeline"""
    print("\n🧪 Testing Simple Inference Pipeline")
    print("=" * 50)
    
    try:
        from simple_inference_pipeline import create_simple_inference_pipeline, demo_simple_inference
        print("✅ Simple pipeline imports working")
        
        # Test pipeline creation
        pipeline = create_simple_inference_pipeline(device="cpu")
        print("✅ Simple pipeline creation working")
        
        # Test with dummy data
        dummy_crops = [
            np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8),
            np.random.randint(0, 255, (120, 60, 3), dtype=np.uint8)
        ]
        
        tracks_data = {1: dummy_crops}
        results = pipeline.process_tracks(tracks_data)
        
        if results and 1 in results:
            print("✅ Simple pipeline processing working")
            return True
        else:
            print("❌ Simple pipeline processing failed")
            return False
            
    except Exception as e:
        print(f"❌ Simple pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_demo():
    """Test the demo function"""
    print("\n🎯 Testing Demo Function")
    print("=" * 30)
    
    try:
        from simple_inference_pipeline import demo_simple_inference
        demo_simple_inference()
        print("✅ Demo function working")
        return True
    except Exception as e:
        print(f"❌ Demo function failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Simple XGait Inference Pipeline Test Suite")
    print("=" * 60)
    
    # Test simple pipeline
    simple_test = test_simple_pipeline()
    
    # Test demo
    demo_test = test_demo()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"   Simple Pipeline Test: {'✅ PASS' if simple_test else '❌ FAIL'}")
    print(f"   Demo Function Test: {'✅ PASS' if demo_test else '❌ FAIL'}")
    
    total_tests = 2
    passed_tests = sum([simple_test, demo_test])
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! The simple pipeline is working correctly.")
        print("\n💡 You can now use the simple pipeline:")
        print("   • python simple_inference_pipeline.py")
        print("   • from simple_inference_pipeline import create_simple_inference_pipeline")
    else:
        print("🚨 Some tests failed. Check the error messages above.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
