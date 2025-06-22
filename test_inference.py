"""
Test Script for XGait Inference Pipeline
Validates that all components are working correctly
"""
import sys
import os
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        # Import from root directory (inference_pipeline.py)
        import inference_pipeline
        from inference_pipeline import create_inference_pipeline, demo_inference_pipeline
        print("✅ Inference pipeline imports working")
    except Exception as e:
        print(f"❌ Inference pipeline import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        # Import from root directory (video_inference.py)
        import video_inference  
        from video_inference import VideoInferenceRunner
        print("✅ Video inference imports working")
    except Exception as e:
        print(f"❌ Video inference import failed: {e}")
        return False
    
    try:
        from src.models.silhouette_model import create_silhouette_extractor
        from src.models.parsing_model import create_human_parsing_model  
        from src.models.xgait_model import create_xgait_inference
        print("✅ Model imports working")
    except Exception as e:
        print(f"❌ Model imports failed: {e}")
        return False
    
    return True

def test_pipeline_creation():
    """Test pipeline creation without models"""
    print("\n🧪 Testing pipeline creation...")
    
    try:
        import inference_pipeline
        from inference_pipeline import create_inference_pipeline
        
        # This will fail gracefully if models don't exist
        pipeline = create_inference_pipeline(
            device="cpu",
            xgait_model_path="dummy_path.pt",
            parsing_model_path="dummy_path.pth", 
            identification_threshold=0.6,
            parallel_processing=False  # Disable to avoid threading issues
        )
        
        print("✅ Pipeline creation working (models may not be loaded)")
        return True
    except Exception as e:
        print(f"❌ Pipeline creation failed: {e}")
        return False

def test_dummy_inference():
    """Test inference with dummy data"""
    print("\n🎯 Testing dummy inference...")
    
    try:
        import inference_pipeline
        from inference_pipeline import demo_inference_pipeline
        demo_inference_pipeline()
        return True
    except Exception as e:
        print(f"❌ Dummy inference failed: {e}")
        return False

def check_model_files():
    """Check if model files exist"""
    print("\n📁 Checking model files...")
    
    required_files = [
        "weights/Gait3D-XGait-120000.pt",
        "weights/schp_resnet101.pth",
        "weights/yolo11m.pt"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            all_exist = False
    
    return all_exist

def test_video_file():
    """Check if test video exists"""
    print("\n🎬 Checking test video...")
    
    test_video = "input/3c.mp4"
    if os.path.exists(test_video):
        print(f"✅ Found test video: {test_video}")
        return True
    else:
        print(f"❌ Missing test video: {test_video}")
        return False

def main():
    """Run all tests"""
    print("🧪 XGait Inference Pipeline Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Model Files Check", check_model_files),
        ("Test Video Check", test_video_file),
        ("Pipeline Creation", test_pipeline_creation),
        ("Dummy Inference", test_dummy_inference),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! The system is ready to use.")
    elif passed >= len(results) - 2:  # Allow missing model files
        print("⚠️  Most tests passed. Download model files to enable full functionality.")
    else:
        print("🚨 Multiple tests failed. Check the error messages above.")
    
    # Usage suggestions
    if passed >= 2:  # At least imports and basic structure work
        print("\n💡 Usage suggestions:")
        print("   • Download required model weights to weights/ directory")
        print("   • Place test video in input/ directory")
        print("   • Run: python run_inference.py --demo")
        print("   • Run: python video_inference.py --video input/3c.mp4")

if __name__ == "__main__":
    main()
