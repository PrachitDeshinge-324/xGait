"""
Quick Start Script for XGait Person Identification
Simple interface to run person identification on videos
"""
import os
import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        "weights/Gait3D-XGait-120000.pt",
        "weights/schp_resnet101.pth", 
        "weights/yolo11m.pt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required model files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease download the required model weights:")
        print("1. XGait model: Gait3D-XGait-120000.pt")
        print("2. SCHP parsing model: schp_resnet101.pth")
        print("3. YOLO detection model: yolo11m.pt")
        return False
    
    return True

def run_inference_pipeline():
    """Run the inference pipeline with simple interface"""
    print("🚀 XGait Person Identification System")
    print("=====================================")
    
    if not check_requirements():
        return
    
    parser = argparse.ArgumentParser(description="XGait Person Identification")
    parser.add_argument("--video", type=str, default="input/3c.mp4", 
                       help="Input video path")
    parser.add_argument("--device", type=str, default="mps", 
                       choices=["cpu", "cuda", "mps"], help="Device to use")
    parser.add_argument("--demo", action="store_true", 
                       help="Run quick demo with built-in test")
    parser.add_argument("--batch", action="store_true", 
                       help="Process in batch mode (no visualization)")
    
    args = parser.parse_args()
    
    if args.demo:
        print("🎯 Running demo mode...")
        run_demo()
    else:
        print(f"📺 Processing video: {args.video}")
        print(f"🖥️  Using device: {args.device}")
        
        try:
            from video_inference import VideoInferenceRunner
            
            runner = VideoInferenceRunner(
                video_path=args.video,
                device=args.device,
                display_output=not args.batch,
                save_results=args.batch
            )
            
            runner.run_inference()
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("Please ensure all dependencies are installed")
        except Exception as e:
            print(f"❌ Error during processing: {e}")

def run_demo():
    """Run a simple demo to test the system"""
    try:
        from inference_pipeline import demo_inference_pipeline
        demo_inference_pipeline()
    except Exception as e:
        print(f"❌ Demo failed: {e}")

def main():
    """Main entry point"""
    run_inference_pipeline()

if __name__ == "__main__":
    main()
