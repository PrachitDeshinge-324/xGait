#!/usr/bin/env python3
"""
Person Tracking Application with XGait Feature Extraction
A comprehensive person tracking system with GaitParsing capabilities

Features:
- Real-time person detection and tracking using YOLO
- GaitParsing pipeline (silhouette extraction, human parsing, XGait features)
- Parallel processing for optimal performance
- Debug visualization and analysis tools

Usage:
    python main.py --input video.mp4                    # Basic tracking
    python main.py --input video.mp4 --enable-gait      # With GaitParsing
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import SystemConfig
from track_persons import PersonTrackingApp


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Person Tracking Application with XGait Feature Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input video.mp4                           # Basic tracking only
  %(prog)s --input video.mp4 --enable-gait            # With GaitParsing
  %(prog)s --input video.mp4 --no-display             # Headless mode
        """
    )
    
    # Input/Output
    parser.add_argument('--input', '-i', required=True,
                      help='Input video file path')
    parser.add_argument('--output', '-o', 
                      help='Output directory for results (optional)')
    parser.add_argument('--output-video', 
                      help='Output path for annotated video (e.g., output.mp4)')
    parser.add_argument('--save-video', action='store_true',
                      help='Save annotated video output')
    
    # Features
    parser.add_argument('--enable-gait', action='store_true',
                      help='Enable GaitParsing pipeline (silhouette, parsing, XGait)')
    
    # Display and Debug
    parser.add_argument('--no-display', action='store_true',
                      help='Run without display window (headless mode)')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode with detailed logging')
    parser.add_argument('--save-debug', action='store_true',
                      help='Save debug visualizations to disk')
    
    # Performance
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'],
                      default='auto', help='Processing device')
    parser.add_argument('--max-frames', type=int,
                      help='Maximum number of frames to process (for testing)')
    
    # Model overrides
    parser.add_argument('--yolo-model', help='Override YOLO model path')
    parser.add_argument('--reid-model', help='Override ReID model path')
    parser.add_argument('--gait-model', help='Override XGait model path')
    
    return parser.parse_args()


def configure_system(args):
    """Configure system based on arguments"""
    config = SystemConfig()
    
    # Update paths
    config.video.input_path = args.input
    config.video.display_window = not args.no_display
    config.video.save_annotated_video = args.save_video or bool(args.output_video)
    
    if args.max_frames is not None:
        config.video.max_frames = args.max_frames
    if args.output_video:
        config.video.output_video_path = args.output_video
    elif args.save_video:
        # Auto-generate output name
        input_path = Path(args.input)
        output_name = f"{input_path.stem}_annotated{input_path.suffix}"
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(exist_ok=True)
            config.video.output_video_path = str(output_dir / output_name)
        else:
            config.video.output_video_path = str(input_path.parent / output_name)
    
    # Feature flags
    enable_gait = args.enable_gait
    
    # Debug configuration
    config.verbose = args.debug
    config.debug_mode = args.save_debug
    
    # Device configuration
    if args.device != 'auto':
        config.model.device = args.device
    
    # Model path overrides
    if args.yolo_model:
        config.model.yolo_model_path = args.yolo_model
    if args.reid_model:
        config.model.reid_model_path = args.reid_model
    if args.gait_model:
        config.model.xgait_model_path = args.gait_model
        
    return config, enable_gait


def main():
    """Main application entry point"""
    try:
        # Parse arguments and configure system
        args = parse_arguments()
        config, enable_gait = configure_system(args)
        
        # Validate input file
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"❌ Error: Input file not found: {input_path}")
            return 1
        
        # Print configuration
        print("🚀 Person Tracking Application with XGait Feature Extraction")
        print("=" * 60)
        print(f"📹 Input: {input_path}")
        print(f"🔧 Device: {config.model.device}")
        print(f"👁️  Display: {'Enabled' if config.video.display_window else 'Disabled (headless)'}")
        print(f"🚶 GaitParsing: {'Enabled' if enable_gait else 'Disabled'}")
        print(f"🐛 Debug mode: {'Enabled' if config.debug_mode else 'Disabled'}")
        if config.video.max_frames:
            print(f"🎬 Max frames: {config.video.max_frames} (testing mode)")
        print("=" * 60)
        
        # Initialize and run application
        app = PersonTrackingApp(
            config=config,
            enable_identification=False,
            enable_gait_parsing=enable_gait
        )
        
        # Process video
        app.process_video()
        
        # Print final statistics
        print("\n" + "=" * 60)
        print("📊 FINAL RESULTS")
        print("=" * 60)
        
        if enable_gait:
            gait_stats = app.get_gait_parsing_stats()
            if gait_stats:
                print(f"🎯 GaitParsing Results:")
                print(f"   • Tracks processed: {gait_stats['tracks_processed']}")
                print(f"   • Total parsing results: {gait_stats['total_parsing_results']}")
                print(f"   • Average processing time: {gait_stats['avg_processing_time']:.3f}s")
                if config.debug_mode:
                    print(f"   • Debug images saved: {gait_stats['debug_images_saved']}")
        
        print("\n✅ Application completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Ensure cleanup
        try:
            if 'app' in locals():
                app.cleanup()
        except:
            pass


if __name__ == "__main__":
    sys.exit(main())
