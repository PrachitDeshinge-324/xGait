#!/usr/bin/env python3
"""
Person Tracking and Identification Application
A comprehensive person tracking system with GaitParsing and ReID capabilities

Features:
- Real-time person detection and tracking using YOLO
- GaitParsing pipeline (silhouette extraction, human parsing, XGait features)
- Person re-identification using TransReID
- Parallel processing for optimal performance
- Debug visualization and analysis tools
- Configurable identification gallery management

Usage:
    python main.py --input video.mp4                    # Basic tracking
    python main.py --input video.mp4 --enable-gait      # With GaitParsing
    python main.py --input video.mp4 --enable-id        # With identification
    python main.py --input video.mp4 --enable-all       # Full pipeline
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import SystemConfig
from track_persons import PersonTrackingApp


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Person Tracking and Identification Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input video.mp4                           # Basic tracking only
  %(prog)s --input video.mp4 --enable-gait            # With GaitParsing
  %(prog)s --input video.mp4 --enable-identification  # With person ID
  %(prog)s --input video.mp4 --enable-all             # Full pipeline
  %(prog)s --input video.mp4 --no-display             # Headless mode
        """
    )
    
    # Input/Output
    parser.add_argument('--input', '-i', required=True,
                      help='Input video file path')
    parser.add_argument('--output', '-o', 
                      help='Output directory for results (optional)')
    
    # Features
    parser.add_argument('--enable-gait', action='store_true',
                      help='Enable GaitParsing pipeline (silhouette, parsing, XGait)')
    parser.add_argument('--enable-identification', '--enable-id', action='store_true',
                      help='Enable person re-identification')
    parser.add_argument('--enable-all', action='store_true',
                      help='Enable all features (GaitParsing + Identification)')
    
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
    
    # Model paths (optional overrides)
    parser.add_argument('--yolo-model', 
                      help='Path to YOLO model (default: weights/yolo11m.pt)')
    parser.add_argument('--reid-model',
                      help='Path to ReID model (default: weights/transreid_vitbase.pth)')
    parser.add_argument('--gait-model',
                      help='Path to XGait model (default: weights/Gait3D-XGait-120000.pt)')
    
    return parser.parse_args()


def configure_system(args):
    """Configure system based on command line arguments"""
    config = SystemConfig()
    
    # Video configuration
    config.video.input_path = args.input
    config.video.display_window = not args.no_display
    if args.max_frames:
        config.video.max_frames = args.max_frames
    
    # Feature flags
    if args.enable_all:
        enable_gait = True
        enable_identification = True
    else:
        enable_gait = args.enable_gait
        enable_identification = args.enable_identification
    
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
        
    return config, enable_gait, enable_identification


def main():
    """Main application entry point"""
    try:
        # Parse arguments and configure system
        args = parse_arguments()
        config, enable_gait, enable_identification = configure_system(args)
        
        # Validate input file
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"❌ Error: Input file not found: {input_path}")
            return 1
        
        # Print configuration
        print("🚀 Person Tracking and Identification Application")
        print("=" * 60)
        print(f"📹 Input: {input_path}")
        print(f"🔧 Device: {config.model.device}")
        print(f"👁️  Display: {'Enabled' if config.video.display_window else 'Disabled (headless)'}")
        print(f"🚶 GaitParsing: {'Enabled' if enable_gait else 'Disabled'}")
        print(f"🔍 Identification: {'Enabled' if enable_identification else 'Disabled'}")
        print(f"🐛 Debug mode: {'Enabled' if config.debug_mode else 'Disabled'}")
        if config.video.max_frames:
            print(f"🎬 Max frames: {config.video.max_frames} (testing mode)")
        print("=" * 60)
        
        # Initialize and run application
        app = PersonTrackingApp(
            config=config,
            enable_identification=enable_identification,
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
        
        if enable_identification:
            id_stats = app.get_identification_stats()
            if id_stats:
                print(f"🔍 Identification Results:")
                print(f"   • Gallery persons: {id_stats.get('persons', 0)}")
                print(f"   • Total features: {id_stats.get('total_features', 0)}")
                print(f"   • Avg features per person: {id_stats.get('avg_features_per_person', 0):.1f}")
            
            # Generate comprehensive gallery analysis
            print(f"\n🎯 Gallery Analysis:")
            print("-" * 40)
            try:
                analysis_dir = app.save_gallery_and_analyze()
                if analysis_dir:
                    print(f"   📁 Analysis saved to: {analysis_dir}")
                
                # Run feature separability analysis
                separability = app.analyze_feature_separability()
                if 'error' not in separability:
                    print(f"   🎯 Separability Score: {separability['separability_score']:.3f}")
                    print(f"   📈 Quality Assessment: {separability['quality_assessment']['overall']}")
                
                # Create final PCA visualization with all tracks
                pca_path = app.run_pca_analysis(save_path="final_pca_analysis.png", show_plot=False)
                if pca_path:
                    print(f"   📊 Final PCA visualization: {pca_path}")
                
            except Exception as e:
                print(f"   ⚠️  Gallery analysis failed: {e}")
        
        print("✅ Processing completed successfully!")
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
