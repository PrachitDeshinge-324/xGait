#!/usr/bin/env python3
"""
Gallery Analysis Tool
Standalone tool for analyzing saved gallery data and creating visualizations
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.gallery.gallery_manager import GalleryManager


def main():
    """Main function for gallery analysis"""
    parser = argparse.ArgumentParser(
        description="Analyze XGait gallery data and create visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --gallery-dir gallery_data                           # Basic analysis
  %(prog)s --gallery-dir gallery_data --pca --report           # Full analysis with PCA
  %(prog)s --gallery-dir gallery_data --export results.json    # Export data
        """
    )
    
    # Input/Output
    parser.add_argument('--gallery-dir', '-g', default='gallery_data',
                        help='Gallery data directory (default: gallery_data)')
    parser.add_argument('--output-dir', '-o', default='gallery_analysis',
                        help='Output directory for analysis results')
    
    # Analysis options
    parser.add_argument('--pca', action='store_true',
                        help='Generate PCA visualization')
    parser.add_argument('--report', action='store_true',
                        help='Generate detailed report')
    parser.add_argument('--separability', action='store_true',
                        help='Analyze feature separability')
    parser.add_argument('--export', metavar='FILE',
                        help='Export gallery data to JSON file')
    
    # Visualization options
    parser.add_argument('--show-plots', action='store_true',
                        help='Display plots interactively')
    parser.add_argument('--pca-components', type=int, default=2,
                        help='Number of PCA components (default: 2)')
    
    args = parser.parse_args()
    
    # Check if gallery directory exists
    gallery_dir = Path(args.gallery_dir)
    if not gallery_dir.exists():
        print(f"❌ Gallery directory not found: {gallery_dir}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        print("🔍 Gallery Analysis Tool")
        print("=" * 50)
        print(f"📁 Gallery directory: {gallery_dir}")
        print(f"📁 Output directory: {output_dir}")
        print()
        
        # Initialize gallery manager
        print("📥 Loading gallery data...")
        gallery = GalleryManager(
            gallery_dir=str(gallery_dir),
            pca_components=args.pca_components
        )
        
        # Get basic summary
        summary = gallery.get_gallery_summary()
        print(f"✅ Loaded gallery:")
        print(f"   • Persons: {summary['persons']}")
        print(f"   • Total features: {summary['total_features']}")
        print(f"   • Avg features per person: {summary['avg_features_per_person']:.1f}")
        print()
        
        if summary['persons'] == 0:
            print("⚠️  Gallery is empty - no analysis to perform")
            return 0
        
        # Generate report
        if args.report or not any([args.pca, args.separability, args.export]):
            print("📄 Generating detailed report...")
            report_path = output_dir / "gallery_report.txt"
            report = gallery.create_detailed_report(str(report_path))
            print(f"   Report saved to: {report_path}")
            
            # Print summary to console
            lines = report.split('\n')
            for line in lines:
                if any(keyword in line for keyword in ['OVERVIEW', 'Separability Score', 'Overall Quality']):
                    print(f"   {line}")
            print()
        
        # Separability analysis
        if args.separability:
            print("🎯 Analyzing feature separability...")
            separability = gallery.analyze_separability()
            
            if 'error' in separability:
                print(f"   ❌ {separability['error']}")
            else:
                print(f"   • Separability Score: {separability['separability_score']:.3f}")
                print(f"   • Overall Quality: {separability['quality_assessment']['overall']}")
                print(f"   • Intra-person Similarity: {separability['intra_person_similarity']['mean']:.3f}")
                print(f"   • Inter-person Similarity: {separability['inter_person_similarity']['mean']:.3f}")
                
                if separability['quality_assessment']['recommendations']:
                    print("   💡 Recommendations:")
                    for rec in separability['quality_assessment']['recommendations']:
                        print(f"     • {rec}")
                
                # Save separability analysis
                sep_path = output_dir / "separability_analysis.json"
                with open(sep_path, 'w') as f:
                    json.dump(separability, f, indent=2)
                print(f"   📊 Separability analysis saved to: {sep_path}")
            print()
        
        # PCA visualization
        if args.pca:
            print("📊 Generating PCA visualization...")
            pca_path = output_dir / "pca_visualization.png"
            
            saved_path = gallery.visualize_feature_space(
                save_path=str(pca_path),
                show_plot=args.show_plots
            )
            
            if saved_path:
                print(f"   📈 PCA visualization saved to: {saved_path}")
                
                # Print PCA summary
                if gallery.pca_fitted:
                    explained_var = gallery.pca.explained_variance_ratio_
                    total_var = sum(explained_var)
                    print(f"   • Explained variance: {total_var:.1%}")
                    print(f"   • PC1: {explained_var[0]:.1%}, PC2: {explained_var[1]:.1%}")
            else:
                print("   ❌ Failed to generate PCA visualization")
            print()
        
        # Export data
        if args.export:
            print(f"💾 Exporting gallery data...")
            export_data = {
                'summary': summary,
                'person_details': summary['person_details'],
                'identification_stats': summary['identification_stats'],
                'settings': summary['settings']
            }
            
            # Add separability if available
            if args.separability:
                separability = gallery.analyze_separability()
                if 'error' not in separability:
                    export_data['separability'] = separability
            
            export_path = Path(args.export)
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"   📄 Data exported to: {export_path}")
            print()
        
        print("✅ Gallery analysis completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
