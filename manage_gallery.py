#!/usr/bin/env python3
"""
Gallery Management CLI Tool
Command-line interface for managing person identification galleries
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.gallery.gallery_manager import GalleryManager


def list_persons(gallery: GalleryManager):
    """List all persons in the gallery"""
    summary = gallery.get_gallery_summary()
    
    if summary['persons'] == 0:
        print("📭 Gallery is empty")
        return
    
    print(f"👥 Gallery contains {summary['persons']} persons:")
    print("-" * 60)
    
    for person_id, details in summary['person_details'].items():
        print(f"🆔 {person_id}")
        print(f"   Features: {details['feature_count']}")
        print(f"   Quality: {details['quality_score']:.3f}")
        print(f"   Tracks: {details['track_ids']}")
        print(f"   Created: {details['created']}")
        print()


def add_person(gallery: GalleryManager, person_id: str, features_file: str):
    """Add a person to the gallery from features file"""
    features_path = Path(features_file)
    if not features_path.exists():
        print(f"❌ Features file not found: {features_path}")
        return False
    
    try:
        # Load features (assume numpy array or text file)
        if features_path.suffix == '.npy':
            features = np.load(features_path)
        elif features_path.suffix == '.txt':
            features = np.loadtxt(features_path)
        else:
            print(f"❌ Unsupported file format: {features_path.suffix}")
            return False
        
        # Add to gallery
        assigned_id = gallery.add_person(person_id, features)
        gallery.save_gallery()
        
        print(f"✅ Added person '{assigned_id}' to gallery")
        print(f"   Features shape: {features.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Error adding person: {e}")
        return False


def remove_person(gallery: GalleryManager, person_id: str):
    """Remove a person from the gallery"""
    with gallery.lock:
        if person_id not in gallery.gallery_features:
            print(f"❌ Person '{person_id}' not found in gallery")
            return False
        
        # Remove person
        del gallery.gallery_features[person_id]
        if person_id in gallery.gallery_metadata:
            del gallery.gallery_metadata[person_id]
        if person_id in gallery.gallery_stats:
            del gallery.gallery_stats[person_id]
        
        # Save changes
        gallery.save_gallery()
        print(f"✅ Removed person '{person_id}' from gallery")
        return True


def clear_gallery(gallery: GalleryManager, confirm: bool = False):
    """Clear the entire gallery"""
    if not confirm:
        response = input("⚠️  This will delete ALL gallery data. Are you sure? (yes/no): ")
        if response.lower() != 'yes':
            print("❌ Operation cancelled")
            return False
    
    gallery._initialize_empty_gallery()
    gallery.save_gallery()
    print("✅ Gallery cleared")
    return True


def export_person(gallery: GalleryManager, person_id: str, output_file: str):
    """Export a person's features to file"""
    if person_id not in gallery.gallery_features:
        print(f"❌ Person '{person_id}' not found in gallery")
        return False
    
    features_list = gallery.gallery_features[person_id]
    features_array = np.array(features_list)
    
    output_path = Path(output_file)
    
    try:
        if output_path.suffix == '.npy':
            np.save(output_path, features_array)
        elif output_path.suffix == '.txt':
            np.savetxt(output_path, features_array)
        else:
            print(f"❌ Unsupported output format: {output_path.suffix}")
            return False
        
        print(f"✅ Exported {len(features_list)} features for '{person_id}' to {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error exporting features: {e}")
        return False


def test_identification(gallery: GalleryManager, features_file: str):
    """Test identification with query features"""
    features_path = Path(features_file)
    if not features_path.exists():
        print(f"❌ Features file not found: {features_path}")
        return
    
    try:
        # Load query features
        if features_path.suffix == '.npy':
            query_features = np.load(features_path)
        elif features_path.suffix == '.txt':
            query_features = np.loadtxt(features_path)
        else:
            print(f"❌ Unsupported file format: {features_path.suffix}")
            return
        
        # Flatten if needed
        if len(query_features.shape) > 1:
            query_features = query_features.flatten()
        
        # Test identification
        person_id, confidence, metadata = gallery.identify_person(query_features, auto_add=False)
        
        print(f"🔍 Identification Result:")
        print(f"   Person ID: {person_id or 'Unknown'}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Action: {metadata['action']}")
        
        if 'all_similarities' in metadata:
            print(f"\n📊 Similarity Scores:")
            for pid, scores in metadata['all_similarities'].items():
                print(f"   {pid}: max={scores['max']:.3f}, avg={scores['avg']:.3f}")
        
    except Exception as e:
        print(f"❌ Error during identification test: {e}")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Gallery Management CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  list                    List all persons in gallery
  add ID FILE             Add person with features from file
  remove ID               Remove person from gallery
  clear                   Clear entire gallery (with confirmation)
  export ID FILE          Export person's features to file
  test FILE               Test identification with query features
  stats                   Show gallery statistics
  
Examples:
  %(prog)s --gallery-dir gallery_data list
  %(prog)s --gallery-dir gallery_data add person_001 features.npy
  %(prog)s --gallery-dir gallery_data remove person_001
  %(prog)s --gallery-dir gallery_data export person_001 exported_features.npy
  %(prog)s --gallery-dir gallery_data test query_features.npy
        """
    )
    
    # Global options
    parser.add_argument('--gallery-dir', '-g', default='gallery_data',
                        help='Gallery data directory (default: gallery_data)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    # Commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    subparsers.add_parser('list', help='List all persons in gallery')
    
    # Add command
    add_parser = subparsers.add_parser('add', help='Add person to gallery')
    add_parser.add_argument('person_id', help='Person identifier')
    add_parser.add_argument('features_file', help='Features file (.npy or .txt)')
    
    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Remove person from gallery')
    remove_parser.add_argument('person_id', help='Person identifier')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear entire gallery')
    clear_parser.add_argument('--force', action='store_true', help='Skip confirmation')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export person features')
    export_parser.add_argument('person_id', help='Person identifier')
    export_parser.add_argument('output_file', help='Output file (.npy or .txt)')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test identification')
    test_parser.add_argument('features_file', help='Query features file (.npy or .txt)')
    
    # Stats command
    subparsers.add_parser('stats', help='Show gallery statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        print("🗃️  Gallery Management CLI")
        print("=" * 40)
        
        # Initialize gallery manager
        gallery = GalleryManager(gallery_dir=args.gallery_dir)
        
        if args.command == 'list':
            list_persons(gallery)
            
        elif args.command == 'add':
            add_person(gallery, args.person_id, args.features_file)
            
        elif args.command == 'remove':
            remove_person(gallery, args.person_id)
            
        elif args.command == 'clear':
            clear_gallery(gallery, confirm=args.force)
            
        elif args.command == 'export':
            export_person(gallery, args.person_id, args.output_file)
            
        elif args.command == 'test':
            test_identification(gallery, args.features_file)
            
        elif args.command == 'stats':
            summary = gallery.get_gallery_summary()
            print(f"📊 Gallery Statistics:")
            print(f"   • Total Persons: {summary['persons']}")
            print(f"   • Total Features: {summary['total_features']}")
            print(f"   • Avg Features per Person: {summary['avg_features_per_person']:.1f}")
            
            if summary['identification_stats']['total_identifications'] > 0:
                id_stats = summary['identification_stats']
                print(f"   • Identification Rate: {id_stats['identification_rate']:.1%}")
                print(f"   • Avg Confidence: {id_stats['avg_confidence']:.3f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
