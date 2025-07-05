"""
Identity management module for person identification and gallery management.
"""

import sys
import os
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.simple_identity_gallery import SimpleIdentityGallery
from src.utils.embedding_visualization import EmbeddingVisualizer


class IdentityManager:
    """Manages person identification and gallery operations"""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize gallery
        self.simple_gallery = SimpleIdentityGallery(
            similarity_threshold=0.45,
            max_embeddings_per_person=12,
            min_quality_threshold=0.25,
            prototype_update_strategy="weighted_average"
        )
        
        # Tracking data
        self.track_embedding_buffer = defaultdict(list)
        self.track_quality_buffer = defaultdict(list)
        self.track_to_person = {}
        self.track_identities = {}
        self.gallery_loaded = False
        
        # Create visualization output directory
        self.visualization_output_dir = Path("visualization_analysis")
        self.visualization_output_dir.mkdir(exist_ok=True)
    
    def load_gallery(self) -> bool:
        """
        Load gallery state from file if it exists.
        
        Returns:
            True if gallery was loaded, False otherwise
        """
        gallery_path = self.visualization_output_dir / "simple_gallery.json"
        if gallery_path.exists():
            print(f"[SimpleGallery] Loading gallery from {gallery_path}")
            self.simple_gallery.load_gallery(gallery_path, clear_track_associations=True)
            self.gallery_loaded = True
            print(f"[SimpleGallery] Loaded {len(self.simple_gallery.gallery)} known persons")
            print(f"[SimpleGallery] Track associations cleared for video independence")
            return True
        return False
    
    def save_gallery(self) -> None:
        """Save gallery state to file"""
        gallery_path = self.visualization_output_dir / "simple_gallery.json"
        print(f"[SimpleGallery] Saving gallery to {gallery_path}")
        self.simple_gallery.save_gallery(gallery_path)
    
    def assign_or_update_identities(self, frame_track_embeddings: Dict, frame_count: int) -> Dict:
        """
        Assign or update identities for tracks in the current frame.
        
        Args:
            frame_track_embeddings: Dictionary mapping track_id to (embedding, quality)
            frame_count: Current frame number
            
        Returns:
            Dictionary mapping track_id to person_name
        """
        if not frame_track_embeddings:
            return {}
        
        # Use the unified assignment method
        frame_assignments = self.simple_gallery.assign_or_update_identities(
            frame_track_embeddings, frame_count
        )
        
        # Store for visualization
        self.track_identities = {}
        for track_id, person_name in frame_assignments.items():
            self.track_identities[track_id] = {
                'identity': person_name,
                'confidence': 1.0,
                'is_new': person_name not in self.simple_gallery.person_to_track.values(),
                'frame_assigned': frame_count
            }
        
        # Periodic monitoring
        if frame_count % 500 == 0 and frame_count > 0:
            print(f"[SimpleGallery] Track summary at frame {frame_count}")
            self.simple_gallery.debug_gallery_state("Interim Track Summary")
        
        return frame_assignments
    
    def update_track_embeddings(self, track_id: int, embedding, quality: float) -> None:
        """
        Update embedding buffer for a track.
        
        Args:
            track_id: Track identifier
            embedding: Feature embedding
            quality: Quality score for the embedding
        """
        self.track_embedding_buffer[track_id].append(embedding)
        self.track_quality_buffer[track_id].append(quality)
    
    def get_gallery_stats(self) -> Dict:
        """
        Get gallery statistics.
        
        Returns:
            Dictionary containing gallery statistics
        """
        return {'num_identities': len(self.simple_gallery.gallery)}
    
    def get_is_new_identity_dict(self) -> Dict:
        """
        Get dictionary mapping track_id to True (if new identity) or False (if from gallery).
        
        Returns:
            Dictionary mapping track_id to boolean
        """
        is_new_identity = {}
        if hasattr(self, 'track_identities'):
            for track_id, info in self.track_identities.items():
                is_new_identity[track_id] = info.get('is_new', False)
        return is_new_identity
    
    def get_all_embeddings(self):
        """Get all embeddings for visualization"""
        self.simple_gallery.set_track_embedding_buffer(self.track_embedding_buffer)
        return self.simple_gallery.get_all_embeddings()
    
    def get_track_embeddings_by_track(self):
        """Get track embeddings organized by track for visualization"""
        return self.simple_gallery.get_track_embeddings_by_track()
    
    def print_final_summary(self) -> None:
        """Print final identification summary"""
        print(f"\n📊 Person Identification Summary:")
        known_persons = set()
        new_tracks = []
        
        for track_id in self.track_embedding_buffer.keys():
            # Check if track is assigned to a person in the gallery
            assigned_person = self.simple_gallery.track_to_person.get(track_id)
            if assigned_person and assigned_person in self.simple_gallery.gallery:
                known_persons.add(assigned_person)
                print(f"   ✅ Track {track_id} automatically identified as: {assigned_person}")
            else:
                new_tracks.append(track_id)
                print(f"   🆕 Track {track_id} is NEW (will need naming)")
        
        print(f"\n📋 Final Summary:")
        print(f"   • Known persons automatically identified: {len(known_persons)}")
        print(f"   • New persons requiring naming: {len(new_tracks)}")
        
        if len(new_tracks) == 0:
            print(f"   🎉 All persons were automatically identified! No manual naming needed.")
        
        # Print gallery statistics
        gallery_summary = self.simple_gallery.get_gallery_summary()
        print(f"[SimpleGallery] Final Gallery Statistics:")
        print(f"  • Total persons: {gallery_summary['num_persons']}")
        print(f"  • Total embeddings processed: {gallery_summary['total_embeddings_added']}")
        print(f"  • Prototype updates: {gallery_summary['prototype_updates']}")
        print(f"  • Average embeddings per person: {gallery_summary['average_embeddings_per_person']:.1f}")
        print(f"  • Tracks manually merged: 0")  # Manual naming has been removed
