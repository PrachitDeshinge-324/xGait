"""
FAISS-based Identity Manager for Person Identification
Uses FAISS vector search for efficient person identification
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict
import logging
import json
import pickle

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.faiss_gallery import FAISSPersonGallery

logger = logging.getLogger(__name__)

class IdentityManager:
    """
    Identity manager using FAISS gallery system
    """
    
    def __init__(self, config):
        self.config = config
        
        # Initialize FAISS gallery system with correct XGait dimensions
        self.faiss_gallery = FAISSPersonGallery(
            embedding_dim=16384,  # XGait embedding dimension (256x64 parts)
            similarity_threshold=0.91,
            max_embeddings_per_person=20
        )
        
        # Minimal tracking data (only for current session)
        self.track_embedding_buffer = defaultdict(list)
        self.track_quality_buffer = defaultdict(list)
        self.track_crop_buffer = defaultdict(list)  # Keep minimal for interactive review
        self.track_to_person = {}
        self.track_identities = {}
        self.gallery_loaded = False
        
        # Create visualization output directory
        self.visualization_output_dir = Path("visualization_analysis")
        self.visualization_output_dir.mkdir(exist_ok=True)
        
        logger.info(f"✅ FAISS Identity Manager initialized (optimized storage)")
    
    def load_gallery(self) -> bool:
        """Load gallery state from files"""
        success = False
        
        # Load FAISS gallery
        faiss_gallery_path = self.visualization_output_dir / "faiss_gallery.pkl"
        if faiss_gallery_path.exists():
            print(f"[FAISSGallery] Loading gallery from {faiss_gallery_path}")
            self.faiss_gallery.load_gallery(faiss_gallery_path, clear_track_associations=True)
            stats = self.faiss_gallery.get_gallery_statistics()
            print(f"[FAISSGallery] Loaded {stats['total_persons']} known persons")
            success = True
        
        self.gallery_loaded = success
        return success
    
    def save_gallery(self) -> None:
        """Save gallery state - optimized to only save FAISS gallery"""
        # Save FAISS gallery (contains all essential embeddings and identities)
        faiss_gallery_path = self.visualization_output_dir / "faiss_gallery.pkl"
        print(f"[FAISSGallery] Saving gallery to {faiss_gallery_path}")
        self.faiss_gallery.save_gallery(faiss_gallery_path)
        
        # Save minimal track metadata for compatibility (without heavy data)
        self.save_track_metadata_only()
    
    def assign_or_update_identities(self, frame_track_embeddings: Dict, frame_count: int) -> Dict:
        """
        Assign or update identities for tracks in the current frame
        Uses the FAISS gallery system
        """
        if not frame_track_embeddings:
            return {}
        
        frame_assignments = {}
        
        # Process with FAISS gallery system
        assignment_confidences = {}  # Store actual confidences
        for track_id, (embedding, quality) in frame_track_embeddings.items():
            # Try to identify with FAISS system
            person_name, confidence = self.faiss_gallery.identify_person(embedding, track_id)
            
            if person_name:
                # Update FAISS gallery with new embedding
                self.faiss_gallery.add_person_embedding(
                    person_name, track_id, embedding, quality, frame_count
                )
                
                frame_assignments[track_id] = person_name
                assignment_confidences[track_id] = confidence  # Store actual confidence
                logger.info(f"FAISS gallery assignment: track {track_id} -> {person_name} "
                          f"(confidence: {confidence:.3f})")
            else:
                # No match in FAISS gallery - don't create new person automatically
                # Let the interactive assignment handle this
                if confidence > 0.5:
                    logger.debug(f"Track {track_id} has no match (confidence: {confidence:.3f}) - will be handled in interactive mode")
        
        # Store for visualization and track assignment tracking
        self.track_identities = {}
        for track_id, person_name in frame_assignments.items():
            actual_confidence = assignment_confidences.get(track_id, 0.8)  # Use actual confidence
            self.track_identities[track_id] = {
                'identity': person_name,
                'confidence': actual_confidence,  # Use actual similarity confidence
                'is_new': False,  # FAISS gallery handles new person detection internally
                'frame_assigned': frame_count
            }
            # Store in track_to_person for interactive mode
            self.track_to_person[track_id] = person_name
        
        # Periodic monitoring
        if frame_count % 500 == 0 and frame_count > 0:
            print(f"[IdentityManager] Track summary at frame {frame_count}")
            self.faiss_gallery.print_gallery_report()
        
        return frame_assignments
    
    def update_track_embeddings(self, track_id: int, embedding, quality: float) -> None:
        """Update embedding buffer for a track"""
        self.track_embedding_buffer[track_id].append(embedding)
        self.track_quality_buffer[track_id].append(quality)
        
        # Keep only recent embeddings
        max_buffer_size = 10
        if len(self.track_embedding_buffer[track_id]) > max_buffer_size:
            self.track_embedding_buffer[track_id].pop(0)
            self.track_quality_buffer[track_id].pop(0)
    
    def update_track_context(self, track_id: int, crop: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
        """Update context data (crop only) for interactive review - minimal storage"""
        self.track_crop_buffer[track_id].append(crop)
        
        # Keep only recent data for interactive review (much smaller buffer)
        max_buffer_size = 3  # Only keep 3 most recent crops
        if len(self.track_crop_buffer[track_id]) > max_buffer_size:
            self.track_crop_buffer[track_id].pop(0)
    
    def update_track_parsing(self, track_id: int, parsing_mask: np.ndarray) -> None:
        """Update parsing mask data - deprecated in optimized version"""
        # Skip storing parsing masks to save memory
        pass
    
    def merge_persons(self, person1_name: str, person2_name: str) -> bool:
        """
        Merge two persons in the FAISS gallery system
        
        Args:
            person1_name: Name of the first person (will be kept)
            person2_name: Name of the second person (will be merged into first)
            
        Returns:
            True if merge was successful, False otherwise
        """
        # Merge in FAISS gallery
        faiss_success = self.faiss_gallery.merge_persons(person1_name, person2_name)
        if faiss_success:
            logger.info(f"✅ FAISS gallery: merged {person2_name} into {person1_name}")
        else:
            logger.warning(f"FAISS gallery merge failed: {person1_name}, {person2_name}")
        
        return faiss_success
    
    def get_gallery_stats(self) -> Dict:
        """Get comprehensive gallery statistics"""
        stats = {
            'faiss_gallery': self.faiss_gallery.get_gallery_statistics()
        }
        return stats
    
    def get_is_new_identity_dict(self) -> Dict:
        """Get dictionary of new identities"""
        new_identities = {}
        for track_id, identity_data in self.track_identities.items():
            new_identities[track_id] = identity_data.get('is_new', False)
        return new_identities
    
    def get_all_embeddings(self):
        """Get all embeddings for visualization"""
        # Get embeddings from FAISS gallery
        return self.faiss_gallery.get_all_embeddings()
    
    def get_combined_embeddings(self):
        """
        Get combined embeddings from both gallery and track buffers for comprehensive visualization
        This provides a more complete picture including recent track data that may not be in gallery yet
        """
        # Get gallery embeddings
        gallery_embeddings = self.faiss_gallery.get_all_embeddings()
        
        # Get track embeddings if available
        if not self.track_embedding_buffer:
            self.load_track_data(load_track_assignments=True)
        
        combined_embeddings = list(gallery_embeddings)
        
        # Add track embeddings that might not be in gallery yet
        for track_id, embeddings in self.track_embedding_buffer.items():
            identity = self.track_to_person.get(track_id, "Unassigned")
            for emb in embeddings:
                combined_embeddings.append((emb, identity, track_id, "track_embedding"))
        
        return combined_embeddings
    
    def print_final_summary(self) -> None:
        """Print final identification summary"""
        print("\n" + "="*60)
        print("FINAL IDENTIFICATION SUMMARY")
        print("="*60)
        
        # FAISS gallery summary
        self.faiss_gallery.print_gallery_report()
        
        print("="*60)
    
    def save_track_metadata_only(self) -> None:
        """Save minimal track metadata without heavy embedding/crop data"""
        track_metadata_path = self.visualization_output_dir / "track_metadata.json"
        
        # Save only essential metadata for compatibility
        metadata = {
            'track_to_person': dict(self.track_to_person),
            'track_identities': dict(self.track_identities),
            'summary': {
                'total_tracks': len(self.track_embedding_buffer),
                'assigned_tracks': len(self.track_to_person),
                'unassigned_tracks': len(self.track_embedding_buffer) - len(self.track_to_person)
            }
        }
        
        try:
            with open(track_metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"[TrackMetadata] Saved minimal metadata to {track_metadata_path}")
            print(f"[TrackMetadata] Total tracks: {metadata['summary']['total_tracks']}, Assigned: {metadata['summary']['assigned_tracks']}")
        except Exception as e:
            logger.error(f"Failed to save track metadata: {e}")

    def save_track_data(self) -> None:
        """Legacy method - now saves only metadata for backward compatibility"""
        print("⚠️ Using legacy save_track_data - saving minimal metadata only")
        self.save_track_metadata_only()
    
    def load_track_data(self, load_track_assignments: bool = False) -> bool:
        """
        Load track data - optimized to get data from FAISS gallery instead of heavy files
        
        Args:
            load_track_assignments: If True, loads track-to-person assignments from metadata.
        """
        # Try to load minimal metadata first
        metadata_path = self.visualization_output_dir / "track_metadata.json"
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                if load_track_assignments:
                    self.track_to_person = {int(k): v for k, v in metadata.get('track_to_person', {}).items()}
                    self.track_identities = {int(k): v for k, v in metadata.get('track_identities', {}).items()}
                    print(f"[TrackMetadata] Loaded assignments from metadata")
                
                return True
            except Exception as e:
                logger.warning(f"Failed to load track metadata: {e}")
        
        # Fallback: try to load from legacy track_data.json if it exists
        track_data_path = self.visualization_output_dir / "track_data.json"
        if track_data_path.exists():
            print("⚠️ Loading from legacy track_data.json - consider running again to optimize storage")
            return self._load_legacy_track_data(track_data_path, load_track_assignments)
        
        # If no track data exists, that's okay - FAISS gallery has the important data
        print("ℹ️ No track metadata found - using FAISS gallery data only")
        return False
    
    def _load_legacy_track_data(self, track_data_path: Path, load_track_assignments: bool) -> bool:
        """Load from legacy track_data.json format"""
        try:
            with open(track_data_path, 'r') as f:
                track_data = json.load(f)
            
            # Only load assignments, not the heavy embedding data
            if load_track_assignments:
                self.track_to_person = {int(k): v for k, v in track_data.get('track_to_person', {}).items()}
                self.track_identities = {int(k): v for k, v in track_data.get('track_identities', {}).items()}
                print(f"[LegacyTrackData] Loaded assignments from legacy file")
            
            return True
        except Exception as e:
            logger.error(f"Failed to load legacy track data: {e}")
            return False
    
    def print_embedding_statistics(self):
        """Print detailed embedding statistics for debugging"""
        print("\n" + "="*60)
        print("📊 EMBEDDING STATISTICS")
        print("="*60)
        
        # FAISS Gallery statistics
        gallery_stats = self.faiss_gallery.get_gallery_statistics()
        print(f"🗃️ FAISS Gallery:")
        print(f"   • Total persons: {gallery_stats['total_persons']}")
        print(f"   • Total embeddings: {gallery_stats['total_embeddings']}")
        print(f"   • Persons: {gallery_stats['persons']}")
        
        # Track buffer statistics
        if not self.track_embedding_buffer:
            self.load_track_data(load_track_assignments=True)
        
        print(f"\n🚶 Track Buffer:")
        print(f"   • Total tracks: {len(self.track_embedding_buffer)}")
        for track_id, embeddings in self.track_embedding_buffer.items():
            identity = self.track_to_person.get(track_id, "Unassigned")
            print(f"   • Track {track_id}: {len(embeddings)} embeddings → {identity}")
        
        # Check embedding dimensions and properties
        all_embeddings = self.faiss_gallery.get_all_embeddings()
        if all_embeddings:
            first_emb = all_embeddings[0][0]
            norms = [np.linalg.norm(emb[0]) for emb in all_embeddings[:5]]
            print(f"\n🔢 Embedding Properties:")
            print(f"   • Dimension: {first_emb.shape[0]}")
            print(f"   • Data type: {first_emb.dtype}")
            print(f"   • Value range: [{first_emb.min():.6f}, {first_emb.max():.6f}]")
            print(f"   • Norms (first 5): {[f'{norm:.3f}' for norm in norms]}")
            print(f"   • Normalized: {'Yes' if all(abs(norm - 1.0) < 0.01 for norm in norms) else 'No'}")
    
    def get_track_embeddings_by_track(self):
        """Get track embeddings organized by track for visualization - optimized to use FAISS gallery"""
        # First check if we have track assignments
        if not self.track_to_person:
            self.load_track_data(load_track_assignments=True)
        
        # If still no track data, try to reconstruct from FAISS gallery
        if not self.track_to_person:
            print("ℹ️ No track assignments found - using FAISS gallery data for visualization")
            return self._get_gallery_embeddings_by_person()
        
        # If we have track assignments but no embeddings in buffer, get from FAISS gallery
        embeddings_by_track = {}
        all_gallery_embeddings = self.faiss_gallery.get_all_embeddings()
        
        # Group gallery embeddings by track
        for embedding, person_name, track_id, emb_type in all_gallery_embeddings:
            if track_id not in embeddings_by_track:
                embeddings_by_track[track_id] = []
            embeddings_by_track[track_id].append((embedding, person_name))
        
        return embeddings_by_track
    
    def _get_gallery_embeddings_by_person(self):
        """Fallback: organize gallery embeddings by person instead of track"""
        embeddings_by_person = {}
        all_gallery_embeddings = self.faiss_gallery.get_all_embeddings()
        
        for embedding, person_name, track_id, emb_type in all_gallery_embeddings:
            if person_name not in embeddings_by_person:
                embeddings_by_person[person_name] = []
            embeddings_by_person[person_name].append((embedding, person_name))
        
        return embeddings_by_person
