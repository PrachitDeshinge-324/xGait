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
        
        # Tracking data
        self.track_embedding_buffer = defaultdict(list)
        self.track_quality_buffer = defaultdict(list)
        self.track_crop_buffer = defaultdict(list)
        self.track_bbox_buffer = defaultdict(list)
        self.track_parsing_buffer = defaultdict(list)  # Store parsing masks
        self.track_to_person = {}
        self.track_identities = {}
        self.gallery_loaded = False
        
        # Create visualization output directory
        self.visualization_output_dir = Path("visualization_analysis")
        self.visualization_output_dir.mkdir(exist_ok=True)
        
        logger.info(f"✅ FAISS Identity Manager initialized")
    
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
        """Save gallery state to files"""
        # Save FAISS gallery
        faiss_gallery_path = self.visualization_output_dir / "faiss_gallery.pkl"
        print(f"[FAISSGallery] Saving gallery to {faiss_gallery_path}")
        self.faiss_gallery.save_gallery(faiss_gallery_path)
            
        # Save track data for manual merging
        self.save_track_data()
    
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
        """Update context data (crop and bbox) for a track"""
        self.track_crop_buffer[track_id].append(crop)
        self.track_bbox_buffer[track_id].append(bbox)
        
        # Keep only recent data
        max_buffer_size = 10
        if len(self.track_crop_buffer[track_id]) > max_buffer_size:
            self.track_crop_buffer[track_id].pop(0)
            self.track_bbox_buffer[track_id].pop(0)
    
    def update_track_parsing(self, track_id: int, parsing_mask: np.ndarray) -> None:
        """Update parsing mask data for a track"""
        self.track_parsing_buffer[track_id].append(parsing_mask)
        
        # Keep only recent data
        max_buffer_size = 10
        if len(self.track_parsing_buffer[track_id]) > max_buffer_size:
            self.track_parsing_buffer[track_id].pop(0)
    
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
    
    def get_track_embeddings_by_track(self):
        """Get track embeddings organized by track for visualization"""
        embeddings_by_track = {}
        
        for track_id, embeddings in self.track_embedding_buffer.items():
            if embeddings:
                # Get assigned identity
                identity = self.track_to_person.get(track_id, "Unassigned")
                embeddings_by_track[track_id] = [(emb, identity) for emb in embeddings]
        
        return embeddings_by_track
    
    def print_final_summary(self) -> None:
        """Print final identification summary"""
        print("\n" + "="*60)
        print("FINAL IDENTIFICATION SUMMARY")
        print("="*60)
        
        # FAISS gallery summary
        self.faiss_gallery.print_gallery_report()
        
        print("="*60)
    
    def save_track_data(self) -> None:
        """Save track embedding and context data to files for manual merging"""
        track_data_path = self.visualization_output_dir / "track_data.json"
        
        # Convert numpy arrays to lists for JSON serialization
        track_data = {
            'track_embeddings': {},
            'track_qualities': dict(self.track_quality_buffer),
            'track_to_person': dict(self.track_to_person),
            'track_identities': dict(self.track_identities),
            'metadata': {
                'saved_at': str(Path.cwd()),
                'total_tracks': len(self.track_embedding_buffer),
                'assigned_tracks': len(self.track_to_person)
            }
        }
        
        # Convert embeddings to lists
        for track_id, embeddings in self.track_embedding_buffer.items():
            track_data['track_embeddings'][str(track_id)] = [emb.tolist() for emb in embeddings]
        
        # Also save crops and bboxes separately (binary format)
        crops_path = self.visualization_output_dir / "track_crops.pkl"
        context_data = {
            'track_crops': dict(self.track_crop_buffer),
            'track_bboxes': dict(self.track_bbox_buffer),
            'track_parsing_masks': dict(self.track_parsing_buffer)
        }
        
        try:
            with open(track_data_path, 'w') as f:
                json.dump(track_data, f, indent=2)
            
            with open(crops_path, 'wb') as f:
                pickle.dump(context_data, f)
            
            print(f"[TrackData] Saved track data to {track_data_path}")
            print(f"[TrackData] Saved context data to {crops_path}")
            print(f"[TrackData] Total tracks: {len(self.track_embedding_buffer)}, Assigned: {len(self.track_to_person)}")
        except Exception as e:
            logger.error(f"Failed to save track data: {e}")
    
    def load_track_data(self, load_track_assignments: bool = False) -> bool:
        """
        Load track data from files
        
        Args:
            load_track_assignments: If True, loads track-to-person assignments from previous videos.
                                  If False, only loads track embeddings/crops for analysis.
        """
        track_data_path = self.visualization_output_dir / "track_data.json"
        crops_path = self.visualization_output_dir / "track_crops.pkl"
        
        if not track_data_path.exists():
            return False
        
        try:
            # Load main track data
            with open(track_data_path, 'r') as f:
                track_data = json.load(f)
            
            # Restore embeddings (convert back to numpy arrays)
            self.track_embedding_buffer = defaultdict(list)
            for track_id, embeddings in track_data['track_embeddings'].items():
                self.track_embedding_buffer[int(track_id)] = [np.array(emb) for emb in embeddings]
            
            # Restore other data
            self.track_quality_buffer = defaultdict(list)
            for track_id, qualities in track_data['track_qualities'].items():
                self.track_quality_buffer[int(track_id)] = qualities
            
            # Only load track assignments if requested (to prevent carry-over between videos)
            if load_track_assignments:
                self.track_to_person = {int(k): v for k, v in track_data['track_to_person'].items()}
                self.track_identities = {int(k): v for k, v in track_data['track_identities'].items()}
                logger.info(f"🔄 Loaded track assignments from previous session")
            else:
                # Clear track assignments to ensure track independence between videos
                self.track_to_person = {}
                self.track_identities = {}
                logger.info(f"🔄 Cleared track assignments for track independence between videos")
            
            # Load context data if available
            if crops_path.exists():
                with open(crops_path, 'rb') as f:
                    context_data = pickle.load(f)
                
                self.track_crop_buffer = defaultdict(list)
                self.track_bbox_buffer = defaultdict(list)
                self.track_parsing_buffer = defaultdict(list)
                
                for track_id, crops in context_data.get('track_crops', {}).items():
                    self.track_crop_buffer[int(track_id)] = crops
                
                for track_id, bboxes in context_data.get('track_bboxes', {}).items():
                    self.track_bbox_buffer[int(track_id)] = bboxes
                
                for track_id, parsing_masks in context_data.get('track_parsing_masks', {}).items():
                    self.track_parsing_buffer[int(track_id)] = parsing_masks
            
            metadata = track_data.get('metadata', {})
            print(f"[TrackData] Loaded track data from {track_data_path}")
            print(f"[TrackData] Total tracks: {metadata.get('total_tracks', len(self.track_embedding_buffer))}")
            print(f"[TrackData] Assigned tracks: {metadata.get('assigned_tracks', len(self.track_to_person))}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load track data: {e}")
            return False
