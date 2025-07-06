"""
Enhanced Identity Manager with Movement and Orientation Profiling
Integrates with existing system while providing enhanced functionality
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

from src.utils.enhanced_person_gallery import EnhancedPersonGallery, MovementType, OrientationType
from src.utils.simple_identity_gallery import SimpleIdentityGallery

logger = logging.getLogger(__name__)

class EnhancedIdentityManager:
    """
    Enhanced identity manager that combines movement/orientation profiling
    with existing SimpleIdentityGallery for backward compatibility
    """
    
    def __init__(self, config, use_enhanced_gallery: bool = True):
        self.config = config
        self.use_enhanced_gallery = use_enhanced_gallery
        
        # Initialize both gallery systems
        self.simple_gallery = SimpleIdentityGallery(
            similarity_threshold=0.45,
            max_embeddings_per_person=12,
            min_quality_threshold=0.25,
            prototype_update_strategy="weighted_average"
        )
        
        if use_enhanced_gallery:
            # Get XGait sequence length from config
            from src.config import xgaitConfig
            xgait_sequence_length = xgaitConfig.min_sequence_length
            
            self.enhanced_gallery = EnhancedPersonGallery(
                max_embeddings_per_context=5,
                similarity_threshold=0.6,
                min_confidence=0.3,
                history_length=xgait_sequence_length
            )
        else:
            self.enhanced_gallery = None
        
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
        
        logger.info(f"✅ Enhanced Identity Manager initialized (Enhanced: {use_enhanced_gallery})")
    
    def load_gallery(self) -> bool:
        """Load gallery state from files"""
        success = False
        
        # Load simple gallery
        simple_gallery_path = self.visualization_output_dir / "simple_gallery.json"
        if simple_gallery_path.exists():
            print(f"[SimpleGallery] Loading gallery from {simple_gallery_path}")
            self.simple_gallery.load_gallery(simple_gallery_path, clear_track_associations=True)
            print(f"[SimpleGallery] Loaded {len(self.simple_gallery.gallery)} known persons")
            success = True
        
        # Load enhanced gallery if enabled
        if self.use_enhanced_gallery:
            enhanced_gallery_path = self.visualization_output_dir / "enhanced_gallery.json"
            if enhanced_gallery_path.exists():
                print(f"[EnhancedGallery] Loading gallery from {enhanced_gallery_path}")
                self.enhanced_gallery.load_gallery(enhanced_gallery_path)
                print(f"[EnhancedGallery] Loaded {len(self.enhanced_gallery.gallery)} known persons")
                success = True
        
        self.gallery_loaded = success
        return success
    
    def save_gallery(self) -> None:
        """Save gallery state to files"""
        # Save simple gallery
        simple_gallery_path = self.visualization_output_dir / "simple_gallery.json"
        print(f"[SimpleGallery] Saving gallery to {simple_gallery_path}")
        self.simple_gallery.save_gallery(simple_gallery_path)
        
        # Save enhanced gallery if enabled
        if self.use_enhanced_gallery and self.enhanced_gallery:
            enhanced_gallery_path = self.visualization_output_dir / "enhanced_gallery.json"
            print(f"[EnhancedGallery] Saving gallery to {enhanced_gallery_path}")
            self.enhanced_gallery.save_gallery(enhanced_gallery_path)
            
        # Save track data for manual merging
        self.save_track_data()
    
    def assign_or_update_identities(self, frame_track_embeddings: Dict, frame_count: int) -> Dict:
        """
        Assign or update identities for tracks in the current frame
        Uses both simple and enhanced galleries
        """
        if not frame_track_embeddings:
            return {}
        
        # Use simple gallery for primary assignment
        frame_assignments = self.simple_gallery.assign_or_update_identities(
            frame_track_embeddings, frame_count
        )
        
        # If enhanced gallery is enabled, also process with enhanced system
        if self.use_enhanced_gallery and self.enhanced_gallery:
            for track_id, (embedding, quality) in frame_track_embeddings.items():
                # Get additional context data
                if (track_id in self.track_crop_buffer and 
                    track_id in self.track_bbox_buffer and 
                    self.track_crop_buffer[track_id] and 
                    self.track_bbox_buffer[track_id]):
                    
                    latest_crop = self.track_crop_buffer[track_id][-1]
                    latest_bbox = self.track_bbox_buffer[track_id][-1]
                    
                    # Try to identify with enhanced system
                    person_name, confidence, movement_profile = self.enhanced_gallery.identify_person(
                        embedding, track_id, latest_bbox, latest_crop, frame_count
                    )
                    
                    if person_name:
                        # Update enhanced gallery
                        self.enhanced_gallery.add_person_embedding(
                            person_name, track_id, embedding, latest_bbox, 
                            latest_crop, frame_count, quality
                        )
                        
                        # Use enhanced result if different from simple gallery
                        if track_id not in frame_assignments or frame_assignments[track_id] != person_name:
                            logger.info(f"Enhanced gallery override: track {track_id} -> {person_name} "
                                      f"(movement: {movement_profile.movement_type.value}, "
                                      f"orientation: {movement_profile.orientation_type.value})")
                            frame_assignments[track_id] = person_name
                    else:
                        # No match in enhanced gallery - could create new person
                        if track_id not in frame_assignments and confidence > 0.5:
                            new_person = self.enhanced_gallery.create_new_person(
                                track_id, embedding, latest_bbox, latest_crop, frame_count, quality
                            )
                            if new_person:
                                frame_assignments[track_id] = new_person
        
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
            print(f"[EnhancedIdentityManager] Track summary at frame {frame_count}")
            self.simple_gallery.debug_gallery_state("Interim Track Summary")
            if self.use_enhanced_gallery and self.enhanced_gallery:
                self.enhanced_gallery.print_gallery_report()
        
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
        Merge two persons in both gallery systems
        
        Args:
            person1_name: Name of the first person (will be kept)
            person2_name: Name of the second person (will be merged into first)
            
        Returns:
            True if merge was successful, False otherwise
        """
        success = True
        
        # First check if persons exist in simple gallery
        if person1_name in self.simple_gallery.gallery and person2_name in self.simple_gallery.gallery:
            # Get person data
            person1_data = self.simple_gallery.gallery[person1_name]
            person2_data = self.simple_gallery.gallery[person2_name]
            
            # Merge embeddings and data
            person1_data.embeddings.extend(person2_data.embeddings)
            person1_data.qualities.extend(person2_data.qualities)
            person1_data.track_associations.extend(person2_data.track_associations)
            person1_data.update_count += person2_data.update_count
            
            # Trim to max size if needed
            if len(person1_data.embeddings) > self.simple_gallery.max_embeddings_per_person:
                # Keep the best embeddings
                indices = np.argsort(person1_data.qualities)[::-1][:self.simple_gallery.max_embeddings_per_person]
                person1_data.embeddings = [person1_data.embeddings[i] for i in indices]
                person1_data.qualities = [person1_data.qualities[i] for i in indices]
            
            # Recompute prototype
            person1_data.prototype = self.simple_gallery._compute_prototype(
                person1_data.embeddings, person1_data.qualities
            )
            
            # Update track mappings
            for track_id in person2_data.track_associations:
                self.simple_gallery.track_to_person[track_id] = person1_name
                self.track_to_person[track_id] = person1_name
            
            # Remove person2 from simple gallery
            del self.simple_gallery.gallery[person2_name]
            
            logger.info(f"✅ Simple gallery: merged {person2_name} into {person1_name}")
        else:
            logger.warning(f"One or both persons not found in simple gallery: {person1_name}, {person2_name}")
            success = False
        
        # Merge in enhanced gallery if enabled
        if self.use_enhanced_gallery and self.enhanced_gallery:
            enhanced_success = self.enhanced_gallery.merge_persons(person1_name, person2_name)
            if enhanced_success:
                logger.info(f"✅ Enhanced gallery: merged {person2_name} into {person1_name}")
            else:
                logger.warning(f"Enhanced gallery merge failed: {person1_name}, {person2_name}")
            success = success and enhanced_success
        
        return success
    
    def get_gallery_stats(self) -> Dict:
        """Get comprehensive gallery statistics"""
        stats = {
            'simple_gallery': {'num_identities': len(self.simple_gallery.gallery)}
        }
        
        if self.use_enhanced_gallery and self.enhanced_gallery:
            stats['enhanced_gallery'] = self.enhanced_gallery.get_gallery_statistics()
        
        return stats
    
    def get_is_new_identity_dict(self) -> Dict:
        """Get dictionary of new identities"""
        new_identities = {}
        for track_id, identity_data in self.track_identities.items():
            new_identities[track_id] = identity_data.get('is_new', False)
        return new_identities
    
    def get_all_embeddings(self):
        """Get all embeddings for visualization"""
        return self.simple_gallery.get_all_embeddings()
    
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
        
        # Simple gallery summary
        self.simple_gallery.print_comprehensive_report()
        
        # Enhanced gallery summary
        if self.use_enhanced_gallery and self.enhanced_gallery:
            self.enhanced_gallery.print_gallery_report()
        
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
    
    def load_track_data(self) -> bool:
        """Load track data from files"""
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
            
            self.track_to_person = {int(k): v for k, v in track_data['track_to_person'].items()}
            self.track_identities = {int(k): v for k, v in track_data['track_identities'].items()}
            
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
