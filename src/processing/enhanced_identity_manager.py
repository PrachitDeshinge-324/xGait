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
        
        # Get interactive mode setting
        self.interactive_mode = getattr(config.video, 'interactive_mode', True)
        print(f"🎮 Identity Manager - Interactive mode: {'Enabled' if self.interactive_mode else 'Disabled'}")
        logger.info(f"🎮 Identity Manager - Interactive mode: {'Enabled' if self.interactive_mode else 'Disabled'}")
        
        # Initialize FAISS gallery system with more reasonable thresholds
        self.faiss_gallery = FAISSPersonGallery(
            embedding_dim=16384,  # XGait embedding dimension (256x64 parts)
            similarity_threshold=0.75,  # More realistic threshold for gait identification
            max_embeddings_per_person=20
        )
        
        # Tracking data with temporal consistency
        self.track_embedding_buffer = defaultdict(list)
        self.track_quality_buffer = defaultdict(list)
        self.track_crop_buffer = defaultdict(list)
        self.track_bbox_buffer = defaultdict(list)
        self.track_parsing_buffer = defaultdict(list)  # Store parsing masks
        self.track_to_person = {}
        self.track_identities = {}
        self.gallery_loaded = False
        
        # Track temporal information for better identity consistency
        self.track_last_seen = {}  # track_id -> frame_number
        self.track_positions = {}  # track_id -> (x, y) center positions
        self.person_candidate_tracks = defaultdict(list)  # person -> [(track_id, confidence, frame)]
        self.person_last_seen = {}  # person_name -> frame_number when last seen
        
        # Identification statistics for conclusion matrix
        self.identification_history = defaultdict(list)  # track_id -> [(frame, person_name, confidence)]
        self.frame_identifications = defaultdict(dict)   # frame -> {track_id: person_name}
        self.person_identification_counts = defaultdict(int)  # person_name -> total_identifications
        self.track_identification_counts = defaultdict(lambda: defaultdict(int))  # track_id -> {person_name: count}
        
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
            load_success = self.faiss_gallery.load_gallery(str(faiss_gallery_path), clear_track_associations=True)
            if load_success:
                stats = self.faiss_gallery.get_gallery_statistics()
                print(f"[FAISSGallery] Successfully loaded {stats['total_persons']} known persons")
                
                # Initialize person_last_seen for all loaded persons
                for person_name in self.faiss_gallery.name_to_indices.keys():
                    if person_name not in self.person_last_seen:
                        self.person_last_seen[person_name] = 0  # Initialize with frame 0
                
                success = True
            else:
                print(f"[FAISSGallery] Failed to load gallery from {faiss_gallery_path}")
        else:
            print(f"[FAISSGallery] No existing gallery found at {faiss_gallery_path}")
        
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
        Uses the FAISS gallery system with automatic new person creation
        """
        if not frame_track_embeddings:
            return {}
        
        frame_assignments = {}
        
        assignment_confidences = {}  # Store actual confidences
        for track_id, (embedding, quality) in frame_track_embeddings.items():
            
            # CRITICAL: Check if this track already has a confirmed person assignment
            if track_id in self.track_to_person:
                # Use existing assignment for consistency - don't re-identify established tracks
                person_name = self.track_to_person[track_id]
                confidence = 0.95  # High confidence for existing assignments
                logger.debug(f"Track {track_id} maintaining existing assignment: {person_name}")
                
                # Update the gallery with new embedding to strengthen the person's profile
                self.faiss_gallery.add_person_embedding(
                    person_name, track_id, embedding, quality, frame_count
                )
                
                frame_assignments[track_id] = person_name
                assignment_confidences[track_id] = confidence
                continue
            
            # Try to identify with FAISS system ONLY if no existing assignment
            person_name, confidence = self.faiss_gallery.identify_person(
                embedding, track_id, frame_number=frame_count
            )
            
            # VERY STRICT matching for new tracks to prevent false positives
            if person_name and confidence >= 0.85:  # Much higher threshold for new track assignments
                # Strong match found - but double-check it's not a spatial conflict
                # Check if this person is already assigned to another track in recent frames
                recent_frames = range(max(0, frame_count - 10), frame_count)
                person_recently_seen = False
                
                for recent_frame in recent_frames:
                    if recent_frame in self.frame_identifications:
                        for other_track_id, other_person in self.frame_identifications[recent_frame].items():
                            if other_person == person_name and other_track_id != track_id:
                                person_recently_seen = True
                                logger.debug(f"Person {person_name} recently seen in track {other_track_id}, "
                                           f"creating new person for track {track_id} to avoid confusion")
                                break
                        if person_recently_seen:
                            break
                
                if not person_recently_seen:
                    # Safe to assign - update gallery with new embedding
                    self.faiss_gallery.add_person_embedding(
                        person_name, track_id, embedding, quality, frame_count
                    )
                    
                    frame_assignments[track_id] = person_name
                    assignment_confidences[track_id] = confidence
                    
                    # Track identification statistics for conclusion matrix
                    self.identification_history[track_id].append((frame_count, person_name, confidence))
                    self.frame_identifications[frame_count][track_id] = person_name
                    self.person_identification_counts[person_name] += 1
                    self.track_identification_counts[track_id][person_name] += 1
                    logger.info(f"FAISS gallery assignment: track {track_id} -> {person_name} "
                              f"(confidence: {confidence:.3f})")
                else:
                    person_name = None  # Force new person creation below
            
            # If no strong match or person recently seen, handle based on mode
            if not person_name or confidence < 0.85:
                print(f"DEBUG: Track {track_id}, Interactive mode: {self.interactive_mode}, person_name: {person_name}, confidence: {confidence}")
                if self.interactive_mode:
                    # INTERACTIVE MODE: Don't create automatic persons - leave unidentified
                    print(f"DEBUG: INTERACTIVE MODE - Skipping track {track_id}")
                    logger.info(f"Track {track_id}: No strong gallery match (conf: {confidence:.3f}) - leaving UNIDENTIFIED for interactive assignment")
                    # Skip this track - don't add to frame_assignments, will show as "UNIDENTIFIED" in visualization
                    continue
                else:
                    # NON-INTERACTIVE MODE: Create new person automatically (old behavior)
                    print(f"DEBUG: NON-INTERACTIVE MODE - Creating person for track {track_id}")
                    person_name = self.faiss_gallery.create_new_person(
                        track_id=track_id,
                        embedding=embedding,
                        quality=quality,
                        frame_number=frame_count
                    )
                    confidence = 0.8  # Default confidence for new person
                    logger.info(f"Created new person: {person_name} for track {track_id} "
                              f"(original confidence: {confidence if 'confidence' in locals() else 'N/A'})")
            
            # Add confirmed assignments to gallery and tracking
            if person_name:  # Only if we have a valid assignment
                # Add to gallery and assignments
                frame_assignments[track_id] = person_name
                assignment_confidences[track_id] = confidence
                
                # Track identification statistics for conclusion matrix
                self.identification_history[track_id].append((frame_count, person_name, confidence))
                self.frame_identifications[frame_count][track_id] = person_name
                self.person_identification_counts[person_name] += 1
                self.track_identification_counts[track_id][person_name] += 1
        
        # *** CRITICAL: Enforce spatial exclusivity - same person cannot be in multiple locations ***
        # Check for duplicate person assignments in the same frame
        person_to_tracks = {}
        for track_id, person_name in frame_assignments.items():
            if person_name not in person_to_tracks:
                person_to_tracks[person_name] = []
            person_to_tracks[person_name].append((track_id, assignment_confidences.get(track_id, 0.8)))
        
        # Count and log spatial conflicts before resolution
        total_conflicts = sum(1 for track_list in person_to_tracks.values() if len(track_list) > 1)
        if total_conflicts > 0:
            logger.warning(f"🚨 FRAME {frame_count}: {total_conflicts} spatial conflicts detected - same person in multiple locations!")
        
        # Resolve conflicts - keep only the highest confidence assignment per person
        resolved_assignments = {}
        for person_name, track_list in person_to_tracks.items():
            if len(track_list) > 1:
                # Multiple tracks assigned to same person - CONFLICT!
                logger.warning(f"🚨 SPATIAL CONFLICT: Person '{person_name}' assigned to {len(track_list)} tracks in frame {frame_count}")
                logger.warning(f"   Conflicted tracks: {[f'T{tid}({conf:.2f})' for tid, conf in track_list]}")
                
                # Keep the track with highest confidence, reassign others
                track_list.sort(key=lambda x: x[1], reverse=True)  # Sort by confidence desc
                best_track_id, best_confidence = track_list[0]
                
                logger.info(f"✅ Keeping track {best_track_id} for {person_name} (confidence: {best_confidence:.3f})")
                resolved_assignments[best_track_id] = person_name
                
                # Reassign conflicting tracks
                for conflicted_track_id, conf in track_list[1:]:
                    logger.warning(f"⚠️ Reassigning conflicted track {conflicted_track_id} (was {person_name}, conf: {conf:.3f})")
                    
                    # Get the original embedding and quality for this track
                    embedding, quality = frame_track_embeddings[conflicted_track_id]
                    
                    # Try to find alternative assignment or create new person
                    alternative_person = self._find_alternative_assignment(conflicted_track_id, person_name, embedding, frame_count)
                    if alternative_person:
                        resolved_assignments[conflicted_track_id] = alternative_person
                        assignment_confidences[conflicted_track_id] = 0.7  # Lower confidence for reassignment
                        logger.info(f"➡️ Track {conflicted_track_id} reassigned to {alternative_person}")
                    else:
                        # Handle new person creation based on interactive mode
                        if self.interactive_mode:
                            # INTERACTIVE MODE: Don't create new persons, leave unidentified
                            logger.info(f"⚠️ Interactive mode: Track {conflicted_track_id} conflict unresolved - leaving UNIDENTIFIED")
                            # Don't add to resolved_assignments, will remain unidentified
                        else:
                            # NON-INTERACTIVE MODE: Create new person for this conflict
                            new_person = self.faiss_gallery.create_new_person(
                                track_id=conflicted_track_id,
                                embedding=embedding,
                                quality=quality,
                                frame_number=frame_count
                            )
                            resolved_assignments[conflicted_track_id] = new_person
                            assignment_confidences[conflicted_track_id] = 0.8  # Default for new person
                            logger.info(f"🆕 Created new person {new_person} for conflicted track {conflicted_track_id}")
            else:
                # No conflict, keep original assignment
                track_id = track_list[0][0]
                resolved_assignments[track_id] = person_name
        
        # Log resolution summary
        if total_conflicts > 0:
            logger.info(f"✅ FRAME {frame_count}: All {total_conflicts} spatial conflicts resolved")
        
        # Update frame_assignments with resolved conflicts
        frame_assignments = resolved_assignments
        
        # Store for visualization and track assignment tracking
        self.track_identities = {}
        for track_id, person_name in frame_assignments.items():
            actual_confidence = assignment_confidences.get(track_id, 0.8)
            self.track_identities[track_id] = {
                'identity': person_name,
                'confidence': actual_confidence,
                'is_new': person_name not in self.person_last_seen,
                'frame_assigned': frame_count
            }
            # Store in track_to_person for consistency
            self.track_to_person[track_id] = person_name
            # Update temporal tracking
            self.track_last_seen[track_id] = frame_count
            self.person_last_seen[person_name] = frame_count
        
        # Periodic monitoring
        if frame_count % 500 == 0 and frame_count > 0:
            print(f"[IdentityManager] Track summary at frame {frame_count}")
            self.faiss_gallery.print_gallery_report()
        
        return frame_assignments
    
    def update_track_embeddings(self, track_id: int, embedding, quality: float) -> None:
        """Update embedding buffer for a track"""
        # Skip None or empty embeddings
        if embedding is None or (hasattr(embedding, 'size') and embedding.size == 0):
            logger.debug(f"Skipping None/empty embedding for track {track_id}")
            return
            
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
    
    def get_gallery_summary(self) -> Dict:
        """Get gallery summary for visualization"""
        faiss_stats = self.faiss_gallery.get_gallery_statistics()
        return {
            'num_identities': faiss_stats['total_persons'],
            'total_tracks': len(self.track_identities),
            'persons': faiss_stats['persons'],
            'total_embeddings': faiss_stats['total_embeddings']
        }
    
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
    
    def get_identification_conclusion_matrix(self) -> Dict:
        """
        Generate conclusion matrix showing how many times each person was identified
        
        Returns:
            Dictionary with identification statistics and matrix
        """
        if not self.identification_history:
            return {
                'total_identifications': 0,
                'unique_persons': 0,
                'tracks_processed': 0,
                'conclusion_matrix': {},
                'person_summary': {},
                'track_summary': {},
                'frames_with_identifications': 0
            }
        
        # Calculate comprehensive statistics
        total_identifications = sum(self.person_identification_counts.values())
        unique_persons = len(self.person_identification_counts)
        tracks_processed = len(self.identification_history)
        frames_with_identifications = len(self.frame_identifications)
        
        # Create person summary (person -> total identifications, tracks, avg confidence)
        person_summary = {}
        for person_name, count in self.person_identification_counts.items():
            # Get all identifications for this person across all tracks
            person_confidences = []
            person_tracks = set()
            
            for track_id, history in self.identification_history.items():
                for frame, identified_person, confidence in history:
                    if identified_person == person_name:
                        person_confidences.append(confidence)
                        person_tracks.add(track_id)
            
            avg_confidence = sum(person_confidences) / len(person_confidences) if person_confidences else 0.0
            
            person_summary[person_name] = {
                'total_identifications': count,
                'unique_tracks': len(person_tracks),
                'tracks': list(person_tracks),
                'average_confidence': avg_confidence,
                'confidence_range': [min(person_confidences), max(person_confidences)] if person_confidences else [0.0, 0.0]
            }
        
        # Create track summary (track -> primary person, identification counts, confidence stats)
        track_summary = {}
        for track_id, identifications in self.track_identification_counts.items():
            if identifications:
                # Find the most frequently identified person for this track
                primary_person = max(identifications.items(), key=lambda x: x[1])
                
                # Get confidence statistics for this track
                track_confidences = [conf for frame, person, conf in self.identification_history[track_id]]
                avg_confidence = sum(track_confidences) / len(track_confidences) if track_confidences else 0.0
                
                track_summary[track_id] = {
                    'primary_person': primary_person[0],
                    'primary_person_count': primary_person[1],
                    'total_identifications': sum(identifications.values()),
                    'identification_breakdown': dict(identifications),
                    'average_confidence': avg_confidence,
                    'confidence_range': [min(track_confidences), max(track_confidences)] if track_confidences else [0.0, 0.0],
                    'consistency_score': primary_person[1] / sum(identifications.values()) if identifications else 0.0
                }
        
        # Create the conclusion matrix (track_id -> person_name -> count)
        conclusion_matrix = {}
        for track_id, person_counts in self.track_identification_counts.items():
            conclusion_matrix[track_id] = dict(person_counts)
        
        return {
            'total_identifications': total_identifications,
            'unique_persons': unique_persons,
            'tracks_processed': tracks_processed,
            'frames_with_identifications': frames_with_identifications,
            'conclusion_matrix': conclusion_matrix,
            'person_summary': person_summary,
            'track_summary': track_summary,
            'identification_timeline': dict(self.frame_identifications)  # frame -> {track: person}
        }

    def print_identification_conclusion_matrix(self) -> None:
        """Print a comprehensive identification conclusion matrix and statistics"""
        matrix_data = self.get_identification_conclusion_matrix()
        
        if matrix_data['total_identifications'] == 0:
            print("\n" + "="*80)
            print("🔍 IDENTIFICATION CONCLUSION MATRIX")
            print("="*80)
            print("❌ No identifications were made during processing.")
            print("="*80)
            return
        
        print("\n" + "="*80)
        print("🔍 IDENTIFICATION CONCLUSION MATRIX")
        print("="*80)
        
        # Overall statistics
        print("📊 OVERALL STATISTICS:")
        print(f"   • Total identifications made: {matrix_data['total_identifications']:,}")
        print(f"   • Unique persons identified: {matrix_data['unique_persons']}")
        print(f"   • Tracks processed: {matrix_data['tracks_processed']}")
        print(f"   • Frames with identifications: {matrix_data['frames_with_identifications']:,}")
        
        # Person-wise summary
        print("\n👥 PERSON IDENTIFICATION SUMMARY:")
        print("-" * 80)
        for person_name, stats in sorted(matrix_data['person_summary'].items(), 
                                        key=lambda x: x[1]['total_identifications'], reverse=True):
            print(f"   🧑 {person_name}:")
            print(f"      Total identifications: {stats['total_identifications']:,}")
            print(f"      Appeared in tracks: {stats['unique_tracks']} tracks {stats['tracks']}")
            print(f"      Average confidence: {stats['average_confidence']:.3f}")
            print(f"      Confidence range: {stats['confidence_range'][0]:.3f} - {stats['confidence_range'][1]:.3f}")
        
        # Track-wise summary
        print("\n🎯 TRACK IDENTIFICATION SUMMARY:")
        print("-" * 80)
        for track_id in sorted(matrix_data['track_summary'].keys()):
            stats = matrix_data['track_summary'][track_id]
            print(f"   Track {track_id}:")
            print(f"      Primary identification: {stats['primary_person']} ({stats['primary_person_count']} times)")
            print(f"      Total identifications: {stats['total_identifications']}")
            print(f"      Consistency score: {stats['consistency_score']:.1%}")
            print(f"      Average confidence: {stats['average_confidence']:.3f}")
            
            # Show identification breakdown if multiple persons identified
            if len(stats['identification_breakdown']) > 1:
                breakdown = ", ".join([f"{person}: {count}" for person, count in 
                                     sorted(stats['identification_breakdown'].items(), 
                                           key=lambda x: x[1], reverse=True)])
                print(f"      Identification breakdown: {breakdown}")
        
        # Detailed matrix table
        print("\n📋 DETAILED CONCLUSION MATRIX:")
        print("-" * 80)
        if matrix_data['conclusion_matrix']:
            # Get all unique persons for header
            all_persons = sorted(set(person for track_counts in matrix_data['conclusion_matrix'].values() 
                                   for person in track_counts.keys()))
            
            # Print header
            header = "Track ID".ljust(10)
            for person in all_persons:
                header += f"{person[:15]:>15}"
            header += "Total".rjust(10) + "Primary ID".rjust(20)
            print(header)
            print("-" * len(header))
            
            # Print each track row
            for track_id in sorted(matrix_data['conclusion_matrix'].keys()):
                row = f"{track_id:>8}  "
                track_counts = matrix_data['conclusion_matrix'][track_id]
                total_count = sum(track_counts.values())
                primary_person = max(track_counts.items(), key=lambda x: x[1])[0] if track_counts else "None"
                
                for person in all_persons:
                    count = track_counts.get(person, 0)
                    if count > 0:
                        row += f"{count:>15}"
                    else:
                        row += f"{'':>15}"
                
                row += f"{total_count:>10}" + f"{primary_person:>20}"
                print(row)
            
            # Print totals row
            totals_row = "TOTALS    "
            for person in all_persons:
                person_total = matrix_data['person_summary'].get(person, {}).get('total_identifications', 0)
                totals_row += f"{person_total:>15}"
            totals_row += f"{matrix_data['total_identifications']:>10}" + f"{'':>20}"
            print("-" * len(header))
            print(totals_row)
        
        print("\n💡 INSIGHTS:")
        
        # Track consistency analysis
        consistent_tracks = sum(1 for stats in matrix_data['track_summary'].values() if stats['consistency_score'] >= 0.8)
        inconsistent_tracks = sum(1 for stats in matrix_data['track_summary'].values() if stats['consistency_score'] < 0.8)
        
        print(f"   • Consistent tracks (≥80% same person): {consistent_tracks}")
        print(f"   • Inconsistent tracks (<80% same person): {inconsistent_tracks}")
        
        # Most/least identified persons
        if matrix_data['person_summary']:
            most_identified = max(matrix_data['person_summary'].items(), key=lambda x: x[1]['total_identifications'])
            least_identified = min(matrix_data['person_summary'].items(), key=lambda x: x[1]['total_identifications'])
            print(f"   • Most identified person: {most_identified[0]} ({most_identified[1]['total_identifications']} times)")
            print(f"   • Least identified person: {least_identified[0]} ({least_identified[1]['total_identifications']} times)")
        
        # Average identifications per frame
        avg_per_frame = matrix_data['total_identifications'] / matrix_data['frames_with_identifications'] if matrix_data['frames_with_identifications'] > 0 else 0
        print(f"   • Average identifications per frame: {avg_per_frame:.1f}")
        
        print("="*80)
    
    def save_track_data(self) -> None:
        """Save track embedding and context data to files for manual merging"""
        track_data_path = self.visualization_output_dir / "track_data.json"
        
        # Convert numpy arrays to lists for JSON serialization
        track_data = {
            'track_embeddings': {},
            'track_qualities': dict(self.track_quality_buffer),
            'track_to_person': dict(self.track_to_person),
            'track_identities': dict(self.track_identities),
            'identification_statistics': {
                'identification_history': {str(k): v for k, v in self.identification_history.items()},
                'person_identification_counts': dict(self.person_identification_counts),
                'track_identification_counts': {str(k): dict(v) for k, v in self.track_identification_counts.items()},
                'frame_identifications': {str(k): v for k, v in self.frame_identifications.items()}
            },
            'metadata': {
                'saved_at': str(Path.cwd()),
                'total_tracks': len(self.track_embedding_buffer),
                'assigned_tracks': len(self.track_to_person),
                'total_identifications': sum(self.person_identification_counts.values()),
                'unique_persons_identified': len(self.person_identification_counts)
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
            print(f"[TrackData] Total identifications: {sum(self.person_identification_counts.values())}")
            print(f"[TrackData] Unique persons identified: {len(self.person_identification_counts)}")
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
                
                # Load identification statistics if available
                if 'identification_statistics' in track_data:
                    id_stats = track_data['identification_statistics']
                    
                    # Restore identification history
                    self.identification_history = defaultdict(list)
                    for track_id, history in id_stats.get('identification_history', {}).items():
                        self.identification_history[int(track_id)] = history
                    
                    # Restore identification counts
                    self.person_identification_counts = defaultdict(int)
                    for person, count in id_stats.get('person_identification_counts', {}).items():
                        self.person_identification_counts[person] = count
                    
                    # Restore track identification counts
                    self.track_identification_counts = defaultdict(lambda: defaultdict(int))
                    for track_id, person_counts in id_stats.get('track_identification_counts', {}).items():
                        for person, count in person_counts.items():
                            self.track_identification_counts[int(track_id)][person] = count
                    
                    # Restore frame identifications
                    self.frame_identifications = defaultdict(dict)
                    for frame, identifications in id_stats.get('frame_identifications', {}).items():
                        self.frame_identifications[int(frame)] = identifications
                
                logger.info(f"🔄 Loaded track assignments from previous session")
            else:
                # Clear track assignments to ensure track independence between videos
                self.track_to_person = {}
                self.track_identities = {}
                
                # Also clear identification statistics for fresh start
                self.identification_history = defaultdict(list)
                self.person_identification_counts = defaultdict(int)
                self.track_identification_counts = defaultdict(lambda: defaultdict(int))
                self.frame_identifications = defaultdict(dict)
                
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
            print(f"[TrackData] Total identifications: {metadata.get('total_identifications', sum(self.person_identification_counts.values()))}")
            print(f"[TrackData] Unique persons: {metadata.get('unique_persons_identified', len(self.person_identification_counts))}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load track data: {e}")
            return False

    def get_identification_statistics(self) -> Dict:
        """Get identification statistics for external use"""
        matrix_data = self.get_identification_conclusion_matrix()
        
        return {
            'total_identifications': matrix_data['total_identifications'],
            'unique_persons_identified': matrix_data['unique_persons'],
            'tracks_with_identifications': matrix_data['tracks_processed'],
            'frames_with_identifications': matrix_data['frames_with_identifications'],
            'person_identification_counts': self.person_identification_counts,
            'consistency_scores': {
                track_id: stats['consistency_score'] 
                for track_id, stats in matrix_data['track_summary'].items()
            },
            'average_confidence_by_person': {
                person: stats['average_confidence']
                for person, stats in matrix_data['person_summary'].items()
            }
        }
    
    def print_final_summary(self) -> None:
        """Print final identification summary with conclusion matrix"""
        print("\n" + "="*80)
        print("📊 FINAL IDENTIFICATION SUMMARY")
        print("="*80)
        
        # FAISS gallery summary
        self.faiss_gallery.print_gallery_report()
        
        # Print the detailed conclusion matrix
        self.print_identification_conclusion_matrix()
        
        print("="*80)
    
    def assign_person_manually(self, track_id: int, person_name: str, frame_number: int) -> bool:
        """
        Manually assign a person name to a track (for interactive mode)
        
        Args:
            track_id: Track ID to assign
            person_name: Person name to assign
            frame_number: Current frame number
            
        Returns:
            True if assignment was successful
        """
        if not self.interactive_mode:
            logger.warning(f"Manual assignment called but interactive mode is disabled")
            return False
            
        try:
            # Get the most recent embedding for this track
            if track_id not in self.track_embedding_buffer or not self.track_embedding_buffer[track_id]:
                logger.error(f"No embeddings available for track {track_id}")
                return False
            
            # Use the most recent embedding and quality
            embedding = self.track_embedding_buffer[track_id][-1]
            quality = self.track_quality_buffer[track_id][-1] if self.track_quality_buffer[track_id] else 0.8
            
            # Check if person already exists in gallery
            if person_name in self.faiss_gallery.name_to_indices:
                # Add embedding to existing person
                success = self.faiss_gallery.add_person_embedding(
                    person_name, track_id, embedding, quality, frame_number
                )
                logger.info(f"✅ Added track {track_id} to existing person {person_name}")
            else:
                # Create new person with this embedding
                success = self.faiss_gallery.add_person_embedding(
                    person_name, track_id, embedding, quality, frame_number
                )
                logger.info(f"🆕 Created new person {person_name} for track {track_id}")
            
            if success:
                # Update tracking
                self.track_to_person[track_id] = person_name
                self.track_identities[track_id] = {
                    'identity': person_name,
                    'confidence': 1.0,  # Manual assignment = perfect confidence
                    'is_new': person_name not in self.person_last_seen,
                    'frame_assigned': frame_number
                }
                
                # Update temporal tracking
                self.track_last_seen[track_id] = frame_number
                self.person_last_seen[person_name] = frame_number
                
                # Update identification history
                self.identification_history[track_id].append((frame_number, person_name, 1.0))
                self.frame_identifications[frame_number][track_id] = person_name
                self.person_identification_counts[person_name] += 1
                self.track_identification_counts[track_id][person_name] += 1
                
                logger.info(f"✅ Manual assignment: track {track_id} → {person_name}")
                return True
            else:
                logger.error(f"Failed to add embedding for manual assignment: track {track_id} → {person_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error in manual assignment: {e}")
            return False
    
    def get_unidentified_tracks(self) -> List[int]:
        """Get list of tracks that haven't been identified yet"""
        all_tracks = set(self.track_embedding_buffer.keys())
        identified_tracks = set(self.track_to_person.keys())
        return list(all_tracks - identified_tracks)
    
    def _find_alternative_assignment(self, track_id: int, excluded_person: str, embedding, frame_count: int) -> str:
        """
        Find alternative person assignment for a conflicted track
        
        Args:
            track_id: Track ID that needs reassignment
            excluded_person: Person name to exclude from consideration
            embedding: Track's embedding for similarity search
            frame_count: Current frame number
            
        Returns:
            Alternative person name or None if no suitable alternative found
        """
        try:
            # Get top matches from FAISS gallery, excluding the conflicted person
            person_name, confidence = self.faiss_gallery.identify_person(
                embedding, track_id, frame_number=frame_count
            )
            
            # If the top match is the excluded person, try to get other candidates
            if person_name == excluded_person:
                # Try with lower thresholds to find alternative matches
                distances, indices = self.faiss_gallery.index.search(
                    embedding.reshape(1, -1).astype(np.float32), 
                    min(10, len(self.faiss_gallery.person_names))  # Get top 10 or available persons
                )
                
                for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx >= 0 and idx < len(self.faiss_gallery.person_names):
                        candidate_person = self.faiss_gallery.person_names[idx]
                        similarity = 1.0 - distance  # Convert distance to similarity
                        
                        # Skip the excluded person and check if similarity is reasonable
                        if (candidate_person != excluded_person and 
                            similarity >= self.faiss_gallery.low_confidence_threshold):
                            logger.debug(f"Alternative assignment found: {candidate_person} "
                                       f"(similarity: {similarity:.3f})")
                            return candidate_person
                
                logger.debug(f"No suitable alternative found for track {track_id}")
                return None
            
            # If top match is different person and confidence is reasonable, use it
            elif confidence >= self.faiss_gallery.low_confidence_threshold:
                logger.debug(f"Alternative assignment: {person_name} (confidence: {confidence:.3f})")
                return person_name
            
            logger.debug(f"Top match confidence too low for track {track_id}: {confidence:.3f}")
            return None
            
        except Exception as e:
            logger.error(f"Error finding alternative assignment for track {track_id}: {e}")
            return None
