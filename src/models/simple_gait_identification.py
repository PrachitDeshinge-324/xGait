#!/usr/bin/env python3
"""
Simplified Gait Identification System
A clean, minimal implementation for person identification using XGait embeddings
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
from sklearn.metrics.pairwise import cosine_similarity
import threading

logger = logging.getLogger(__name__)


class SimpleGaitIdentification:
    """
    Simplified gait identification system using XGait embeddings
    
    Features:
    - 30-frame sequence feature extraction
    - Simple gallery as dictionary mapping person names to embeddings
    - Automatic assignment of new person IDs
    - Ensures no duplicate person assignments in single frame
    - Persistent gallery storage
    """
    
    def __init__(self, 
                 gallery_file: str = "gallery_data/simple_gallery.json",
                 similarity_threshold: float = 0.75,  # Raised threshold to better distinguish different people
                 sequence_length: int = 30,
                 verbose: bool = False):
        """
        Initialize Simple Gait Identification System
        
        Args:
            gallery_file: Path to save/load gallery data
            similarity_threshold: Minimum similarity for positive identification
            sequence_length: Number of frames to use for feature extraction
            verbose: Enable verbose debug output
        """
        self.gallery_file = Path(gallery_file)
        self.gallery_file.parent.mkdir(exist_ok=True)
        
        self.similarity_threshold = similarity_threshold
        self.sequence_length = sequence_length
        self.verbose = verbose
        
        # Dynamic Gallery System - person_name -> multiple embeddings with metadata
        self.gallery = {}  # person_id -> {'embeddings': [embeddings], 'centroid': centroid, 'track_sources': [track_ids]}
        
        # Track sequences - track_id -> list of silhouettes
        self.track_sequences = {}
        
        # Track parsing sequences - track_id -> list of parsing masks
        self.track_parsing_sequences = {}
        
        # Track features - track_id -> embedding
        self.track_features = {}
        
        # Track assignments - track_id -> person_id (persistent assignments)
        self.track_assignments = {}
        
        # Person assignments for current frame (to avoid duplicates)
        self.current_frame_assignments = {}
        
        # Track identification confidence - track_id -> confidence
        self.track_confidence = {}
        
        # Gallery evolution parameters
        self.max_embeddings_per_person = 20  # Maximum embeddings to store per person
        self.min_embeddings_for_centroid = 3  # Minimum embeddings needed to compute centroid
        self.gallery_update_threshold = 0.1  # Update centroid if new embedding changes it significantly
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Person counter for new IDs
        self.next_person_id = 1
        
        # Load existing gallery
        self._load_gallery()
        
        logger.info(f"✅ Simple Gait Identification initialized")
        logger.info(f"   Gallery file: {self.gallery_file}")
        logger.info(f"   Similarity threshold: {self.similarity_threshold}")
        logger.info(f"   Sequence length: {self.sequence_length}")
        logger.info(f"   Loaded persons: {len(self.gallery)}")
    
    def _load_gallery(self) -> None:
        """Load gallery data from file"""
        try:
            if self.gallery_file.exists():
                with open(self.gallery_file, 'r') as f:
                    data = json.load(f)
                
                # Handle both old and new gallery formats
                gallery_data = data.get('gallery', {})
                
                for person_id, person_data in gallery_data.items():
                    if isinstance(person_data, list):
                        # Old format: person_id -> embedding
                        self.gallery[person_id] = {
                            'embeddings': [np.array(person_data)],
                            'centroid': np.array(person_data),
                            'track_sources': []
                        }
                    elif isinstance(person_data, dict):
                        # New format: person_id -> {'embeddings': [...], 'centroid': [...], 'track_sources': [...]}
                        embeddings = [np.array(emb) for emb in person_data.get('embeddings', [])]
                        centroid = np.array(person_data.get('centroid', embeddings[0] if embeddings else []))
                        track_sources = person_data.get('track_sources', [])
                        
                        self.gallery[person_id] = {
                            'embeddings': embeddings,
                            'centroid': centroid,
                            'track_sources': track_sources
                        }
                
                # Load track assignments and confidence
                self.track_assignments = data.get('track_assignments', {})
                
                # Convert track confidence keys to back to int
                track_conf_data = data.get('track_confidence', {})
                self.track_confidence = {
                    int(track_id): float(confidence)
                    for track_id, confidence in track_conf_data.items()
                }
                
                self.next_person_id = data.get('next_person_id', 1)
                
                # Log gallery statistics
                total_embeddings = sum(len(person_data['embeddings']) for person_data in self.gallery.values())
                logger.info(f"📁 Loaded dynamic gallery with {len(self.gallery)} persons, {total_embeddings} total embeddings, {len(self.track_assignments)} track assignments")
            else:
                logger.info("📁 No existing gallery found, starting fresh")
                
        except Exception as e:
            logger.error(f"❌ Error loading gallery: {e}")
            self.gallery = {}
            self.next_person_id = 1
    
    def _save_gallery(self) -> None:
        """Save gallery data to file"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            gallery_data = {}
            total_embeddings = 0
            
            for person_id, person_data in self.gallery.items():
                embeddings = [emb.tolist() for emb in person_data['embeddings']]
                centroid = person_data['centroid'].tolist()
                track_sources = person_data['track_sources']
                
                gallery_data[person_id] = {
                    'embeddings': embeddings,
                    'centroid': centroid,
                    'track_sources': track_sources,
                    'num_embeddings': len(embeddings)
                }
                total_embeddings += len(embeddings)
            
            data = {
                'gallery': gallery_data,
                'track_assignments': self.track_assignments.copy(),
                'track_confidence': {
                    str(track_id): float(confidence)
                    for track_id, confidence in self.track_confidence.items()
                },
                'next_person_id': self.next_person_id,
                'timestamp': datetime.now().isoformat(),
                'num_persons': len(self.gallery),
                'total_embeddings': total_embeddings,
                'gallery_type': 'dynamic_evolving'
            }
            
            with open(self.gallery_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"💾 Dynamic gallery saved: {len(self.gallery)} persons, {total_embeddings} embeddings, {len(self.track_assignments)} track assignments")
            
        except Exception as e:
            logger.error(f"❌ Error saving gallery: {e}")
    
    def _compute_centroid(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Compute centroid (mean) of embeddings
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Centroid embedding
        """
        if not embeddings:
            return np.array([])
        
        # Stack embeddings and compute mean
        stacked = np.stack(embeddings, axis=0)
        centroid = np.mean(stacked, axis=0)
        
        # Normalize centroid
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
            
        return centroid
    
    def _add_embedding_to_person(self, person_id: str, new_embedding: np.ndarray, track_id: int) -> bool:
        """
        Add a new embedding to a person's gallery and update centroid
        
        Args:
            person_id: Person identifier
            new_embedding: New embedding to add
            track_id: Source track ID
            
        Returns:
            True if centroid was significantly updated, False otherwise
        """
        if person_id not in self.gallery:
            logger.warning(f"⚠️ Person {person_id} not found in gallery")
            return False
        
        person_data = self.gallery[person_id]
        
        # Add new embedding
        person_data['embeddings'].append(new_embedding.copy())
        
        # Add track source if not already present
        if track_id not in person_data['track_sources']:
            person_data['track_sources'].append(track_id)
        
        # Limit number of embeddings per person
        if len(person_data['embeddings']) > self.max_embeddings_per_person:
            # Remove oldest embedding
            person_data['embeddings'].pop(0)
            if self.verbose:
                logger.info(f"🔄 Removed oldest embedding for {person_id} (limit: {self.max_embeddings_per_person})")
        
        # Compute new centroid
        old_centroid = person_data['centroid'].copy()
        new_centroid = self._compute_centroid(person_data['embeddings'])
        
        # Check if centroid changed significantly
        if len(old_centroid) > 0:
            centroid_similarity = cosine_similarity(
                old_centroid.reshape(1, -1), 
                new_centroid.reshape(1, -1)
            )[0, 0]
            centroid_change = 1.0 - centroid_similarity
            
            significant_change = centroid_change > self.gallery_update_threshold
        else:
            significant_change = True
        
        # Update centroid
        person_data['centroid'] = new_centroid
        
        if self.verbose and significant_change:
            logger.info(f"🔄 Updated centroid for {person_id}: {len(person_data['embeddings'])} embeddings, change: {centroid_change:.3f}")
        
        return significant_change
    
    def _update_person_gallery(self, person_id: str, track_id: int, new_features: np.ndarray) -> None:
        """
        Update person's gallery with new features from a track
        
        Args:
            person_id: Person identifier
            track_id: Track that generated the features
            new_features: New gait features
        """
        with self.lock:
            # Add embedding and check if centroid changed significantly
            centroid_updated = self._add_embedding_to_person(person_id, new_features, track_id)
            
            # Save gallery if centroid was significantly updated
            if centroid_updated:
                self._save_gallery()
                
                if self.verbose:
                    person_data = self.gallery[person_id]
                    logger.info(f"📈 Gallery evolved for {person_id}: {len(person_data['embeddings'])} embeddings from {len(person_data['track_sources'])} tracks")
    
    def _get_person_representative_embedding(self, person_id: str) -> np.ndarray:
        """
        Get the most representative embedding for a person (centroid)
        
        Args:
            person_id: Person identifier
            
        Returns:
            Representative embedding (centroid)
        """
        if person_id not in self.gallery:
            return np.array([])
        
        person_data = self.gallery[person_id]
        
        # Use centroid if we have enough embeddings, otherwise use latest
        if len(person_data['embeddings']) >= self.min_embeddings_for_centroid:
            return person_data['centroid']
        else:
            return person_data['embeddings'][-1]  # Latest embedding
    
    def add_silhouette_to_sequence(self, track_id: int, silhouette: np.ndarray, parsing_mask: np.ndarray = None) -> None:
        """
        Add a silhouette and parsing mask to a track's sequence
        
        Args:
            track_id: Track identifier
            silhouette: Silhouette mask
            parsing_mask: Human parsing mask (optional)
        """
        with self.lock:
            if track_id not in self.track_sequences:
                self.track_sequences[track_id] = []
                self.track_parsing_sequences[track_id] = []
            
            self.track_sequences[track_id].append(silhouette.copy())
    
            # Add parsing mask if provided, otherwise add None
            if parsing_mask is not None:
                self.track_parsing_sequences[track_id].append(parsing_mask.copy())
            else:
                self.track_parsing_sequences[track_id].append(None)
            
            # Keep only recent frames
            if len(self.track_sequences[track_id]) > self.sequence_length:
                self.track_sequences[track_id].pop(0)
                self.track_parsing_sequences[track_id].pop(0)
    
    def extract_gait_features(self, track_id: int, xgait_model) -> Optional[np.ndarray]:
        """
        Extract gait features from a track's sequence using XGait model
        
        Args:
            track_id: Track identifier
            xgait_model: XGait model for feature extraction
            
        Returns:
            Feature embedding or None if insufficient frames
        """
        with self.lock:
            if track_id not in self.track_sequences:
                return None
            
            sequence = self.track_sequences[track_id]
            parsing_sequence = self.track_parsing_sequences.get(track_id, [])
            
            if len(sequence) < self.sequence_length:
                return None
            
            try:
                # Get the last sequence_length frames
                silhouettes = sequence[-self.sequence_length:]
                
                # Get corresponding parsing masks (may contain None values)
                parsing_masks = parsing_sequence[-self.sequence_length:] if len(parsing_sequence) >= self.sequence_length else None
                
                # Clean parsing masks - filter out None values and ensure same length
                if parsing_masks and len(parsing_masks) == len(silhouettes):
                    # Check if we have actual parsing masks (not just None values)
                    has_real_masks = any(mask is not None for mask in parsing_masks)
                    if not has_real_masks:
                        parsing_masks = None
                else:
                    parsing_masks = None
                
                # Use the XGait model to extract features from the sequence
                features = xgait_model.extract_features_from_sequence(
                    silhouettes=silhouettes,
                    parsing_masks=parsing_masks
                )
                
                # Store features for this track
                self.track_features[track_id] = features
                
                return features
                
            except Exception as e:
                logger.error(f"❌ Error extracting features for track {track_id}: {e}")
                return None
    
    def identify_or_assign_person(self, track_id: int, features: np.ndarray) -> Tuple[str, float]:
        """
        Identify or assign a person for a track with persistent assignment and dynamic gallery evolution
        
        Args:
            track_id: Track identifier
            features: XGait features for the track
            
        Returns:
            Tuple of (person_id, confidence)
        """
        with self.lock:
            # Check if track already has a persistent assignment
            if track_id in self.track_assignments:
                assigned_person = self.track_assignments[track_id]
                if assigned_person in self.gallery:
                    # Get representative embedding (centroid) for comparison
                    representative_embedding = self._get_person_representative_embedding(assigned_person)
                    
                    # Verify the assignment is still valid
                    similarity = cosine_similarity(
                        features.reshape(1, -1),
                        representative_embedding.reshape(1, -1)
                    )[0][0]
                    
                    # If similarity is reasonable, keep the assignment and update gallery
                    if similarity >= self.similarity_threshold * 0.8:  # Higher threshold for existing assignments
                        self.track_confidence[track_id] = similarity
                        
                        # Update person's gallery with new features
                        self._update_person_gallery(assigned_person, track_id, features)
                        
                        logger.debug(f"Track {track_id} confirmed as '{assigned_person}' (similarity: {similarity:.3f})")
                        return assigned_person, similarity
                    else:
                        logger.info(f"Track {track_id} assignment to '{assigned_person}' invalid (similarity: {similarity:.3f})")
                        # Remove invalid assignment
                        del self.track_assignments[track_id]
                        if track_id in self.track_confidence:
                            del self.track_confidence[track_id]
            
            # Store features for this track
            self.track_features[track_id] = features
            
            # Find best match in gallery using representative embeddings
            best_person_id = None
            best_similarity = 0.0
            
            for person_id in self.gallery.keys():
                representative_embedding = self._get_person_representative_embedding(person_id)
                
                if representative_embedding.size > 0:
                    similarity = cosine_similarity(
                        features.reshape(1, -1),
                        representative_embedding.reshape(1, -1)
                    )[0][0]
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_person_id = person_id
            
            # If good match found, assign persistently and update gallery
            if best_similarity >= self.similarity_threshold and best_person_id:
                self.track_assignments[track_id] = best_person_id
                self.track_confidence[track_id] = best_similarity
                
                # Update person's gallery with new features
                self._update_person_gallery(best_person_id, track_id, features)
                
                logger.info(f"Track {track_id} assigned to existing person '{best_person_id}' (similarity: {best_similarity:.3f})")
                return best_person_id, best_similarity
            else:
                # Create new person and assign persistently
                new_person_id = self._create_new_person(track_id, features)
                return new_person_id, 1.0
    
    def _create_new_person(self, track_id: int, features: np.ndarray) -> str:
        """
        Create a new person ID and assign it to a track with dynamic gallery entry
        
        Args:
            track_id: Track identifier
            features: Gait features for the track
            
        Returns:
            New person ID
        """
        new_person_id = f"person_{self.next_person_id}"
        self.next_person_id += 1
        
        # Add to dynamic gallery with initial embedding
        self.gallery[new_person_id] = {
            'embeddings': [features.copy()],
            'centroid': features.copy(),
            'track_sources': [track_id]
        }
        
        # Assign persistently
        self.track_assignments[track_id] = new_person_id
        self.track_confidence[track_id] = 1.0
        
        # Save gallery
        self._save_gallery()
        
        logger.info(f"🆕 New person '{new_person_id}' created for track {track_id} with dynamic gallery")
        
        return new_person_id
    
    def process_frame_identifications(self, frame_track_features: Dict[int, np.ndarray]) -> Dict[int, Tuple[str, float]]:
        """
        Process identifications for all tracks in a frame using enhanced conflict resolution
        to prevent duplicate person assignments
        
        Args:
            frame_track_features: Dictionary mapping track_id -> features
            
        Returns:
            Dictionary mapping track_id -> (person_id, confidence)
        """
        with self.lock:
            results = {}
            frame_person_assignments = {}  # person_id -> (track_id, confidence)
            
            # First pass: Get all potential track identifications with similarity scores
            track_candidates = {}  # track_id -> [(person_id, confidence), ...]
            
            for track_id, features in frame_track_features.items():
                # Get top candidates instead of just best match
                candidates = self._get_identification_candidates(track_id, features, top_k=3)
                track_candidates[track_id] = (features, candidates)
            
            # Second pass: Optimal assignment using Hungarian-like approach
            # Sort tracks by best confidence to process highest confidence matches first
            sorted_tracks = sorted(track_candidates.items(), 
                                 key=lambda x: x[1][1][0][1] if x[1][1] else 0, 
                                 reverse=True)
            
            for track_id, (features, candidates) in sorted_tracks:
                assigned = False
                
                # Try each candidate in order of confidence
                for person_id, confidence in candidates:
                    if person_id not in frame_person_assignments:
                        # This person is available - assign it
                        frame_person_assignments[person_id] = (track_id, confidence)
                        results[track_id] = (person_id, confidence)
                        
                        # Update persistent assignment and gallery
                        self.track_assignments[track_id] = person_id
                        self.track_confidence[track_id] = confidence
                        self._update_person_gallery(person_id, track_id, features)
                        
                        if self.verbose:
                            logger.info(f"✅ Track {track_id} assigned to '{person_id}' (confidence: {confidence:.3f})")
                        
                        assigned = True
                        break
                    else:
                        # Check if this assignment makes more sense than existing one
                        existing_track_id, existing_confidence = frame_person_assignments[person_id]
                        
                        # Multi-factor conflict resolution
                        should_reassign = self._should_reassign_person(
                            track_id, features, confidence,
                            existing_track_id, track_candidates[existing_track_id][0], existing_confidence,
                            person_id
                        )
                        
                        if should_reassign:
                            # Reassign person to current track
                            frame_person_assignments[person_id] = (track_id, confidence)
                            
                            # Update persistent assignment and gallery
                            self.track_assignments[track_id] = person_id
                            self.track_confidence[track_id] = confidence
                            self._update_person_gallery(person_id, track_id, features)
                            
                            # Find alternative assignment for displaced track
                            displaced_features = track_candidates[existing_track_id][0]
                            displaced_candidates = track_candidates[existing_track_id][1]
                            
                            # Try to find alternative assignment for displaced track
                            alternative_assigned = False
                            for alt_person_id, alt_confidence in displaced_candidates[1:]:  # Skip first (was conflicting)
                                if alt_person_id not in frame_person_assignments:
                                    frame_person_assignments[alt_person_id] = (existing_track_id, alt_confidence)
                                    results[existing_track_id] = (alt_person_id, alt_confidence)
                                    
                                    self.track_assignments[existing_track_id] = alt_person_id
                                    self.track_confidence[existing_track_id] = alt_confidence
                                    self._update_person_gallery(alt_person_id, existing_track_id, displaced_features)
                                    
                                    alternative_assigned = True
                                    if self.verbose:
                                        logger.info(f"🔄 Displaced track {existing_track_id} reassigned to '{alt_person_id}' (confidence: {alt_confidence:.3f})")
                                    break
                            
                            # If no alternative found, create new person for displaced track
                            if not alternative_assigned:
                                new_person_id = self._create_new_person(existing_track_id, displaced_features)
                                results[existing_track_id] = (new_person_id, 1.0)
                                if self.verbose:
                                    logger.info(f"🆕 Created new person '{new_person_id}' for displaced track {existing_track_id}")
                            
                            results[track_id] = (person_id, confidence)
                            
                            if self.verbose:
                                logger.info(f"🔄 Reassigned '{person_id}' from track {existing_track_id} to track {track_id}")
                            
                            assigned = True
                            break
                
                # If no assignment found, create new person
                if not assigned:
                    new_person_id = self._create_new_person(track_id, features)
                    results[track_id] = (new_person_id, 1.0)
                    if self.verbose:
                        logger.info(f"🆕 Created new person '{new_person_id}' for track {track_id}")
            
            return results
    
    def process_frame_with_validation(self, frame_track_features: Dict[int, np.ndarray]) -> Dict[int, Tuple[str, float]]:
        """
        Process frame identifications with built-in validation and conflict resolution
        
        Args:
            frame_track_features: Dictionary mapping track_id -> features
            
        Returns:
            Dictionary mapping track_id -> (person_id, confidence)
        """
        # Process identifications
        results = self.process_frame_identifications(frame_track_features)
        
        # Validate assignments
        assignments = {track_id: person_id for track_id, (person_id, _) in results.items()}
        is_valid = self.validate_frame_assignments(assignments)
        
        if not is_valid:
            # Get detailed summary for debugging
            summary = self.get_frame_assignment_summary(results)
            logger.error(f"❌ Frame validation failed! Summary: {summary}")
            
            # Emergency conflict resolution - force unique assignments
            results = self._emergency_conflict_resolution(frame_track_features, results)
            
            # Re-validate
            new_assignments = {track_id: person_id for track_id, (person_id, _) in results.items()}
            if self.validate_frame_assignments(new_assignments):
                logger.info("✅ Emergency conflict resolution successful")
            else:
                logger.error("❌ Emergency conflict resolution failed")
        
        return results
    
    def _emergency_conflict_resolution(self, frame_track_features: Dict[int, np.ndarray], 
                                     conflicted_results: Dict[int, Tuple[str, float]]) -> Dict[int, Tuple[str, float]]:
        """
        Emergency conflict resolution when normal methods fail
        
        Args:
            frame_track_features: Original frame features
            conflicted_results: Results with conflicts
            
        Returns:
            Conflict-free results
        """
        logger.warning("🚨 Applying emergency conflict resolution")
        
        # Group tracks by assigned person
        person_to_tracks = {}
        for track_id, (person_id, confidence) in conflicted_results.items():
            if person_id not in person_to_tracks:
                person_to_tracks[person_id] = []
            person_to_tracks[person_id].append((track_id, confidence))
        
        # Find conflicts
        conflicts = {pid: tracks for pid, tracks in person_to_tracks.items() if len(tracks) > 1}
        
        # Resolve each conflict
        new_results = conflicted_results.copy()
        
        for person_id, conflicted_tracks in conflicts.items():
            # Sort by confidence (highest first)
            conflicted_tracks.sort(key=lambda x: x[1], reverse=True)
            
            # Keep the highest confidence track for this person
            winner_track_id, winner_confidence = conflicted_tracks[0]
            logger.info(f"   🏆 Keeping track {winner_track_id} for '{person_id}' (conf: {winner_confidence:.3f})")
            
            # Create new persons for the other tracks
            for track_id, confidence in conflicted_tracks[1:]:
                features = frame_track_features[track_id]
                new_person_id = self._create_new_person(track_id, features)
                new_results[track_id] = (new_person_id, 1.0)
                logger.info(f"   🆕 Created '{new_person_id}' for displaced track {track_id}")
        
        return new_results
    
    def _get_identification_candidates(self, track_id: int, features: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Get top-k identification candidates for a track
        
        Args:
            track_id: Track identifier
            features: Feature vector
            top_k: Number of top candidates to return
            
        Returns:
            List of (person_id, confidence) tuples sorted by confidence
        """
        if not self.gallery:
            return []
        
        candidates = []
        
        # Check persistent assignment first
        if track_id in self.track_assignments:
            persistent_person_id = self.track_assignments[track_id]
            if persistent_person_id in self.gallery:
                # Use centroid for persistent assignment comparison
                rep_embedding = self._get_person_representative_embedding(persistent_person_id)
                if rep_embedding.size > 0:
                    similarity = cosine_similarity(
                        features.reshape(1, -1),
                        rep_embedding.reshape(1, -1)
                    )[0][0]
                    
                    # Boost confidence for persistent assignments if similarity is reasonable
                    if similarity >= self.similarity_threshold * 0.8:  # Lower threshold for existing assignments
                        candidates.append((persistent_person_id, similarity * 1.1))  # Small boost
        
        # Compare with all persons in gallery
        for person_id in self.gallery:
            # Skip if already added as persistent assignment
            if track_id in self.track_assignments and self.track_assignments[track_id] == person_id:
                continue
                
            rep_embedding = self._get_person_representative_embedding(person_id)
            if rep_embedding.size > 0:
                similarity = cosine_similarity(
                    features.reshape(1, -1),
                    rep_embedding.reshape(1, -1)
                )[0][0]
                
                if similarity >= self.similarity_threshold:
                    candidates.append((person_id, similarity))
        
        # Sort by confidence and return top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]
    
    def _should_reassign_person(self, track_id: int, features: np.ndarray, confidence: float,
                               existing_track_id: int, existing_features: np.ndarray, existing_confidence: float,
                               person_id: str) -> bool:
        """
        Determine if a person should be reassigned from one track to another
        
        Args:
            track_id: New track requesting assignment
            features: New track features
            confidence: New track confidence
            existing_track_id: Currently assigned track
            existing_features: Existing track features
            existing_confidence: Existing track confidence
            person_id: Person being contested
            
        Returns:
            True if person should be reassigned to new track
        """
        # Factor 1: Confidence difference
        confidence_diff = confidence - existing_confidence
        
        # Factor 2: Cross-track similarity (high similarity suggests track fragmentation)
        cross_similarity = cosine_similarity(
            features.reshape(1, -1),
            existing_features.reshape(1, -1)
        )[0][0]
        
        # Factor 3: Persistent assignment history
        new_track_persistent = (track_id in self.track_assignments and 
                               self.track_assignments[track_id] == person_id)
        existing_track_persistent = (existing_track_id in self.track_assignments and 
                                   self.track_assignments[existing_track_id] == person_id)
        
        # Factor 4: Track stability (prefer tracks with longer history)
        new_track_stability = self.track_confidence.get(track_id, 0.0)
        existing_track_stability = self.track_confidence.get(existing_track_id, 0.0)
        
        if self.verbose:
            logger.info(f"   🔍 Reassignment analysis:")
            logger.info(f"     Confidence diff: {confidence_diff:.3f}")
            logger.info(f"     Cross-similarity: {cross_similarity:.3f}")
            logger.info(f"     New track persistent: {new_track_persistent}")
            logger.info(f"     Existing track persistent: {existing_track_persistent}")
            logger.info(f"     New track stability: {new_track_stability:.3f}")
            logger.info(f"     Existing track stability: {existing_track_stability:.3f}")
        
        # Decision logic
        # If cross-similarity is very high, prefer the track with better persistent assignment
        if cross_similarity >= 0.9:
            if new_track_persistent and not existing_track_persistent:
                return True
            elif existing_track_persistent and not new_track_persistent:
                return False
            else:
                # Both or neither persistent, use confidence and stability
                return confidence_diff > 0.05 or new_track_stability > existing_track_stability
        
        # If cross-similarity is moderate, be more conservative
        elif cross_similarity >= 0.7:
            # Require significant confidence advantage
            return confidence_diff > 0.1 and (new_track_persistent or confidence > 0.8)
        
        # If cross-similarity is low, these are likely different people
        else:
            # Don't reassign if tracks look different
            return False
    
    def get_track_sequence_status(self, track_id: int) -> Dict[str, Any]:
        """
        Get status of a track's sequence
        
        Args:
            track_id: Track identifier
            
        Returns:
            Status information
        """
        with self.lock:
            sequence_length = len(self.track_sequences.get(track_id, []))
            has_features = track_id in self.track_features
            
            return {
                'track_id': track_id,
                'sequence_length': sequence_length,
                'sequence_ready': sequence_length >= self.sequence_length,
                'has_features': has_features,
                'progress': min(1.0, sequence_length / self.sequence_length)
            }
    
    def get_gallery_summary(self) -> Dict[str, Any]:
        """Get summary of the dynamic gallery"""
        with self.lock:
            total_embeddings = 0
            person_details = {}
            
            for person_id, person_data in self.gallery.items():
                embeddings_count = len(person_data['embeddings'])
                track_sources = len(person_data['track_sources'])
                
                person_details[person_id] = {
                    'embeddings_count': embeddings_count,
                    'track_sources': track_sources,
                    'track_ids': person_data['track_sources']
                }
                total_embeddings += embeddings_count
            
            return {
                'num_persons': len(self.gallery),
                'total_embeddings': total_embeddings,
                'avg_embeddings_per_person': total_embeddings / len(self.gallery) if self.gallery else 0,
                'person_ids': list(self.gallery.keys()),
                'person_details': person_details,
                'next_person_id': self.next_person_id,
                'similarity_threshold': self.similarity_threshold,
                'sequence_length': self.sequence_length,
                'max_embeddings_per_person': self.max_embeddings_per_person,
                'min_embeddings_for_centroid': self.min_embeddings_for_centroid,
                'gallery_type': 'dynamic_evolving'
            }
    
    def clear_gallery(self) -> None:
        """Clear the gallery"""
        with self.lock:
            self.gallery.clear()
            self.next_person_id = 1
            self._save_gallery()
            logger.info("🗑️ Gallery cleared")
    
    def cleanup(self) -> None:
        """Cleanup and save data"""
        self._save_gallery()
        logger.info("🧹 Simple Gait Identification cleanup completed")
    
    def cleanup_track(self, track_id: int) -> None:
        """
        Clean up data for a track that has disappeared
        
        Args:
            track_id: Track identifier to clean up
        """
        with self.lock:
            # Remove track data but keep gallery assignments (person persists)
            if track_id in self.track_sequences:
                del self.track_sequences[track_id]
            if track_id in self.track_parsing_sequences:
                del self.track_parsing_sequences[track_id]
            if track_id in self.track_features:
                del self.track_features[track_id]
            # Note: Keep track_assignments and track_confidence for potential re-appearance
            
            logger.debug(f"Cleaned up data for track {track_id}")
    
    def get_track_assignment(self, track_id: int) -> Optional[str]:
        """
        Get the persistent assignment for a track
        
        Args:
            track_id: Track identifier
            
        Returns:
            Person ID if assigned, None otherwise
        """
        with self.lock:
            return self.track_assignments.get(track_id)
    
    def validate_frame_assignments(self, assignments: Dict[int, str]) -> bool:
        """
        Validate that no person is assigned to multiple tracks in the same frame
        
        Args:
            assignments: Dictionary mapping track_id -> person_id
            
        Returns:
            True if assignments are valid (no duplicates)
        """
        person_counts = {}
        for track_id, person_id in assignments.items():
            person_counts[person_id] = person_counts.get(person_id, 0) + 1
        
        duplicates = {pid: count for pid, count in person_counts.items() if count > 1}
        
        if duplicates:
            logger.warning(f"⚠️ Frame assignment validation failed! Duplicate assignments: {duplicates}")
            return False
        
        return True
    
    def get_frame_assignment_summary(self, track_assignments: Dict[int, Tuple[str, float]]) -> Dict:
        """
        Get summary of frame assignments for debugging
        
        Args:
            track_assignments: Dictionary mapping track_id -> (person_id, confidence)
            
        Returns:
            Summary statistics
        """
        persons_in_frame = {}
        total_confidence = 0.0
        
        for track_id, (person_id, confidence) in track_assignments.items():
            if person_id not in persons_in_frame:
                persons_in_frame[person_id] = []
            persons_in_frame[person_id].append((track_id, confidence))
            total_confidence += confidence
        
        duplicates = {pid: tracks for pid, tracks in persons_in_frame.items() if len(tracks) > 1}
        
        return {
            'total_tracks': len(track_assignments),
            'unique_persons': len(persons_in_frame),
            'avg_confidence': total_confidence / len(track_assignments) if track_assignments else 0.0,
            'persons_in_frame': persons_in_frame,
            'duplicates': duplicates,
            'has_duplicates': len(duplicates) > 0
        }
    
    def cleanup_invalid_assignments(self) -> int:
        """
        Clean up invalid track assignments (tracks assigned to non-existent persons)
        
        Returns:
            Number of assignments cleaned up
        """
        with self.lock:
            invalid_tracks = []
            
            for track_id, person_id in self.track_assignments.items():
                if person_id not in self.gallery:
                    invalid_tracks.append(track_id)
            
            for track_id in invalid_tracks:
                del self.track_assignments[track_id]
                if track_id in self.track_confidence:
                    del self.track_confidence[track_id]
                logger.info(f"🧹 Cleaned up invalid assignment: track {track_id}")
            
            return len(invalid_tracks)


def create_simple_gait_identification(**kwargs) -> SimpleGaitIdentification:
    """Factory function to create SimpleGaitIdentification"""
    return SimpleGaitIdentification(**kwargs)


if __name__ == "__main__":
    # Test the simple gait identification system
    print("🧪 Testing Simple Gait Identification")
    
    # Create system with verbose output
    gait_id = SimpleGaitIdentification(
        gallery_file="test_gallery.json",
        similarity_threshold=0.70,  # Lower threshold for testing
        verbose=True
    )
    
    # Test with dummy data
    test_embeddings = [
        np.random.randn(256) * 0.1 + i for i in range(3)
    ]
    
    # Normalize embeddings
    for i, emb in enumerate(test_embeddings):
        test_embeddings[i] = emb / np.linalg.norm(emb)
    
    # Test frame identification
    frame_features = {
        1: test_embeddings[0],
        2: test_embeddings[1],
        3: test_embeddings[2]
    }
    
    results = gait_id.process_frame_identifications(frame_features)
    
    print(f"🔍 Frame identification results:")
    for track_id, (person_id, confidence) in results.items():
        print(f"   Track {track_id}: {person_id} (confidence: {confidence:.3f})")
    
    # Test gallery summary
    summary = gait_id.get_gallery_summary()
    print(f"📊 Initial Gallery summary:")
    print(f"   • Persons: {summary['num_persons']}")
    print(f"   • Total embeddings: {summary['total_embeddings']}")
    print(f"   • Avg embeddings/person: {summary['avg_embeddings_per_person']:.1f}")
    
    # Test dynamic gallery evolution - add more features for the same people
    print(f"\n🔄 Testing dynamic gallery evolution...")
    
    # Simulate more track data for person_1 (similar to first embedding with small noise)
    for i in range(3):
        similar_feature = test_embeddings[0] + np.random.randn(256) * 0.01  # Much smaller noise
        similar_feature = similar_feature / np.linalg.norm(similar_feature)
        person_id, confidence = gait_id.identify_or_assign_person(10 + i, similar_feature)
        print(f"   Track {10 + i}: {person_id} (confidence: {confidence:.3f})")
    
    # Test with another person
    for i in range(2):
        similar_feature = test_embeddings[1] + np.random.randn(256) * 0.01  # Similar to person_2
        similar_feature = similar_feature / np.linalg.norm(similar_feature)
        person_id, confidence = gait_id.identify_or_assign_person(20 + i, similar_feature)
        print(f"   Track {20 + i}: {person_id} (confidence: {confidence:.3f})")
    
    # Check evolved gallery
    summary = gait_id.get_gallery_summary()
    print(f"\n📊 Evolved Gallery summary:")
    print(f"   • Persons: {summary['num_persons']}")
    print(f"   • Total embeddings: {summary['total_embeddings']}")
    print(f"   • Avg embeddings/person: {summary['avg_embeddings_per_person']:.1f}")
    
    for person_id, details in summary['person_details'].items():
        print(f"   • {person_id}: {details['embeddings_count']} embeddings from tracks {details['track_sources']}")
    
    # Test with another similar feature (should match using centroid)
    test_feature = test_embeddings[0] + np.random.randn(256) * 0.005  # Very small noise
    test_feature = test_feature / np.linalg.norm(test_feature)
    
    person_id, confidence = gait_id.identify_or_assign_person(99, test_feature)
    print(f"\n🎯 Final test with evolved centroid: {person_id} (confidence: {confidence:.3f})")
    
    # Cleanup
    gait_id.cleanup()
    print("✅ Dynamic gallery test completed")
