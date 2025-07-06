import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PersonData:
    """Data structure for storing multiple embeddings per person"""
    person_name: str
    prototype: np.ndarray
    embeddings: List[np.ndarray]
    qualities: List[float]
    track_associations: List[int]
    creation_time: datetime
    last_update: datetime
    update_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'person_name': self.person_name,
            'prototype': self.prototype.tolist(),
            'embeddings': [emb.tolist() for emb in self.embeddings],
            'qualities': self.qualities,
            'track_associations': self.track_associations,
            'creation_time': self.creation_time.isoformat(),
            'last_update': self.last_update.isoformat(),
            'update_count': self.update_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonData':
        """Create from dictionary"""
        return cls(
            person_name=data['person_name'],
            prototype=np.array(data['prototype']),
            embeddings=[np.array(emb) for emb in data['embeddings']],
            qualities=data['qualities'],
            track_associations=data['track_associations'],
            creation_time=datetime.fromisoformat(data['creation_time']),
            last_update=datetime.fromisoformat(data['last_update']),
            update_count=data['update_count']
        )

class SimpleIdentityGallery:
    """
    Enhanced Identity Gallery with multiple embeddings per person and stable prototypes
    
    Features:
    - Stores multiple embeddings per person with quality scores
    - Uses stable prototypes for matching instead of single embeddings
    - Intelligent updating that adds to collections rather than overwriting
    - Buffer management to prevent unlimited growth
    - Simplified API for easy integration
    """
    
    def __init__(self, similarity_threshold: float = 0.7, min_quality_threshold: float = 0.3, 
                 max_embeddings_per_person: int = 20, prototype_update_strategy: str = "weighted_average",
                 identity_config: Optional[Any] = None):
        """
        Initialize the SimpleIdentityGallery
        
        Args:
            similarity_threshold: Minimum similarity for person matching
            min_quality_threshold: Minimum quality threshold for embeddings
            max_embeddings_per_person: Maximum embeddings to store per person
            prototype_update_strategy: Strategy for updating prototypes
            identity_config: IdentityConfig object for advanced configuration
        """
        # Use config if provided, otherwise use individual parameters
        if identity_config:
            self.similarity_threshold = identity_config.similarity_threshold
            self.min_quality_threshold = identity_config.min_quality_threshold
            self.max_embeddings_per_person = identity_config.max_embeddings_per_person
            self.prototype_update_strategy = identity_config.prototype_update_strategy
            self.min_embeddings_for_stable_prototype = identity_config.min_embeddings_for_stable_prototype
            self.high_confidence_threshold = identity_config.high_confidence_threshold
        else:
            # Use individual parameters for backward compatibility
            self.similarity_threshold = similarity_threshold
            self.min_quality_threshold = min_quality_threshold
            self.max_embeddings_per_person = max_embeddings_per_person
            self.prototype_update_strategy = prototype_update_strategy
            self.min_embeddings_for_stable_prototype = 3
            self.high_confidence_threshold = 0.85
        
        self.person_counter = 1
        self.gallery: Dict[str, PersonData] = {}  # person_name -> PersonData
        self.track_to_person: Dict[int, str] = {}  # track_id -> person_name
        self.person_to_track: Dict[str, int] = {}  # person_name -> track_id (for initial assignment)
        self.track_embedding_buffer: Dict[int, List[np.ndarray]] = {}  # track_id -> list of embeddings
        
        # Statistics
        self.total_embeddings_added = 0
        self.prototype_updates = 0
        self.new_person_creations = 0
        
        logger.info(f"✅ Enhanced Identity Gallery initialized")
        logger.info(f"   Similarity threshold: {similarity_threshold}")
        logger.info(f"   Max embeddings per person: {max_embeddings_per_person}")
        logger.info(f"   Prototype strategy: {prototype_update_strategy}")

    def _generate_person_name(self) -> str:
        """Generate a unique person name"""
        name = f"person_{self.person_counter:03d}"
        self.person_counter += 1
        return name
    
    def _compute_prototype(self, embeddings: List[np.ndarray], qualities: List[float]) -> np.ndarray:
        """
        Compute prototype embedding from multiple embeddings
        
        Args:
            embeddings: List of embedding vectors
            qualities: List of quality scores
            
        Returns:
            Prototype embedding
        """
        if not embeddings:
            raise ValueError("Cannot compute prototype from empty embeddings list")
        
        if len(embeddings) == 1:
            return embeddings[0].copy()
        
        if self.prototype_update_strategy == "best_quality":
            # Use the embedding with highest quality
            best_idx = int(np.argmax(qualities))
            return embeddings[best_idx].copy()
        
        elif self.prototype_update_strategy == "recent_average":
            # Average of the most recent embeddings (up to 5)
            recent_embeddings = embeddings[-5:]
            return np.mean(recent_embeddings, axis=0)
        
        else:  # "weighted_average"
            # Weighted average based on quality scores
            qualities_array = np.array(qualities)
            weights = qualities_array / np.sum(qualities_array)
            
            weighted_sum = np.zeros_like(embeddings[0])
            for emb, weight in zip(embeddings, weights):
                weighted_sum += weight * emb
            
            return weighted_sum
    
    def _add_embedding_to_person(self, person_name: str, embedding: np.ndarray, quality: float, track_id: int):
        """
        Add an embedding to an existing person's collection
        
        Args:
            person_name: Name of the person
            embedding: New embedding to add
            quality: Quality score of the embedding
            track_id: Track ID associated with this embedding
        """
        person_data = self.gallery[person_name]
        
        # Check if we need to make room (remove lowest quality embedding)
        if len(person_data.embeddings) >= self.max_embeddings_per_person:
            # Find and remove the embedding with lowest quality
            min_quality_idx = int(np.argmin(person_data.qualities))
            removed_quality = person_data.qualities[min_quality_idx]  # Store before removal
            person_data.embeddings.pop(min_quality_idx)
            person_data.qualities.pop(min_quality_idx)
            logger.debug(f"Removed lowest quality embedding for {person_name} (quality: {removed_quality:.3f})")
        
        # Add new embedding
        person_data.embeddings.append(embedding.copy())
        person_data.qualities.append(quality)
        
        # Update track associations
        if track_id not in person_data.track_associations:
            person_data.track_associations.append(track_id)
        
        # Recompute prototype
        person_data.prototype = self._compute_prototype(person_data.embeddings, person_data.qualities)
        
        # Update metadata
        person_data.last_update = datetime.now()
        person_data.update_count += 1
        
        self.total_embeddings_added += 1
        self.prototype_updates += 1
        
        logger.debug(f"Added embedding to {person_name} (quality: {quality:.3f}, total: {len(person_data.embeddings)})")
    
    def _create_new_person(self, embedding: np.ndarray, quality: float, track_id: int) -> str:
        """
        Create a new person with the given embedding
        
        Args:
            embedding: Initial embedding
            quality: Quality score of the embedding
            track_id: Track ID associated with this embedding
            
        Returns:
            Name of the newly created person
        """
        person_name = self._generate_person_name()
        
        person_data = PersonData(
            person_name=person_name,
            prototype=embedding.copy(),
            embeddings=[embedding.copy()],
            qualities=[quality],
            track_associations=[track_id],
            creation_time=datetime.now(),
            last_update=datetime.now(),
            update_count=1
        )
        
        self.gallery[person_name] = person_data
        self.track_to_person[track_id] = person_name
        self.person_to_track[person_name] = track_id
        
        self.total_embeddings_added += 1
        self.new_person_creations += 1
        
        logger.info(f"🆕 Created new person: {person_name} (quality: {quality:.3f}, total persons: {len(self.gallery)})")
        
        return person_name
    
    def assign_or_update_identities(self, track_embeddings: Dict[int, Tuple[np.ndarray, float]], 
                                   frame_number: int) -> Dict[int, str]:
        """
        Unified method to assign or update identities for tracks in a frame
        
        Args:
            track_embeddings: Dict mapping track_id to (embedding, quality) tuples
            frame_number: Current frame number
            
        Returns:
            Dict mapping track_id to assigned person_name
        """
        assigned = {}
        used_persons = set()
        
        for track_id, (embedding, quality) in track_embeddings.items():
            # Skip low quality embeddings
            if quality < self.min_quality_threshold:
                logger.debug(f"Skipping low quality embedding for track {track_id} (quality: {quality:.3f})")
                continue
            
            # Check if track has a previous assignment that's still available
            prev_person = self.track_to_person.get(track_id)
            if prev_person and prev_person not in used_persons and prev_person in self.gallery:
                # Update existing person with new embedding
                self._add_embedding_to_person(prev_person, embedding, quality, track_id)
                assigned[track_id] = prev_person
                used_persons.add(prev_person)
                logger.debug(f"Updated existing assignment: track {track_id} -> {prev_person}")
                continue
            
            # Find best matching person from gallery (excluding already used ones)
            best_person = None
            best_similarity = 0.0
            
            for person_name, person_data in self.gallery.items():
                if person_name in used_persons:
                    continue
                
                similarity = self._cosine_similarity(embedding, person_data.prototype)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_person = person_name
            
            # Assign based on similarity threshold
            if best_person and best_similarity >= self.similarity_threshold:
                # Match to existing person
                self._add_embedding_to_person(best_person, embedding, quality, track_id)
                assigned[track_id] = best_person
                used_persons.add(best_person)
                self.track_to_person[track_id] = best_person
                logger.debug(f"Matched track {track_id} to existing person {best_person} (similarity: {best_similarity:.3f})")
            else:
                # DO NOT create new person automatically - let manual naming handle it
                # Just skip assignment for now - manual naming will handle new persons
                logger.debug(f"No match found for track {track_id} (best similarity: {best_similarity:.3f}) - will need manual naming")
                continue
                self.unassigned_track_qualities[track_id].append(quality)
                
                # Keep only recent embeddings to prevent memory bloat
                max_unassigned = 20
                if len(self.unassigned_track_embeddings[track_id]) > max_unassigned:
                    self.unassigned_track_embeddings[track_id].pop(0)
                    self.unassigned_track_qualities[track_id].pop(0)
                
                logger.debug(f"Track {track_id} has no match above threshold {self.similarity_threshold:.3f} (best: {best_similarity:.3f}) - storing for manual naming")
        
        return assigned

    
    def add_track_embeddings(self, track_id: int, embeddings: List[np.ndarray], qualities: Optional[List[float]] = None) -> Optional[str]:
        """
        Legacy method: Aggregate embeddings for a track and assign/update person
        
        Args:
            track_id: Track ID
            embeddings: List of embeddings
            qualities: List of quality scores (optional)
            
        Returns:
            Assigned person name or None if no valid embeddings
        """
        if not embeddings:
            return None
        
        # Compute qualities if not provided
        if qualities is None:
            qualities = [0.5] * len(embeddings)  # Default medium quality
        
        # Select best embedding based on quality
        if len(qualities) == len(embeddings):
            best_idx = int(np.argmax(qualities))
            best_embedding = embeddings[best_idx]
            best_quality = qualities[best_idx]
        else:
            # Fallback to average if qualities don't match
            best_embedding = np.mean(embeddings, axis=0)
            best_quality = np.mean(qualities) if qualities else 0.5
        
        # Use the unified method
        result = self.assign_or_update_identities({track_id: (best_embedding, best_quality)}, frame_number=0)
        return result.get(track_id)

    def assign_identities_for_frame(self, track_embeddings: Dict[int, np.ndarray], frame_number: int) -> Dict[int, str]:
        """
        Legacy method: Assign person names to tracks in a frame
        
        Args:
            track_embeddings: Dict mapping track_id to embedding
            frame_number: Current frame number
            
        Returns:
            Dict mapping track_id to assigned person_name
        """
        # Convert to the new format (embedding, quality)
        track_embeddings_with_quality = {}
        for track_id, embedding in track_embeddings.items():
            quality = 0.5  # Default quality for legacy calls
            track_embeddings_with_quality[track_id] = (embedding, quality)
        
        return self.assign_or_update_identities(track_embeddings_with_quality, frame_number)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def save_gallery(self, filepath: str):
        """Save gallery with all person data to a JSON file"""
        data = {
            'persons': {person_name: person_data.to_dict() for person_name, person_data in self.gallery.items()},
            'metadata': {
                'total_embeddings_added': self.total_embeddings_added,
                'prototype_updates': self.prototype_updates,
                'new_person_creations': self.new_person_creations,
                'person_counter': self.person_counter,
                'similarity_threshold': self.similarity_threshold,
                'max_embeddings_per_person': self.max_embeddings_per_person,
                'prototype_update_strategy': self.prototype_update_strategy
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"💾 Gallery saved to {filepath} ({len(self.gallery)} persons)")

    def load_gallery(self, filepath: str, clear_track_associations: bool = True):
        """
        Load gallery with all person data from a JSON file
        
        Args:
            filepath: Path to the gallery JSON file
            clear_track_associations: If True, clears track associations to ensure 
                                    track independence between videos
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Clear existing data
        self.gallery.clear()
        self.track_to_person.clear()
        self.person_to_track.clear()
        
        # Load persons
        for person_name, person_dict in data['persons'].items():
            person_data = PersonData.from_dict(person_dict)
            
            if clear_track_associations:
                # Clear track associations to ensure track independence between videos
                person_data.track_associations = []
                logger.info(f"🔄 Cleared track associations for {person_name} (track independence)")
            else:
                # Rebuild track mappings from most recent track association
                if person_data.track_associations:
                    recent_track = person_data.track_associations[-1]
                    self.track_to_person[recent_track] = person_name
                    self.person_to_track[person_name] = recent_track
            
            self.gallery[person_name] = person_data
        
        # Load metadata
        if 'metadata' in data:
            metadata = data['metadata']
            self.total_embeddings_added = metadata.get('total_embeddings_added', 0)
            self.prototype_updates = metadata.get('prototype_updates', 0)
            self.new_person_creations = metadata.get('new_person_creations', 0)
            self.person_counter = metadata.get('person_counter', 1)
            # Update thresholds if they were saved
            self.similarity_threshold = metadata.get('similarity_threshold', self.similarity_threshold)
            self.max_embeddings_per_person = metadata.get('max_embeddings_per_person', self.max_embeddings_per_person)
            self.prototype_update_strategy = metadata.get('prototype_update_strategy', self.prototype_update_strategy)
        
        # Ensure person_counter is correct
        if self.gallery:
            max_counter = max([int(name.split('_')[-1]) for name in self.gallery.keys() if name.startswith('person_')], default=0)
            self.person_counter = max(self.person_counter, max_counter + 1)
        
        logger.info(f"🔄 Gallery loaded from {filepath} ({len(self.gallery)} persons)")

    def set_track_embedding_buffer(self, track_embedding_buffer: Dict[int, List[np.ndarray]]):
        """
        Set the reference to the track_embedding_buffer from the main app.
        This ensures visualization methods always have access to the latest embeddings.
        """
        self.track_embedding_buffer = track_embedding_buffer

    def get_all_embeddings(self) -> List[Tuple[np.ndarray, str, int, str]]:
        """
        Get all embeddings for visualization.
        Returns:
            List of (embedding, identity, track_id, type) tuples
            type can be "track_embedding", "gallery_embedding", or "prototype"
        """
        all_embeddings = []
        
        # Add all track embeddings (if available)
        if hasattr(self, 'track_embedding_buffer') and self.track_embedding_buffer:
            for track_id, emb_list in self.track_embedding_buffer.items():
                for emb in emb_list:
                    # Try to find assigned identity
                    identity = self.track_to_person.get(track_id, "Unassigned")
                    all_embeddings.append((emb, identity, track_id, "track_embedding"))
        
        # Add gallery embeddings (all stored embeddings per person)
        for person_name, person_data in self.gallery.items():
            # Add prototype
            all_embeddings.append((person_data.prototype, person_name, -1, "prototype"))
            
            # Add all individual embeddings
            for emb in person_data.embeddings:
                all_embeddings.append((emb, person_name, -1, "gallery_embedding"))
        
        return all_embeddings

    def get_track_embeddings_by_track(self) -> Dict[int, List[Tuple[np.ndarray, str]]]:
        """
        Get embeddings organized by track ID for visualization.
        Returns:
            Dict mapping track_id to list of (embedding, identity) tuples
        """
        track_data = {}
        if hasattr(self, 'track_embedding_buffer') and self.track_embedding_buffer:
            for track_id, emb_list in self.track_embedding_buffer.items():
                identity = self.track_to_person.get(track_id, "Unassigned")
                track_data[track_id] = [(emb, identity) for emb in emb_list]
        return track_data
    
    def get_gallery_summary(self) -> Dict[str, Any]:
        """Get comprehensive gallery statistics"""
        return {
            'num_persons': len(self.gallery),
            'person_names': list(self.gallery.keys()),
            'total_embeddings_added': self.total_embeddings_added,
            'prototype_updates': self.prototype_updates,
            'new_person_creations': self.new_person_creations,
            'total_gallery_embeddings': sum(len(person_data.embeddings) for person_data in self.gallery.values()),
            'average_embeddings_per_person': (
                sum(len(person_data.embeddings) for person_data in self.gallery.values()) / 
                max(len(self.gallery), 1)
            ),
            'person_qualities': {
                person_name: {
                    'avg_quality': np.mean(person_data.qualities),
                    'max_quality': np.max(person_data.qualities),
                    'num_embeddings': len(person_data.embeddings),
                    'update_count': person_data.update_count
                }
                for person_name, person_data in self.gallery.items()
            }
        }
    
    def consolidate_similar_persons(self, consolidation_threshold: float = 0.85) -> int:
        """
        Consolidate persons with very similar prototypes
        
        Args:
            consolidation_threshold: Similarity threshold for consolidation
            
        Returns:
            Number of persons consolidated
        """
        consolidated_count = 0
        persons_to_remove = set()
        
        person_names = list(self.gallery.keys())
        logger.info(f"🔍 Starting consolidation with threshold {consolidation_threshold:.3f}")
        logger.info(f"   Initial persons: {person_names}")
        
        # Log similarity matrix for debugging
        logger.info("📊 Similarity Matrix:")
        for i in range(len(person_names)):
            for j in range(i + 1, len(person_names)):
                person_a = person_names[i]
                person_b = person_names[j]
                similarity = self._cosine_similarity(
                    self.gallery[person_a].prototype,
                    self.gallery[person_b].prototype
                )
                logger.info(f"   {person_a} vs {person_b}: {similarity:.3f}")
        
        for i in range(len(person_names)):
            for j in range(i + 1, len(person_names)):
                person_a = person_names[i]
                person_b = person_names[j]
                
                if person_a in persons_to_remove or person_b in persons_to_remove:
                    continue
                
                # Compare prototypes
                similarity = self._cosine_similarity(
                    self.gallery[person_a].prototype,
                    self.gallery[person_b].prototype
                )
                
                if similarity >= consolidation_threshold:
                    # Merge person_b into person_a
                    person_a_data = self.gallery[person_a]
                    person_b_data = self.gallery[person_b]
                    
                    # Combine embeddings and qualities
                    person_a_data.embeddings.extend(person_b_data.embeddings)
                    person_a_data.qualities.extend(person_b_data.qualities)
                    person_a_data.track_associations.extend(person_b_data.track_associations)
                    person_a_data.update_count += person_b_data.update_count
                    
                    # Trim to max size if needed
                    if len(person_a_data.embeddings) > self.max_embeddings_per_person:
                        # Keep the best embeddings
                        indices = np.argsort(person_a_data.qualities)[::-1][:self.max_embeddings_per_person]
                        person_a_data.embeddings = [person_a_data.embeddings[i] for i in indices]
                        person_a_data.qualities = [person_a_data.qualities[i] for i in indices]
                    
                    # Recompute prototype
                    person_a_data.prototype = self._compute_prototype(person_a_data.embeddings, person_a_data.qualities)
                    
                    # Update track mappings
                    for track_id in person_b_data.track_associations:
                        self.track_to_person[track_id] = person_a
                    
                    # Mark person_b for removal
                    persons_to_remove.add(person_b)
                    consolidated_count += 1
                    
                    logger.info(f"🔗 Consolidated {person_b} into {person_a} (similarity: {similarity:.3f})")
        
        # Remove consolidated persons
        for person_name in persons_to_remove:
            del self.gallery[person_name]
            if person_name in self.person_to_track:
                del self.person_to_track[person_name]
        
        if consolidated_count > 0:
            logger.info(f"✅ Consolidated {consolidated_count} persons. Remaining: {len(self.gallery)}")
        
        return consolidated_count
    
    def debug_gallery_state(self, label: str = "Gallery State"):
        """Debug method to print detailed gallery information"""
        logger.info(f"🔍 {label}:")
        logger.info(f"   Total persons: {len(self.gallery)}")
        
        for person_name, person_data in self.gallery.items():
            logger.info(f"   • {person_name}:")
            logger.info(f"     - Embeddings: {len(person_data.embeddings)}")
            logger.info(f"     - Avg quality: {np.mean(person_data.qualities):.3f}")
            logger.info(f"     - Track associations: {person_data.track_associations}")
            logger.info(f"     - Update count: {person_data.update_count}")
            logger.info(f"     - Creation time: {person_data.creation_time.strftime('%H:%M:%S')}")
            logger.info(f"     - Last update: {person_data.last_update.strftime('%H:%M:%S')}")
    
    def get_consolidation_candidates(self, threshold: float = 0.85, consolidation_threshold: float = None) -> List[Tuple[str, str, float]]:
        """Get list of person pairs that would be consolidated at given threshold"""
        # Support both parameter names for backward compatibility
        if consolidation_threshold is not None:
            threshold = consolidation_threshold
            
        candidates = []
        person_names = list(self.gallery.keys())
        
        for i in range(len(person_names)):
            for j in range(i + 1, len(person_names)):
                person_a = person_names[i]
                person_b = person_names[j]
                similarity = self._cosine_similarity(
                    self.gallery[person_a].prototype,
                    self.gallery[person_b].prototype
                )
                if similarity >= threshold:
                    candidates.append((person_a, person_b, similarity))
        
        return sorted(candidates, key=lambda x: x[2], reverse=True)
    
    def smart_consolidate_similar_persons(self, consolidation_threshold: float = 0.85, min_embedding_overlap: int = 3) -> int:
        """
        Smart consolidation that considers multiple factors to prevent over-merging
        
        Args:
            consolidation_threshold: Similarity threshold for consolidation
            min_embedding_overlap: Minimum number of similar embeddings between persons
            
        Returns:
            Number of persons consolidated
        """
        consolidated_count = 0
        persons_to_remove = set()
        
        person_names = list(self.gallery.keys())
        logger.info(f"🧠 Starting SMART consolidation with threshold {consolidation_threshold:.3f}")
        logger.info(f"   Initial persons: {person_names}")
        
        for i in range(len(person_names)):
            for j in range(i + 1, len(person_names)):
                person_a = person_names[i]
                person_b = person_names[j]
                
                if person_a in persons_to_remove or person_b in persons_to_remove:
                    continue
                
                person_a_data = self.gallery[person_a]
                person_b_data = self.gallery[person_b]
                
                # 1. Check prototype similarity
                prototype_similarity = self._cosine_similarity(
                    person_a_data.prototype, person_b_data.prototype
                )
                
                if prototype_similarity < consolidation_threshold:
                    continue
                
                # 2. Check cross-embedding similarities (more robust)
                cross_similarities = []
                for emb_a in person_a_data.embeddings[:5]:  # Check up to 5 embeddings
                    for emb_b in person_b_data.embeddings[:5]:
                        sim = self._cosine_similarity(emb_a, emb_b)
                        cross_similarities.append(sim)
                
                if not cross_similarities:
                    continue
                
                avg_cross_similarity = np.mean(cross_similarities)
                max_cross_similarity = np.max(cross_similarities)
                high_similarity_count = sum(1 for sim in cross_similarities if sim >= consolidation_threshold)
                
                # 3. Check temporal overlap (creation times shouldn't be too far apart)
                time_diff = abs((person_a_data.creation_time - person_b_data.creation_time).total_seconds())
                
                # 4. Decision criteria (more conservative)
                should_consolidate = (
                    prototype_similarity >= consolidation_threshold and
                    avg_cross_similarity >= (consolidation_threshold - 0.1) and  # Slightly lower threshold for average
                    max_cross_similarity >= consolidation_threshold and
                    high_similarity_count >= min_embedding_overlap and
                    time_diff < 300  # Created within 5 minutes of each other
                )
                
                logger.info(f"   📊 {person_a} vs {person_b}:")
                logger.info(f"      Prototype sim: {prototype_similarity:.3f}")
                logger.info(f"      Avg cross sim: {avg_cross_similarity:.3f}")
                logger.info(f"      Max cross sim: {max_cross_similarity:.3f}")
                logger.info(f"      High sim count: {high_similarity_count}/{len(cross_similarities)}")
                logger.info(f"      Time diff: {time_diff:.1f}s")
                logger.info(f"      Decision: {'MERGE' if should_consolidate else 'KEEP SEPARATE'}")
                
                if should_consolidate:
                    # Merge person_b into person_a (same logic as before)
                    person_a_data.embeddings.extend(person_b_data.embeddings)
                    person_a_data.qualities.extend(person_b_data.qualities)
                    person_a_data.track_associations.extend(person_b_data.track_associations)
                    person_a_data.update_count += person_b_data.update_count
                    
                    # Trim to max size if needed
                    if len(person_a_data.embeddings) > self.max_embeddings_per_person:
                        indices = np.argsort(person_a_data.qualities)[::-1][:self.max_embeddings_per_person]
                        person_a_data.embeddings = [person_a_data.embeddings[i] for i in indices]
                        person_a_data.qualities = [person_a_data.qualities[i] for i in indices]
                    
                    # Recompute prototype
                    person_a_data.prototype = self._compute_prototype(person_a_data.embeddings, person_a_data.qualities)
                    
                    # Update track mappings
                    for track_id in person_b_data.track_associations:
                        self.track_to_person[track_id] = person_a
                    
                    persons_to_remove.add(person_b)
                    consolidated_count += 1
                    
                    logger.info(f"🔗 SMART consolidated {person_b} into {person_a}")
        
        # Remove consolidated persons
        for person_name in persons_to_remove:
            del self.gallery[person_name]
            if person_name in self.person_to_track:
                del self.person_to_track[person_name]
        
        if consolidated_count > 0:
            logger.info(f"✅ SMART consolidated {consolidated_count} persons. Remaining: {len(self.gallery)}")
        
        return consolidated_count

    def create_person_from_manual_naming(self, track_id: int, person_name: str) -> bool:
        """
        Create a new person from manual naming using stored unassigned embeddings
        
        Args:
            track_id: Track ID to create person for
            person_name: User-provided name for the person
            
        Returns:
            True if person was created successfully, False otherwise
        """
        if not hasattr(self, 'unassigned_track_embeddings'):
            logger.warning(f"No unassigned embeddings found for track {track_id}")
            return False
            
        if track_id not in self.unassigned_track_embeddings or not self.unassigned_track_embeddings[track_id]:
            logger.warning(f"No stored embeddings found for track {track_id}")
            return False
        
        # Get stored embeddings and qualities
        embeddings = self.unassigned_track_embeddings[track_id]
        qualities = self.unassigned_track_qualities.get(track_id, [0.5] * len(embeddings))
        
        # Create person with the best quality embedding as prototype
        if qualities:
            best_idx = np.argmax(qualities)
            best_embedding = embeddings[best_idx]
            best_quality = qualities[best_idx]
        else:
            best_embedding = embeddings[0]
            best_quality = 0.5
        
        # Create person data
        person_data = PersonData(
            person_name=person_name,
            prototype=best_embedding.copy(),
            embeddings=[emb.copy() for emb in embeddings],
            qualities=qualities.copy(),
            track_associations=[track_id],
            creation_time=datetime.now(),
            last_update=datetime.now(),
            update_count=len(embeddings)
        )
        
        # Add to gallery
        self.gallery[person_name] = person_data
        self.track_to_person[track_id] = person_name
        self.person_to_track[person_name] = track_id
        
        # Clean up unassigned embeddings
        del self.unassigned_track_embeddings[track_id]
        if track_id in self.unassigned_track_qualities:
            del self.unassigned_track_qualities[track_id]
        
        # Update statistics
        self.total_embeddings_added += len(embeddings)
        self.new_person_creations += 1
        
        logger.info(f"✅ Created person '{person_name}' for track {track_id} with {len(embeddings)} embeddings")
        return True
    
    def create_person_from_track(self, person_name: str, track_id: int, 
                                embeddings: List[np.ndarray], qualities: List[float]) -> bool:
        """
        Create a new person with user-provided name from track data
        
        Args:
            person_name: User-provided name for the person
            track_id: Track ID to associate with this person
            embeddings: List of embeddings for this track
            qualities: List of quality scores for the embeddings
            
        Returns:
            bool: True if person was created successfully, False otherwise
        """
        if not embeddings:
            logger.warning(f"Cannot create person {person_name} - no embeddings provided")
            return False
        
        # Create prototype from all embeddings
        prototype = self._compute_prototype(embeddings, qualities)
        
        person_data = PersonData(
            person_name=person_name,
            prototype=prototype,
            embeddings=embeddings.copy(),
            qualities=qualities.copy(),
            track_associations=[track_id],
            creation_time=datetime.now(),
            last_update=datetime.now(),
            update_count=len(embeddings)
        )
        
        self.gallery[person_name] = person_data
        self.track_to_person[track_id] = person_name
        self.person_to_track[person_name] = track_id
        
        self.total_embeddings_added += len(embeddings)
        self.new_person_creations += 1
        
        logger.info(f"🆕 Created person '{person_name}' from track {track_id} with {len(embeddings)} embeddings")
        return True
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the gallery
        
        Returns:
            Dictionary containing detailed statistics
        """
        stats = {
            'gallery_overview': {
                'total_persons': len(self.gallery),
                'total_embeddings': sum(len(person.embeddings) for person in self.gallery.values()),
                'total_tracks_associated': len(self.track_to_person),
                'avg_embeddings_per_person': 0.0,
                'avg_quality_score': 0.0,
            },
            'persons_detail': {},
            'quality_distribution': {
                'high_quality': 0,  # > 0.8
                'medium_quality': 0,  # 0.5 - 0.8
                'low_quality': 0,  # < 0.5
            },
            'embedding_statistics': {
                'min_embeddings': 0,  # Changed from float('inf') to 0
                'max_embeddings': 0,
                'total_embeddings_added': self.total_embeddings_added,
                'new_person_creations': self.new_person_creations,
                'person_updates': self.prototype_updates,
            },
            'track_statistics': {
                'active_tracks': len(self.track_to_person),
                'unassigned_tracks': 0,
            }
        }
        
        if not self.gallery:
            return stats
        
        # Calculate detailed statistics
        total_embeddings = 0
        total_quality = 0
        quality_counts = {'high': 0, 'medium': 0, 'low': 0}
        
        # Initialize min_embeddings properly for comparison
        min_embeddings = float('inf')
        
        for person_name, person_data in self.gallery.items():
            num_embeddings = len(person_data.embeddings)
            avg_quality = np.mean(person_data.qualities) if person_data.qualities else 0.0
            
            # Update person details
            stats['persons_detail'][person_name] = {
                'num_embeddings': num_embeddings,
                'avg_quality': avg_quality,
                'creation_time': person_data.creation_time.isoformat(),
                'last_update': person_data.last_update.isoformat(),
                'update_count': person_data.update_count,
                'associated_tracks': person_data.track_associations,
            }
            
            # Update aggregated statistics
            total_embeddings += num_embeddings
            total_quality += avg_quality * num_embeddings
            
            # Update quality distribution
            for quality in person_data.qualities:
                if quality > 0.8:
                    quality_counts['high'] += 1
                elif quality > 0.5:
                    quality_counts['medium'] += 1
                else:
                    quality_counts['low'] += 1
            
            # Update embedding statistics
            min_embeddings = min(min_embeddings, num_embeddings)
            stats['embedding_statistics']['max_embeddings'] = max(
                stats['embedding_statistics']['max_embeddings'], num_embeddings
            )
        
        # Set final min_embeddings
        stats['embedding_statistics']['min_embeddings'] = min_embeddings if min_embeddings != float('inf') else 0
        
        # Calculate averages
        stats['gallery_overview']['avg_embeddings_per_person'] = total_embeddings / len(self.gallery)
        stats['gallery_overview']['avg_quality_score'] = total_quality / total_embeddings if total_embeddings > 0 else 0.0
        
        # Update quality distribution
        stats['quality_distribution']['high_quality'] = quality_counts['high']
        stats['quality_distribution']['medium_quality'] = quality_counts['medium']
        stats['quality_distribution']['low_quality'] = quality_counts['low']
        
        return stats
    
    def print_comprehensive_report(self) -> None:
        """Print a comprehensive report of the gallery statistics"""
        stats = self.get_comprehensive_stats()
        
        print("\n" + "="*60)
        print("🎭 PERSON IDENTIFICATION GALLERY REPORT")
        print("="*60)
        
        # Gallery overview
        overview = stats['gallery_overview']
        print(f"📊 Gallery Overview:")
        print(f"   • Total Persons: {overview['total_persons']}")
        print(f"   • Total Embeddings: {overview['total_embeddings']}")
        print(f"   • Active Track Associations: {overview['total_tracks_associated']}")
        print(f"   • Avg Embeddings per Person: {overview['avg_embeddings_per_person']:.1f}")
        print(f"   • Avg Quality Score: {overview['avg_quality_score']:.3f}")
        
        # Quality distribution
        quality = stats['quality_distribution']
        print(f"\n📈 Quality Distribution:")
        print(f"   • High Quality (>0.8): {quality['high_quality']}")
        print(f"   • Medium Quality (0.5-0.8): {quality['medium_quality']}")
        print(f"   • Low Quality (<0.5): {quality['low_quality']}")
        
        # Embedding statistics
        emb_stats = stats['embedding_statistics']
        print(f"\n🔢 Embedding Statistics:")
        print(f"   • Min Embeddings per Person: {emb_stats['min_embeddings']}")
        print(f"   • Max Embeddings per Person: {emb_stats['max_embeddings']}")
        print(f"   • Total Embeddings Added: {emb_stats['total_embeddings_added']}")
        print(f"   • New Person Creations: {emb_stats['new_person_creations']}")
        print(f"   • Person Updates: {emb_stats['person_updates']}")
        
        # Individual person details
        if stats['persons_detail']:
            print(f"\n👥 Individual Person Details:")
            for person_name, details in stats['persons_detail'].items():
                print(f"   • {person_name}:")
                print(f"     - Embeddings: {details['num_embeddings']}")
                print(f"     - Avg Quality: {details['avg_quality']:.3f}")
                print(f"     - Updates: {details['update_count']}")
                print(f"     - Associated Tracks: {details['associated_tracks']}")
        
        print("="*60)
    
    def backup_gallery(self, backup_path: Optional[str] = None) -> str:
        """
        Create a backup of the current gallery
        
        Args:
            backup_path: Optional custom backup path
            
        Returns:
            Path to the backup file
        """
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"visualization_analysis/simple_gallery_backup_{timestamp}.json"
        
        # Create backup directory if it doesn't exist
        backup_dir = Path(backup_path).parent
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Save current gallery to backup location
        self.save_gallery(backup_path)
        
        logger.info(f"Gallery backed up to: {backup_path}")
        return backup_path
    
    def restore_gallery(self, backup_path: str, clear_track_associations: bool = True) -> bool:
        """
        Restore gallery from a backup file
        
        Args:
            backup_path: Path to the backup file
            clear_track_associations: Whether to clear track associations
            
        Returns:
            True if restoration was successful
        """
        try:
            if not Path(backup_path).exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Load from backup
            self.load_gallery(backup_path, clear_track_associations)
            logger.info(f"Gallery restored from: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore gallery from {backup_path}: {e}")
            return False
    
    def cleanup_old_backups(self, backup_dir: str = "visualization_analysis", 
                           keep_last_n: int = 5) -> List[str]:
        """
        Clean up old backup files, keeping only the most recent ones
        
        Args:
            backup_dir: Directory containing backup files
            keep_last_n: Number of recent backups to keep
            
        Returns:
            List of deleted backup files
        """
        backup_pattern = "simple_gallery_backup_*.json"
        backup_files = sorted(Path(backup_dir).glob(backup_pattern))
        
        deleted_files = []
        if len(backup_files) > keep_last_n:
            files_to_delete = backup_files[:-keep_last_n]
            for file_path in files_to_delete:
                try:
                    file_path.unlink()
                    deleted_files.append(str(file_path))
                    logger.info(f"Deleted old backup: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to delete backup {file_path}: {e}")
        
        return deleted_files
