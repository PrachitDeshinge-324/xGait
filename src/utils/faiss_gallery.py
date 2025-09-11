"""
FAISS-based Person Gallery System for Person Identification
Simple and efficient embedding-based person identification using FAISS vector search
"""

import faiss
import numpy as np
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PersonEmbedding:
    """Simple embedding with metadata"""
    embedding: np.ndarray
    person_name: str
    track_id: int
    quality: float
    timestamp: datetime
    frame_number: int

class FAISSPersonGallery:
    """
    Simple FAISS-based person gallery for efficient similarity search
    
    Features:
    - Fast similarity search using FAISS
    - Simple person-to-embedding mapping
    - Quality-based embedding management
    - Persistent storage
    """
    
    def __init__(self, 
                 embedding_dim: int = 16384,  # XGait: 256x64 = 16384
                 similarity_threshold: float = 0.91,
                 max_embeddings_per_person: int = 10):
        """
        Initialize FAISS gallery
        
        Args:
            embedding_dim: Dimension of embeddings (XGait: 16384 = 256x64 parts)
            similarity_threshold: Minimum similarity for person identification
            max_embeddings_per_person: Maximum embeddings to store per person
        """
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self.max_embeddings_per_person = max_embeddings_per_person
        
        # Multi-tier similarity thresholds for better identification
        self.high_confidence_threshold = similarity_threshold  # 0.91 -> high confidence match
        self.medium_confidence_threshold = similarity_threshold * 0.82  # ~0.75 -> medium confidence  
        self.low_confidence_threshold = similarity_threshold * 0.65  # ~0.60 -> low confidence
        
        # Temporal matching parameters
        self.temporal_window = 100  # frames to consider for temporal matching
        self.spatial_tolerance = 200  # pixels for spatial proximity matching
        
        # FAISS index for fast similarity search
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner Product (cosine after normalization)
        
        # Person management with temporal tracking
        self.person_embeddings: List[PersonEmbedding] = []  # Parallel to FAISS index
        self.person_names: List[str] = []  # Person names corresponding to embeddings
        self.person_counter = 1
        self.name_to_indices: Dict[str, List[int]] = {}  # Map person name to embedding indices
        self.person_last_seen: Dict[str, int] = {}  # Track when each person was last seen
        self.pending_persons: Dict[str, List[PersonEmbedding]] = {}  # Buffer for potential matches
        
        # Track loaded vs session persons for priority handling
        self.loaded_persons: set = set()  # Persons loaded from persistent gallery
        self.session_persons: set = set()  # Persons created in current session
        
        # Statistics
        self.total_embeddings = 0
        self.total_persons = 0
        
        logger.info("✅ FAISS Person Gallery initialized")
        logger.info(f"   Embedding dimension: {embedding_dim}")
        logger.info(f"   Similarity threshold: {similarity_threshold}")
        logger.info(f"   Max embeddings per person: {max_embeddings_per_person}")
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding for cosine similarity"""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    def _generate_person_name(self) -> str:
        """Generate unique person name"""
        name = f"Person_{self.person_counter:03d}"
        self.person_counter += 1
        return name
    
    def add_person_embedding(self, 
                           person_name: str,
                           track_id: int,
                           embedding: np.ndarray,
                           quality: float,
                           frame_number: int) -> bool:
        """
        Add embedding for a person
        
        Args:
            person_name: Name of the person
            track_id: Track identifier
            embedding: Embedding vector
            quality: Quality score
            frame_number: Frame number
            
        Returns:
            True if embedding was added successfully
        """
        # Check for None or empty embeddings
        if embedding is None or (hasattr(embedding, 'size') and embedding.size == 0):
            logger.debug(f"Skipping None/empty embedding for person {person_name}")
            return False
            
        if embedding.shape[0] != self.embedding_dim:
            logger.error(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embedding.shape[0]}")
            return False
        
        # Normalize embedding
        normalized_embedding = self._normalize_embedding(embedding)
        
        # Create person embedding record
        person_embedding = PersonEmbedding(
            embedding=normalized_embedding,
            person_name=person_name,
            track_id=track_id,
            quality=quality,
            timestamp=datetime.now(),
            frame_number=frame_number
        )
        
        # Check if person exists
        if person_name not in self.name_to_indices:
            self.name_to_indices[person_name] = []
            self.total_persons += 1
            
            # Track as session person if not already loaded from gallery
            if person_name not in self.loaded_persons:
                self.session_persons.add(person_name)
                logger.debug(f"Tracking {person_name} as session person")
        
        # Manage per-person embedding limit
        person_indices = self.name_to_indices[person_name]
        if len(person_indices) >= self.max_embeddings_per_person:
            # Remove lowest quality embedding
            # Filter out None entries (previously marked for removal)
            valid_indices = [idx for idx in person_indices if idx < len(self.person_embeddings) and self.person_embeddings[idx] is not None]
            
            if valid_indices:
                qualities = [(self.person_embeddings[idx].quality, idx) for idx in valid_indices]
                qualities.sort()  # Sort by quality (ascending)
                worst_idx = qualities[0][1]
                
                # Remove from FAISS index (mark for removal)
                # Note: FAISS doesn't support individual deletion, so we'll rebuild index periodically
                self._mark_for_removal(worst_idx)
                
                # Remove from our tracking
                person_indices.remove(worst_idx)
            else:
                # Clean up invalid indices
                self.name_to_indices[person_name] = []
        
        # Add to FAISS index
        self.index.add(normalized_embedding.reshape(1, -1))
        
        # Add to our parallel structures
        embedding_idx = len(self.person_embeddings)
        self.person_embeddings.append(person_embedding)
        self.person_names.append(person_name)
        self.name_to_indices[person_name].append(embedding_idx)
        
        self.total_embeddings += 1
        
        logger.debug(f"Added embedding for {person_name} (quality: {quality:.3f})")
        return True
    
    def _mark_for_removal(self, idx: int):
        """Mark embedding for removal (FAISS doesn't support individual deletion)"""
        if idx < len(self.person_embeddings):
            self.person_embeddings[idx] = None  # Mark as deleted
            # Clean up periodically when we have too many None entries
            none_count = sum(1 for emb in self.person_embeddings if emb is None)
            if none_count > 10:  # Cleanup threshold
                self._cleanup_gallery()
    
    def _cleanup_gallery(self):
        """Clean up the gallery by removing None entries and rebuilding FAISS index"""
        # Create new lists without None entries
        new_embeddings = []
        new_person_names = []
        
        # Create mapping from old index to new index
        old_to_new_idx = {}
        new_idx = 0
        
        for old_idx, emb in enumerate(self.person_embeddings):
            if emb is not None:
                new_embeddings.append(emb)
                new_person_names.append(self.person_names[old_idx])
                old_to_new_idx[old_idx] = new_idx
                new_idx += 1
        
        # Update name_to_indices with new indices
        new_name_to_indices = {}
        for person_name, old_indices in self.name_to_indices.items():
            new_indices = []
            for old_idx in old_indices:
                if old_idx in old_to_new_idx:
                    new_indices.append(old_to_new_idx[old_idx])
            if new_indices:  # Only keep persons with valid embeddings
                new_name_to_indices[person_name] = new_indices
        
        # Rebuild FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        if new_embeddings:
            embeddings_array = np.array([emb.embedding for emb in new_embeddings])
            self.index.add(embeddings_array)
        
        # Update all references
        self.person_embeddings = new_embeddings
        self.person_names = new_person_names
        self.name_to_indices = new_name_to_indices
        self.total_embeddings = len(new_embeddings)
        self.total_persons = len(new_name_to_indices)
        
        logger.debug(f"Gallery cleaned up: {len(new_embeddings)} embeddings remaining")

    def identify_person(self, 
                       embedding: np.ndarray,
                       track_id: Optional[int] = None,
                       frame_number: Optional[int] = None) -> Tuple[Optional[str], float]:
        """
        Identify person from embedding with multi-tier matching
        
        Args:
            embedding: Query embedding
            track_id: Optional track ID for temporal consistency
            frame_number: Optional frame number for temporal analysis
            
        Returns:
            Tuple of (person_name, similarity_score)
        """
        if self.index.ntotal == 0:
            return None, 0.0
        
        # Check for None or empty embeddings
        if embedding is None or (hasattr(embedding, 'size') and embedding.size == 0):
            logger.debug("Query embedding is None or empty")
            return None, 0.0
        
        if embedding.shape[0] != self.embedding_dim:
            logger.error(f"Query embedding dimension mismatch: expected {self.embedding_dim}, got {embedding.shape[0]}")
            return None, 0.0
        
        # Normalize query embedding
        normalized_query = self._normalize_embedding(embedding)
        
        # Search FAISS index for top-k candidates
        k = min(5, self.index.ntotal)  # Get top 5 candidates for analysis
        similarities, indices = self.index.search(normalized_query.reshape(1, -1), k=k)
        
        if len(similarities[0]) == 0:
            return None, 0.0
        
        # Analyze candidates with multi-tier approach
        best_match = None
        best_similarity = 0.0
        
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx >= len(self.person_embeddings) or self.person_embeddings[idx] is None:
                continue
                
            candidate_person = self.person_names[idx]
            
            # High confidence match - accept immediately
            if similarity >= self.high_confidence_threshold:
                self._update_person_last_seen(candidate_person, frame_number)
                return candidate_person, float(similarity)
            
            # Medium/Low confidence - apply additional validation
            if similarity >= self.low_confidence_threshold:
                # Gallery priority boost - loaded persons get higher priority than session persons
                gallery_boost = 0.0
                if candidate_person in self.loaded_persons:
                    gallery_boost = 0.15  # Strong boost for persons from persistent gallery
                    logger.debug(f"Gallery priority boost applied for {candidate_person}: +{gallery_boost:.3f}")
                elif candidate_person in self.session_persons:
                    gallery_boost = 0.0  # No boost for session-only persons
                
                # Temporal validation - prefer recently seen persons
                temporal_boost = 0.0
                if frame_number and candidate_person in self.person_last_seen:
                    frames_since_seen = frame_number - self.person_last_seen[candidate_person]
                    if frames_since_seen <= self.temporal_window:
                        # Apply temporal boost (up to 0.1 boost for recent sightings)
                        # But reduce temporal boost for session persons to avoid overriding gallery priority
                        max_temporal_boost = 0.05 if candidate_person in self.session_persons else 0.1
                        temporal_boost = max_temporal_boost * (1.0 - frames_since_seen / self.temporal_window)
                
                # Combine all boosts
                total_boost = gallery_boost + temporal_boost
                adjusted_similarity = similarity + total_boost
                
                logger.debug(f"Candidate {candidate_person}: sim={similarity:.3f}, "
                           f"gallery_boost={gallery_boost:.3f}, temporal_boost={temporal_boost:.3f}, "
                           f"adjusted={adjusted_similarity:.3f}")
                
                # Check if this is now the best candidate
                if adjusted_similarity > best_similarity and adjusted_similarity >= self.medium_confidence_threshold:
                    best_match = candidate_person
                    best_similarity = similarity  # Store original similarity, not boosted
        
        if best_match:
            self._update_person_last_seen(best_match, frame_number)
            return best_match, float(best_similarity)
        
        return None, float(similarities[0][0]) if len(similarities[0]) > 0 else 0.0
    
    def _update_person_last_seen(self, person_name: str, frame_number: Optional[int]):
        """Update when a person was last seen"""
        if frame_number is not None:
            self.person_last_seen[person_name] = frame_number
    
    def create_new_person(self,
                         track_id: int,
                         embedding: np.ndarray,
                         quality: float,
                         frame_number: int,
                         custom_name: Optional[str] = None) -> str:
        """
        Create a new person in the gallery
        
        Args:
            track_id: Track identifier
            embedding: Embedding vector
            quality: Quality score
            frame_number: Frame number
            custom_name: Optional custom name (otherwise auto-generated)
            
        Returns:
            Person name
        """
        person_name = custom_name if custom_name else self._generate_person_name()
        
        success = self.add_person_embedding(
            person_name, track_id, embedding, quality, frame_number
        )
        
        if success:
            logger.info(f"🆕 Created new person: {person_name}")
            return person_name
        else:
            logger.error(f"Failed to create new person for track {track_id}")
            return None
    
    def merge_persons(self, person1_name: str, person2_name: str) -> bool:
        """
        Merge two persons in the gallery
        
        Args:
            person1_name: Name of first person (will be kept)
            person2_name: Name of second person (will be merged into first)
            
        Returns:
            True if merge was successful
        """
        if person1_name not in self.name_to_indices or person2_name not in self.name_to_indices:
            logger.error(f"Cannot merge - one or both persons not found: {person1_name}, {person2_name}")
            return False
        
        if person1_name == person2_name:
            logger.warning(f"Cannot merge person with themselves: {person1_name}")
            return False
        
        # Get indices for both persons
        person1_indices = self.name_to_indices[person1_name]
        person2_indices = self.name_to_indices[person2_name]
        
        # Transfer embeddings from person2 to person1
        for idx in person2_indices:
            if idx < len(self.person_embeddings) and self.person_embeddings[idx] is not None:
                # Update person name
                self.person_embeddings[idx].person_name = person1_name
                self.person_names[idx] = person1_name
                person1_indices.append(idx)
        
        # Remove person2
        del self.name_to_indices[person2_name]
        self.total_persons -= 1
        
        # Manage embedding limit for merged person
        if len(person1_indices) > self.max_embeddings_per_person:
            # Keep only the best quality embeddings
            qualities = [(self.person_embeddings[idx].quality, idx) for idx in person1_indices 
                        if idx < len(self.person_embeddings) and self.person_embeddings[idx] is not None]
            qualities.sort(reverse=True)  # Sort by quality (descending)
            
            # Keep only the best embeddings
            keep_indices = [idx for _, idx in qualities[:self.max_embeddings_per_person]]
            
            # Mark others for removal
            for idx in person1_indices:
                if idx not in keep_indices:
                    self._mark_for_removal(idx)
            
            self.name_to_indices[person1_name] = keep_indices
        
        logger.info(f"🔗 Merged {person2_name} into {person1_name}")
        return True
    
    def get_gallery_statistics(self) -> Dict:
        """Get gallery statistics"""
        active_embeddings = sum(1 for emb in self.person_embeddings if emb is not None)
        
        return {
            'total_persons': self.total_persons,
            'total_embeddings': active_embeddings,
            'persons': list(self.name_to_indices.keys()),
            'embeddings_per_person': {
                name: len(indices) for name, indices in self.name_to_indices.items()
            },
            'faiss_index_size': self.index.ntotal
        }
    
    def get_person_summary(self, person_name: str) -> Dict:
        """Get summary for a specific person"""
        if person_name not in self.name_to_indices:
            return {}
        
        indices = self.name_to_indices[person_name]
        embeddings = [self.person_embeddings[idx] for idx in indices 
                     if idx < len(self.person_embeddings) and self.person_embeddings[idx] is not None]
        
        if not embeddings:
            return {}
        
        return {
            'person_name': person_name,
            'total_embeddings': len(embeddings),
            'average_quality': np.mean([emb.quality for emb in embeddings]),
            'track_associations': list(set(emb.track_id for emb in embeddings)),
            'creation_time': min(emb.timestamp for emb in embeddings).isoformat(),
            'last_update': max(emb.timestamp for emb in embeddings).isoformat()
        }
    
    def save_gallery(self, filepath: str) -> bool:
        """Save gallery to file"""
        try:
            save_data = {
                'embeddings': [emb for emb in self.person_embeddings if emb is not None],
                'name_to_indices': self.name_to_indices,
                'person_counter': self.person_counter,
                'config': {
                    'embedding_dim': self.embedding_dim,
                    'similarity_threshold': self.similarity_threshold,
                    'max_embeddings_per_person': self.max_embeddings_per_person
                },
                'statistics': self.get_gallery_statistics(),
                'save_time': datetime.now().isoformat()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            
            logger.info(f"💾 FAISS gallery saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save FAISS gallery: {e}")
            return False
    
    def load_gallery(self, filepath: str, clear_track_associations: bool = True) -> bool:
        """Load gallery from file"""
        try:
            if not Path(filepath).exists():
                return False
            
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            # Restore embeddings
            self.person_embeddings = save_data['embeddings']
            self.name_to_indices = save_data['name_to_indices']
            self.person_counter = save_data['person_counter']
            
            # Rebuild FAISS index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.person_names = []
            
            # Add embeddings to FAISS index
            embeddings_to_add = []
            loaded_person_names = set()  # Track which persons are loaded from gallery
            for i, emb in enumerate(self.person_embeddings):
                if emb is not None:
                    # Check for dimension compatibility and try to fix
                    if emb.embedding.shape[0] != self.embedding_dim:
                        logger.warning(f"⚠️ Embedding dimension mismatch: gallery expects {self.embedding_dim}, "
                                     f"loaded embedding has {emb.embedding.shape[0]}. Attempting to reshape/adapt...")
                        
                        # Try to reshape if it's a flattened version of expected dimensions
                        if emb.embedding.size == self.embedding_dim:
                            emb.embedding = emb.embedding.reshape(-1)
                            logger.info("✅ Successfully reshaped embedding to correct dimensions")
                        elif emb.embedding.shape[0] > self.embedding_dim:
                            # Truncate if too large
                            emb.embedding = emb.embedding[:self.embedding_dim]
                            logger.info("✅ Truncated embedding to correct dimensions")
                        elif emb.embedding.shape[0] < self.embedding_dim:
                            # Pad if too small
                            pad_size = self.embedding_dim - emb.embedding.shape[0]
                            emb.embedding = np.pad(emb.embedding, (0, pad_size), mode='constant', constant_values=0)
                            logger.info("✅ Padded embedding to correct dimensions")
                        else:
                            logger.warning("❌ Could not fix embedding dimensions, skipping")
                            continue
                    
                    embeddings_to_add.append(emb.embedding)
                    self.person_names.append(emb.person_name)
                    loaded_person_names.add(emb.person_name)  # Track loaded persons
            
            # Mark all loaded persons as loaded (higher priority)
            self.loaded_persons.update(loaded_person_names)
            logger.info(f"📋 Loaded persons marked for priority: {sorted(loaded_person_names)}")
            
            if embeddings_to_add:
                embeddings_array = np.array(embeddings_to_add)
                self.index.add(embeddings_array)
                logger.info(f"✅ Loaded {len(embeddings_to_add)} embeddings into FAISS index")
            elif self.person_embeddings:
                logger.warning("⚠️ No compatible embeddings found in gallery file. "
                             "This may be due to a dimension mismatch (old 256-dim vs new 16384-dim features).")
                return False
            
            # Update statistics
            self.total_persons = len(self.name_to_indices)
            self.total_embeddings = len([emb for emb in self.person_embeddings if emb is not None])
            
            logger.info(f"🔄 FAISS gallery loaded from {filepath}")
            logger.info(f"   Loaded {self.total_persons} persons with {self.total_embeddings} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load FAISS gallery: {e}")
            return False
    
    def clear_gallery(self):
        """Clear all gallery data"""
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.person_embeddings = []
        self.person_names = []
        self.name_to_indices = {}
        self.person_counter = 1
        self.total_embeddings = 0
        self.total_persons = 0
        
        logger.info("🗑️ FAISS gallery cleared")
    
    def print_gallery_report(self):
        """Print comprehensive gallery report"""
        stats = self.get_gallery_statistics()
        
        print("\n" + "=" * 60)
        print("📊 FAISS GALLERY REPORT")
        print("=" * 60)
        print(f"Total Persons: {stats['total_persons']}")
        print(f"Total Embeddings: {stats['total_embeddings']}")
        print(f"FAISS Index Size: {stats['faiss_index_size']}")
        
        if stats['persons']:
            print("\nPerson Summary:")
            for person_name in sorted(stats['persons']):
                person_summary = self.get_person_summary(person_name)
                if person_summary:
                    print(f"  👤 {person_name}:")
                    print(f"     Embeddings: {person_summary['total_embeddings']}")
                    print(f"     Avg Quality: {person_summary['average_quality']:.3f}")
                    print(f"     Tracks: {person_summary['track_associations']}")
        
        print("=" * 60)
    
    def get_all_embeddings(self) -> List[Tuple[np.ndarray, str, int, str]]:
        """
        Get all embeddings for visualization
        
        Returns:
            List of (embedding, identity, track_id, type) tuples
        """
        all_embeddings = []
        
        for emb in self.person_embeddings:
            if emb is not None:
                all_embeddings.append((
                    emb.embedding,
                    emb.person_name,
                    emb.track_id,
                    "gallery_embedding"
                ))
        
        return all_embeddings
    
    def rebuild_index(self):
        """Rebuild FAISS index to remove deleted embeddings"""
        # Filter out None embeddings
        active_embeddings = []
        active_person_names = []
        new_person_embeddings = []
        new_name_to_indices = {}
        
        for i, emb in enumerate(self.person_embeddings):
            if emb is not None:
                new_idx = len(active_embeddings)
                active_embeddings.append(emb.embedding)
                active_person_names.append(emb.person_name)
                new_person_embeddings.append(emb)
                
                # Update indices mapping
                if emb.person_name not in new_name_to_indices:
                    new_name_to_indices[emb.person_name] = []
                new_name_to_indices[emb.person_name].append(new_idx)
        
        # Rebuild FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        if active_embeddings:
            embeddings_array = np.array(active_embeddings)
            self.index.add(embeddings_array)
        
        # Update internal structures
        self.person_embeddings = new_person_embeddings
        self.person_names = active_person_names
        self.name_to_indices = new_name_to_indices
        self.total_embeddings = len(new_person_embeddings)
        
        logger.info(f"🔄 FAISS index rebuilt: {self.total_embeddings} active embeddings")
    
    def has_person(self, person_name: str) -> bool:
        """Check if person exists in gallery"""
        return person_name in self.name_to_indices
