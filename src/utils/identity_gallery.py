"""
Identity Gallery Management System for Person Identification
Handles dynamic identity management with collision avoidance and embedding updates
"""
import numpy as np
import threading
import time
import logging
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class IdentityInfo:
    """Information about a person identity in the gallery"""
    person_id: str
    creation_time: datetime
    last_update_time: datetime
    embedding: np.ndarray
    embedding_history: List[np.ndarray]
    associated_tracks: Set[int]
    quality_score: float
    update_count: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'person_id': self.person_id,
            'creation_time': self.creation_time.isoformat(),
            'last_update_time': self.last_update_time.isoformat(),
            'associated_tracks': list(self.associated_tracks),
            'quality_score': self.quality_score,
            'update_count': self.update_count,
            'embedding_history_length': len(self.embedding_history)
        }

    @staticmethod
    def from_dict(d):
        return IdentityInfo(
            person_id=d['person_id'],
            creation_time=datetime.fromisoformat(d['creation_time']),
            last_update_time=datetime.fromisoformat(d['last_update_time']),
            embedding=np.array(d.get('embedding', [])),  # You may need to save/load this in to_dict/save_gallery_state
            embedding_history=[],  # Optionally load if you save it
            associated_tracks=set(d['associated_tracks']),
            quality_score=d['quality_score'],
            update_count=d['update_count']
        )

@dataclass
class TrackEmbeddingRecord:
    """Record of embeddings extracted from a specific track"""
    track_id: int
    frame_number: int
    embedding: np.ndarray
    sequence_quality: float
    extraction_time: datetime
    assigned_identity: Optional[str] = None

class IdentityGalleryManager:
    """
    Advanced gallery manager for person identification with collision avoidance
    
    Features:
    - Dynamic identity creation and updates
    - Frame-level collision avoidance (prevents same identity for multiple tracks in frame)
    - Embedding quality assessment and improvement over time
    - Comprehensive tracking of all embeddings for visualization
    """
    
    def __init__(self, 
                 similarity_threshold: float = 0.7,
                 min_quality_threshold: float = 0.5,
                 max_embeddings_per_identity: int = 10,
                 embedding_update_strategy: str = "weighted_average"):
        """
        Initialize the identity gallery manager
        
        Args:
            similarity_threshold: Minimum similarity for identity matching
            min_quality_threshold: Minimum quality for accepting new embeddings
            max_embeddings_per_identity: Maximum embeddings to store per identity
            embedding_update_strategy: Strategy for updating embeddings ("average", "weighted_average", "replace_best")
        """
        self.similarity_threshold = similarity_threshold
        self.min_quality_threshold = min_quality_threshold
        self.max_embeddings_per_identity = max_embeddings_per_identity
        self.embedding_update_strategy = embedding_update_strategy
        
        # Core data structures
        self.identities: Dict[str, IdentityInfo] = {}
        self.track_embeddings: Dict[int, List[TrackEmbeddingRecord]] = defaultdict(list)
        self.frame_assignments: Dict[int, Dict[int, str]] = defaultdict(dict)  # frame_num -> {track_id: identity}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.next_person_id = 1
        self.total_embeddings_processed = 0
        self.collision_avoided_count = 0
        self.identity_updates_count = 0
        
        logger.info(f"✅ Identity Gallery Manager initialized")
        logger.info(f"   Similarity threshold: {similarity_threshold}")
        logger.info(f"   Update strategy: {embedding_update_strategy}")
    
    def _generate_person_id(self) -> str:
        """Generate a unique person ID"""
        person_id = f"Person_{self.next_person_id:03d}"
        self.next_person_id += 1
        return person_id
    
    def _compute_embedding_quality(self, embedding: np.ndarray) -> float:
        """
        Compute quality score for an embedding
        
        Args:
            embedding: The embedding vector
            
        Returns:
            Quality score between 0 and 1
        """
        # Simple quality metrics
        norm = np.linalg.norm(embedding)
        variance = np.var(embedding)
        sparsity = np.sum(np.abs(embedding) < 0.01) / len(embedding)
        
        # Normalize and combine metrics
        quality = (
            min(norm / 10.0, 1.0) * 0.4 +  # Prefer stronger activations
            min(variance * 10.0, 1.0) * 0.4 +  # Prefer varied activations
            (1.0 - sparsity) * 0.2  # Prefer less sparse embeddings
        )
        
        return min(max(quality, 0.0), 1.0)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _find_best_matching_identity(self, embedding: np.ndarray, 
                                   excluded_identities: Set[str] = None) -> Tuple[Optional[str], float]:
        """
        Find the best matching identity for an embedding
        
        Args:
            embedding: Query embedding
            excluded_identities: Identities to exclude from matching
            
        Returns:
            Tuple of (best_identity, similarity_score)
        """
        if not self.identities:
            return None, 0.0
        
        excluded_identities = excluded_identities or set()
        best_identity = None
        best_similarity = 0.0
        
        for person_id, identity_info in self.identities.items():
            if person_id in excluded_identities:
                continue
                
            similarity = self._cosine_similarity(embedding, identity_info.embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_identity = person_id
        
        return best_identity, best_similarity
    
    def _update_identity_embedding(self, person_id: str, new_embedding: np.ndarray, quality: float):
        """Update an identity's embedding using the configured strategy"""
        identity_info = self.identities[person_id]
        
        if self.embedding_update_strategy == "replace_best":
            if quality > identity_info.quality_score:
                identity_info.embedding = new_embedding.copy()
                identity_info.quality_score = quality
        
        elif self.embedding_update_strategy == "weighted_average":
            # Weight by quality and recency
            current_weight = 0.3 + 0.4 * identity_info.quality_score
            new_weight = 0.3 + 0.4 * quality
            
            total_weight = current_weight + new_weight
            identity_info.embedding = (
                (current_weight * identity_info.embedding + new_weight * new_embedding) / total_weight
            )
            identity_info.quality_score = max(identity_info.quality_score, quality)
        
        else:  # "average"
            # Simple average
            alpha = 1.0 / (identity_info.update_count + 1)
            identity_info.embedding = (
                (1 - alpha) * identity_info.embedding + alpha * new_embedding
            )
            identity_info.quality_score = max(identity_info.quality_score, quality)
        
        # Update metadata
        identity_info.last_update_time = datetime.now()
        identity_info.update_count += 1
        identity_info.embedding_history.append(new_embedding.copy())
        
        # Limit history size
        if len(identity_info.embedding_history) > self.max_embeddings_per_identity:
            identity_info.embedding_history.pop(0)
        
        self.identity_updates_count += 1
    
    def process_track_embedding(self, 
                              track_id: int, 
                              embedding: np.ndarray, 
                              frame_number: int,
                              sequence_quality: float = None) -> Tuple[str, float, bool]:
        """
        Process a new embedding from a track with collision avoidance
        
        Args:
            track_id: ID of the track
            embedding: XGait embedding vector
            frame_number: Frame number where this embedding was extracted
            sequence_quality: Quality of the gait sequence (optional)
            
        Returns:
            Tuple of (assigned_identity, confidence, is_new_identity)
        """
        with self.lock:
            print(f"length of embedding: {len(embedding)}")
            # --- Removed forced 1-to-1 mapping by track_id ---

            # Compute embedding quality if not provided
            if sequence_quality is None:
                sequence_quality = self._compute_embedding_quality(embedding)
            # Check quality threshold
            if sequence_quality < self.min_quality_threshold:
                logger.warning(f"Low quality embedding for track {track_id}, skipping")
                return None, sequence_quality, False
            # Create embedding record
            embedding_record = TrackEmbeddingRecord(
                track_id=track_id,
                frame_number=frame_number,
                embedding=embedding.copy(),
                sequence_quality=sequence_quality,
                extraction_time=datetime.now()
            )
            # Get identities already assigned to other tracks in this frame
            frame_assignments = self.frame_assignments[frame_number]
            excluded_identities = set(frame_assignments.values())
            # Find best and second-best matching identity (excluding those already assigned in this frame)
            best_identity = None
            best_similarity = 0.0
            second_best_similarity = 0.0
            for person_id, identity_info in self.identities.items():
                if person_id in excluded_identities:
                    continue
                similarity = self._cosine_similarity(embedding, identity_info.embedding)
                if similarity > best_similarity:
                    second_best_similarity = best_similarity
                    best_similarity = similarity
                    best_identity = person_id
                elif similarity > second_best_similarity:
                    second_best_similarity = similarity
            similarity = best_similarity
            margin = best_similarity - second_best_similarity
            is_new_identity = False
            assigned_identity = None

            # --- Grace period and mismatch streak logic ---
            grace_period = 15  # frames before new identity can be created for a new track
            mismatch_streak_required = 6  # consecutive mismatches before new identity
            # Track mismatch streaks for each track
            if not hasattr(self, '_track_mismatch_streaks'):
                self._track_mismatch_streaks = defaultdict(int)
            if not hasattr(self, '_track_grace_start_frame'):
                self._track_grace_start_frame = {}
            # Initialize grace period start
            if track_id not in self._track_grace_start_frame:
                self._track_grace_start_frame[track_id] = frame_number
            # --- Temporal Consistency: Require minimum assignment duration before switching ---
            min_assignment_duration = 3  # frames
            last_identity = self.frame_assignments[frame_number - 1].get(track_id) if frame_number > 0 else None
            # Count how long this track has been assigned to the same identity
            assignment_streak = 1
            for prev_frame in range(frame_number - 1, max(frame_number - 10, -1), -1):
                prev_id = self.frame_assignments[prev_frame].get(track_id)
                if prev_id == last_identity:
                    assignment_streak += 1
                else:
                    break
            # If the last assigned identity is still a good match, and streak is long enough, keep it
            if last_identity and best_identity == last_identity and similarity >= (self.similarity_threshold - 0.08):
                assigned_identity = last_identity
                self._track_mismatch_streaks[track_id] = 0
                self._update_identity_embedding(assigned_identity, embedding, sequence_quality)
                self.identities[assigned_identity].associated_tracks.add(track_id)
                logger.info(f"🔄 Track {track_id} kept with {assigned_identity} (temporal consistency, streak={assignment_streak})")
            elif last_identity and assignment_streak >= min_assignment_duration and similarity >= (self.similarity_threshold - 0.15):
                assigned_identity = last_identity
                self._track_mismatch_streaks[track_id] = 0
                self._update_identity_embedding(assigned_identity, embedding, sequence_quality)
                self.identities[assigned_identity].associated_tracks.add(track_id)
                logger.info(f"🔄 Track {track_id} kept with {assigned_identity} (relaxed temporal consistency, streak={assignment_streak})")
            else:
                # --- Adaptive threshold based on fragmentation ---
                fragmentation = len(self.identities) / max(len(self.track_embeddings), 1)
                adaptive_threshold = self.similarity_threshold
                if fragmentation > 2.0:
                    adaptive_threshold = max(0.25, self.similarity_threshold - 0.35)
                elif len(self.identities) > 6:
                    adaptive_threshold = max(0.30, self.similarity_threshold - 0.30)
                elif len(self.identities) <= 3:
                    adaptive_threshold = max(0.25, self.similarity_threshold - 0.40)
                # --- Grace period for new tracks ---
                is_new_track = (frame_number - self._track_grace_start_frame[track_id]) < grace_period
                if is_new_track:
                    logger.info(f"⏳ Track {track_id} in grace period (frame {frame_number}, started {self._track_grace_start_frame[track_id]})")
                # --- Mismatch streak logic with margin check ---
                margin_required = 0.05
                if best_identity and similarity >= adaptive_threshold and margin >= margin_required:
                    assigned_identity = best_identity
                    self._track_mismatch_streaks[track_id] = 0
                    self._update_identity_embedding(best_identity, embedding, sequence_quality)
                    self.identities[best_identity].associated_tracks.add(track_id)
                    logger.info(f"📝 Track {track_id} matched to {best_identity} (similarity: {similarity:.3f}, margin: {margin:.3f}, threshold: {adaptive_threshold:.3f})")
                else:
                    # --- Soft update for near-matches ---
                    soft_update_margin = 0.02
                    soft_update_threshold = max(0.15, adaptive_threshold - 0.10)
                    if best_identity and similarity >= soft_update_threshold and margin >= soft_update_margin:
                        # Soft update: update the gallery with a small weight
                        logger.info(f"🟡 Soft update: Track {track_id} near-matched to {best_identity} (similarity: {similarity:.3f}, margin: {margin:.3f}, threshold: {soft_update_threshold:.3f})")
                        # Use a lower quality for soft update
                        self._update_identity_embedding(best_identity, embedding, min(sequence_quality, 0.3))
                    self._track_mismatch_streaks[track_id] += 1
                    if is_new_track or self._track_mismatch_streaks[track_id] < mismatch_streak_required:
                        # Do not create a new identity yet, keep last or None
                        assigned_identity = last_identity if last_identity else None
                        logger.info(f"⏳ Track {track_id} mismatch streak {self._track_mismatch_streaks[track_id]} (grace or waiting)")
                    else:
                        assigned_identity = self._generate_person_id()
                        is_new_identity = True
                        self._track_mismatch_streaks[track_id] = 0
                        self._track_grace_start_frame[track_id] = frame_number  # reset grace for new identity
                        identity_info = IdentityInfo(
                            person_id=assigned_identity,
                            creation_time=datetime.now(),
                            last_update_time=datetime.now(),
                            embedding=embedding.copy(),
                            embedding_history=[embedding.copy()],
                            associated_tracks={track_id},
                            quality_score=sequence_quality,
                            update_count=1
                        )
                        self.identities[assigned_identity] = identity_info
                        if excluded_identities:
                            self.collision_avoided_count += 1
                            logger.info(f"🚫 Collision avoided: Track {track_id} -> {assigned_identity} "
                                      f"(frame {frame_number} already has {len(excluded_identities)} identities)")
                        else:
                            logger.info(f"🆕 New identity created: Track {track_id} -> {assigned_identity} "
                                      f"(total identities: {len(self.identities)})")
                            if len(self.identities) > 8:
                                logger.warning(f"⚠️  High identity count detected: {len(self.identities)} identities. "
                                         f"Consider running consolidation.")
            # Record frame assignment
            self.frame_assignments[frame_number][track_id] = assigned_identity
            # Store embedding record
            embedding_record.assigned_identity = assigned_identity
            self.track_embeddings[track_id].append(embedding_record)
            self.total_embeddings_processed += 1
            return assigned_identity, similarity if not is_new_identity else 1.0, is_new_identity

    def postprocess_consolidation(self, threshold: float = 0.85):
        """
        Post-hoc consolidation: merge identities with high similarity after a run.
        """
        with self.lock:
            merged = set()
            identity_ids = list(self.identities.keys())
            for i in range(len(identity_ids)):
                for j in range(i + 1, len(identity_ids)):
                    id_a = identity_ids[i]
                    id_b = identity_ids[j]
                    if id_a in merged or id_b in merged:
                        continue
                    emb_a = self.identities[id_a].embedding
                    emb_b = self.identities[id_b].embedding
                    sim = self._cosine_similarity(emb_a, emb_b)
                    if sim >= threshold:
                        # Merge id_b into id_a
                        self.identities[id_a].associated_tracks.update(self.identities[id_b].associated_tracks)
                        self.identities[id_a].embedding_history.extend(self.identities[id_b].embedding_history)
                        self.identities[id_a].update_count += self.identities[id_b].update_count
                        merged.add(id_b)
                        logger.info(f"🔗 Consolidated {id_b} into {id_a} (similarity={sim:.3f})")
            # Remove merged identities
            for mid in merged:
                del self.identities[mid]
            logger.info(f"✅ Post-hoc consolidation complete. Remaining identities: {len(self.identities)}")
    
    def identify_person(self, query_embedding: np.ndarray, track_id: Optional[int] = None) -> Tuple[Optional[str], float, Dict]:
        """
        Identify a person from a query embedding
        
        Args:
            query_embedding: Query embedding vector
            track_id: Optional track ID for context
            
        Returns:
            Tuple of (person_id, confidence, metadata)
        """
        with self.lock:
            best_identity, similarity = self._find_best_matching_identity(query_embedding)
            
            metadata = {
                'total_identities': len(self.identities),
                'query_quality': self._compute_embedding_quality(query_embedding),
                'track_id': track_id
            }
            
            if best_identity and similarity >= self.similarity_threshold:
                metadata['match_quality'] = self.identities[best_identity].quality_score
                metadata['match_update_count'] = self.identities[best_identity].update_count
                return best_identity, similarity, metadata
            else:
                return None, similarity, metadata
    
    def get_all_embeddings(self) -> List[Tuple[np.ndarray, str, int, str]]:
        """
        Get all embeddings for visualization
        
        Returns:
            List of (embedding, identity, track_id, type) tuples
            type can be "track_embedding" or "gallery_embedding"
        """
        with self.lock:
            all_embeddings = []
            
            # Add all track embeddings
            for track_id, records in self.track_embeddings.items():
                for record in records:
                    all_embeddings.append((
                        record.embedding,
                        record.assigned_identity or "Unassigned",
                        track_id,
                        "track_embedding"
                    ))
            
            # Add gallery embeddings
            for person_id, identity_info in self.identities.items():
                all_embeddings.append((
                    identity_info.embedding,
                    person_id,
                    -1,  # No specific track for gallery embeddings
                    "gallery_embedding"
                ))
            
            return all_embeddings
    
    def get_track_embeddings_by_track(self) -> Dict[int, List[Tuple[np.ndarray, str]]]:
        """
        Get embeddings organized by track ID
        
        Returns:
            Dict mapping track_id to list of (embedding, identity) tuples
        """
        with self.lock:
            track_data = {}
            for track_id, records in self.track_embeddings.items():
                track_data[track_id] = [
                    (record.embedding, record.assigned_identity or "Unassigned")
                    for record in records
                ]
            return track_data
    
    def get_gallery_summary(self) -> Dict:
        """Get comprehensive gallery statistics"""
        with self.lock:
            return {
                'num_identities': len(self.identities),
                'identity_ids': list(self.identities.keys()),
                'total_embeddings_processed': self.total_embeddings_processed,
                'collision_avoided_count': self.collision_avoided_count,
                'identity_updates_count': self.identity_updates_count,
                'total_tracks': len(self.track_embeddings),
                'average_embeddings_per_track': (
                    sum(len(records) for records in self.track_embeddings.values()) / 
                    max(len(self.track_embeddings), 1)
                ),
                'gallery_quality_scores': {
                    person_id: info.quality_score 
                    for person_id, info in self.identities.items()
                }
            }
    
    def clear_gallery(self):
        """Clear all gallery data"""
        with self.lock:
            self.identities.clear()
            self.track_embeddings.clear()
            self.frame_assignments.clear()
            self.next_person_id = 1
            self.total_embeddings_processed = 0
            self.collision_avoided_count = 0
            self.identity_updates_count = 0
            # Clear all per-track state
            if hasattr(self, '_track_mismatch_streaks'):
                self._track_mismatch_streaks.clear()
            if hasattr(self, '_track_grace_start_frame'):
                self._track_grace_start_frame.clear()
            logger.info("🗑️ Gallery cleared")
    
    def save_gallery_state(self, filepath: str):
        """Save gallery state to file"""
        with self.lock:
            state = {
                'identities': {
                    person_id: info.to_dict() 
                    for person_id, info in self.identities.items()
                },
                'summary': self.get_gallery_summary(),
                'save_time': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"💾 Gallery state saved to {filepath}")
    
    def load_gallery_state(self, filepath: str):
        """Load gallery state from file"""
        with self.lock:
            with open(filepath, 'r') as f:
                state = json.load(f)
            self.identities.clear()
            for person_id, info in state['identities'].items():
                self.identities[person_id] = IdentityInfo.from_dict(info)
            # Optionally restore other stats if needed
            if self.identities:
                self.next_person_id = max(
                    [int(pid.split('_')[-1]) for pid in self.identities.keys() if pid.startswith("Person_")]
                ) + 1
            else:
                self.next_person_id = 1
            # Clear all per-track state (so track_id is not reused across videos)
            if hasattr(self, '_track_mismatch_streaks'):
                self._track_mismatch_streaks.clear()
            if hasattr(self, '_track_grace_start_frame'):
                self._track_grace_start_frame.clear()
            logger.info(f"🔄 Gallery state loaded from {filepath}")
    
    def export_embeddings_for_analysis(self, output_dir: str):
        """Export all embeddings and metadata for external analysis"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        with self.lock:
            # Export track embeddings
            track_embeddings_data = []
            for track_id, records in self.track_embeddings.items():
                for i, record in enumerate(records):
                    track_embeddings_data.append({
                        'track_id': track_id,
                        'sequence_index': i,
                        'frame_number': record.frame_number,
                        'assigned_identity': record.assigned_identity,
                        'sequence_quality': record.sequence_quality,
                        'extraction_time': record.extraction_time.isoformat(),
                        'embedding': record.embedding.tolist()
                    })
            
            # Export gallery embeddings
            gallery_embeddings_data = []
            for person_id, info in self.identities.items():
                gallery_embeddings_data.append({
                    'person_id': person_id,
                    'creation_time': info.creation_time.isoformat(),
                    'last_update_time': info.last_update_time.isoformat(),
                    'associated_tracks': list(info.associated_tracks),
                    'quality_score': info.quality_score,
                    'update_count': info.update_count,
                    'current_embedding': info.embedding.tolist(),
                    'embedding_history': [emb.tolist() for emb in info.embedding_history]
                })
            
            # Save data
            with open(output_path / 'track_embeddings.json', 'w') as f:
                json.dump(track_embeddings_data, f, indent=2)
            
            with open(output_path / 'gallery_embeddings.json', 'w') as f:
                json.dump(gallery_embeddings_data, f, indent=2)
            
            with open(output_path / 'summary.json', 'w') as f:
                json.dump(self.get_gallery_summary(), f, indent=2)
            
            logger.info(f"📊 Embeddings exported to {output_path}")
    
    def consolidate_fragmented_tracks(self, consolidation_threshold: float = 0.85) -> Dict[str, List[int]]:
        """
        Consolidate tracks that likely belong to the same person based on embedding similarity
        
        Args:
            consolidation_threshold: Similarity threshold for consolidating tracks
            
        Returns:
            Dictionary mapping final identities to list of consolidated track IDs
        """
        with self.lock:
            # Track which identities are to be consolidated
            to_consolidate = defaultdict(list)
            
            # Compare each pair of identities
            identity_ids = list(self.identities.keys())
            for i in range(len(identity_ids)):
                for j in range(i + 1, len(identity_ids)):
                    person_id_a = identity_ids[i]
                    person_id_b = identity_ids[j]
                    
                    # Skip if already marked for consolidation
                    if person_id_a in to_consolidate or person_id_b in to_consolidate:
                        continue
                    
                    # Compute similarity between representative embeddings
                    embedding_a = self.identities[person_id_a].embedding
                    embedding_b = self.identities[person_id_b].embedding
                    similarity = self._cosine_similarity(embedding_a, embedding_b)
                    
                    if similarity >= consolidation_threshold:
                        # Mark both identities for consolidation
                        to_consolidate[person_id_a].append(person_id_b)
                        to_consolidate[person_id_b].append(person_id_a)
            
            # Perform consolidation
            consolidated_identities = {}
            for person_id, merged_ids in to_consolidate.items():
                if person_id in consolidated_identities:
                    continue  # Already merged
                
                # Merge all associated tracks
                merged_tracks = set()
                for id_to_merge in merged_ids:
                    merged_tracks.update(self.identities[id_to_merge].associated_tracks)
                
                # Create new consolidated identity
                new_identity = IdentityInfo(
                    person_id=person_id,
                    creation_time=datetime.now(),
                    last_update_time=datetime.now(),
                    embedding=self.identities[person_id].embedding.copy(),
                    embedding_history=[self.identities[person_id].embedding.copy()],
                    associated_tracks=merged_tracks,
                    quality_score=self.identities[person_id].quality_score,
                    update_count=self.identities[person_id].update_count
                )
                
                consolidated_identities[person_id] = new_identity
                
                # Update all merged identities to point to the new identity
                for id_to_merge in merged_ids:
                    consolidated_identities[id_to_merge] = new_identity
            
            # Replace old identities with consolidated ones
            self.identities = {pid: info for pid, info in consolidated_identities.items()}
            
            logger.info(f"🔄 Consolidated fragmented tracks, {len(to_consolidate)} groups merged")
    
    def run_automatic_consolidation(self, initial_delay: int = 300, interval: int = 600):
        """
        Run automatic consolidation of fragmented tracks at regular intervals
        
        Args:
            initial_delay: Initial delay before starting consolidation (in seconds)
            interval: Interval between consecutive consolidations (in seconds)
        """
        logger.info(f"⏳ Automatic consolidation scheduled: initial_delay={initial_delay}s, interval={interval}s")
        
        time.sleep(initial_delay)
        
        while True:
            with self.lock:
                # Skip if not enough identities for consolidation
                if len(self.identities) < 2:
                    break
            
            logger.info("🔄 Running automatic consolidation of fragmented tracks")
            self.consolidate_fragmented_tracks(consolidation_threshold=0.85)
            
            with self.lock:
                # Avoid too frequent consolidation
                time.sleep(interval)
    
    def analyze_track_fragmentation(self) -> dict:
        """
        Analyze track fragmentation to identify potential issues
        Returns:
            Analysis report with fragmentation metrics
        """
        with self.lock:
            total_tracks = len(self.track_embeddings)
            total_identities = len(self.identities)
            if total_identities == 0:
                return {'fragmentation_ratio': 0, 'tracks_per_identity': 0}
            # Calculate tracks per identity
            tracks_per_identity = {}
            for person_id, identity_info in self.identities.items():
                tracks_per_identity[person_id] = len(identity_info.associated_tracks)
            avg_tracks_per_identity = total_tracks / total_identities
            max_tracks_per_identity = max(tracks_per_identity.values()) if tracks_per_identity else 0
            # Estimate ideal number of people (assume avg 2-3 tracks per person due to occlusions)
            estimated_people = total_tracks / 2.5
            fragmentation_ratio = total_identities / max(estimated_people, 1)
            analysis = {
                'total_tracks': total_tracks,
                'total_identities': total_identities,
                'avg_tracks_per_identity': avg_tracks_per_identity,
                'max_tracks_per_identity': max_tracks_per_identity,
                'estimated_actual_people': int(estimated_people),
                'fragmentation_ratio': fragmentation_ratio,
                'is_fragmented': fragmentation_ratio > 1.5,
                'tracks_per_identity_distribution': tracks_per_identity
            }
            return analysis