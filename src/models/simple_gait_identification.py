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
                 similarity_threshold: float = 0.7,
                 sequence_length: int = 30):
        """
        Initialize Simple Gait Identification System
        
        Args:
            gallery_file: Path to save/load gallery data
            similarity_threshold: Minimum similarity for positive identification
            sequence_length: Number of frames to use for feature extraction
        """
        self.gallery_file = Path(gallery_file)
        self.gallery_file.parent.mkdir(exist_ok=True)
        
        self.similarity_threshold = similarity_threshold
        self.sequence_length = sequence_length
        
        # Gallery data - person_name -> embedding
        self.gallery = {}
        
        # Track sequences - track_id -> list of silhouettes
        self.track_sequences = {}
        
        # Track parsing sequences - track_id -> list of parsing masks
        self.track_parsing_sequences = {}
        
        # Track features - track_id -> embedding
        self.track_features = {}
        
        # Person assignments for current frame (to avoid duplicates)
        self.current_frame_assignments = {}
        
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
                
                # Convert back to numpy arrays
                self.gallery = {
                    person_id: np.array(embedding) 
                    for person_id, embedding in data.get('gallery', {}).items()
                }
                
                self.next_person_id = data.get('next_person_id', 1)
                
                logger.info(f"📁 Loaded gallery with {len(self.gallery)} persons")
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
            data = {
                'gallery': {
                    person_id: embedding.tolist() 
                    for person_id, embedding in self.gallery.items()
                },
                'next_person_id': self.next_person_id,
                'timestamp': datetime.now().isoformat(),
                'num_persons': len(self.gallery)
            }
            
            with open(self.gallery_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"💾 Gallery saved with {len(self.gallery)} persons")
            
        except Exception as e:
            logger.error(f"❌ Error saving gallery: {e}")
    
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
        Identify a person or assign new ID based on features
        
        Args:
            track_id: Track identifier
            features: XGait embedding
            
        Returns:
            Tuple of (person_id, confidence)
        """
        with self.lock:
            # Check if we already have features for this track
            if track_id not in self.track_features:
                self.track_features[track_id] = features
            
            # Find best match in gallery
            best_person_id = None
            best_similarity = 0.0
            
            for person_id, gallery_embedding in self.gallery.items():
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    features.reshape(1, -1), 
                    gallery_embedding.reshape(1, -1)
                )[0][0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_person_id = person_id
            
            # Decide on identification
            if best_similarity >= self.similarity_threshold:
                # Existing person identified
                return best_person_id, best_similarity
            else:
                # Assign new person ID
                new_person_id = f"person_{self.next_person_id}"
                self.next_person_id += 1
                
                # Add to gallery
                self.gallery[new_person_id] = features.copy()
                
                # Save gallery
                self._save_gallery()
                
                logger.info(f"🆕 New person '{new_person_id}' added to gallery")
                
                return new_person_id, 1.0
    
    def process_frame_identifications(self, frame_track_features: Dict[int, np.ndarray]) -> Dict[int, Tuple[str, float]]:
        """
        Process identifications for all tracks in a frame, ensuring no duplicates
        
        Args:
            frame_track_features: Dictionary mapping track_id -> features
            
        Returns:
            Dictionary mapping track_id -> (person_id, confidence)
        """
        with self.lock:
            results = {}
            assigned_persons = set()
            
            # Sort tracks by confidence to prioritize stronger matches
            track_similarities = []
            
            for track_id, features in frame_track_features.items():
                best_person_id = None
                best_similarity = 0.0
                
                for person_id, gallery_embedding in self.gallery.items():
                    similarity = cosine_similarity(
                        features.reshape(1, -1),
                        gallery_embedding.reshape(1, -1)
                    )[0][0]
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_person_id = person_id
                
                track_similarities.append((track_id, best_person_id, best_similarity, features))
            
            # Sort by similarity (highest first)
            track_similarities.sort(key=lambda x: x[2], reverse=True)
            
            # Assign persons avoiding duplicates
            for track_id, best_person_id, best_similarity, features in track_similarities:
                if (best_similarity >= self.similarity_threshold and 
                    best_person_id not in assigned_persons):
                    # Assign existing person
                    results[track_id] = (best_person_id, best_similarity)
                    assigned_persons.add(best_person_id)
                else:
                    # Assign new person
                    new_person_id = f"person_{self.next_person_id}"
                    self.next_person_id += 1
                    
                    # Add to gallery
                    self.gallery[new_person_id] = features.copy()
                    
                    results[track_id] = (new_person_id, 1.0)
                    assigned_persons.add(new_person_id)
            
            # Save gallery if new persons were added
            if len(assigned_persons) > len([p for p in assigned_persons if p in self.gallery]):
                self._save_gallery()
            
            return results
    
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
        """Get summary of the gallery"""
        with self.lock:
            return {
                'num_persons': len(self.gallery),
                'person_ids': list(self.gallery.keys()),
                'next_person_id': self.next_person_id,
                'similarity_threshold': self.similarity_threshold,
                'sequence_length': self.sequence_length
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


def create_simple_gait_identification(**kwargs) -> SimpleGaitIdentification:
    """Factory function to create SimpleGaitIdentification"""
    return SimpleGaitIdentification(**kwargs)


if __name__ == "__main__":
    # Test the simple gait identification system
    print("🧪 Testing Simple Gait Identification")
    
    # Create system
    gait_id = SimpleGaitIdentification(gallery_file="test_gallery.json")
    
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
    print(f"📊 Gallery summary: {summary}")
    
    # Test with similar features (should match)
    similar_feature = test_embeddings[0] + np.random.randn(256) * 0.01
    similar_feature = similar_feature / np.linalg.norm(similar_feature)
    
    person_id, confidence = gait_id.identify_or_assign_person(4, similar_feature)
    print(f"🎯 Similar feature test: {person_id} (confidence: {confidence:.3f})")
    
    # Cleanup
    gait_id.cleanup()
    print("✅ Test completed")
