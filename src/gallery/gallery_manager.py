#!/usr/bin/env python3
"""
Gallery Manager for Person Identification
Comprehensive system for storing, managing, and analyzing XGait features with persistent storage and PCA visualization
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


class GalleryManager:
    """
    Comprehensive gallery management system for person identification using XGait features
    
    Features:
    - Persistent storage and loading of gallery data
    - Feature matching and identification with confidence scoring
    - Automatic gallery updates for unseen individuals
    - PCA visualization and analysis of feature separability
    - Thread-safe operations for concurrent access
    - Quality metrics and statistics
    """
    
    def __init__(self, 
                 gallery_dir: str = "gallery_data",
                 similarity_threshold: float = 0.7,
                 auto_add_threshold: float = 0.5,
                 max_features_per_person: int = 20,
                 pca_components: int = 2):
        """
        Initialize Gallery Manager
        
        Args:
            gallery_dir: Directory to store persistent gallery data
            similarity_threshold: Minimum similarity for positive identification
            auto_add_threshold: Threshold below which new tracks are auto-added as new persons
            max_features_per_person: Maximum number of feature vectors stored per person
            pca_components: Number of PCA components for visualization
        """
        self.gallery_dir = Path(gallery_dir)
        self.gallery_dir.mkdir(exist_ok=True)
        
        self.similarity_threshold = similarity_threshold
        self.auto_add_threshold = auto_add_threshold
        self.max_features_per_person = max_features_per_person
        self.pca_components = pca_components
        
        # Gallery data structures
        self.gallery_features = {}  # person_id -> List[feature_vectors]
        self.gallery_metadata = {}  # person_id -> metadata (timestamps, track_ids, etc.)
        self.gallery_stats = {}     # person_id -> statistics
        
        # Analysis components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_components)
        self.pca_fitted = False
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Tracking
        self.identification_history = []  # History of all identifications
        self.next_person_id = 1
        
        # Load existing gallery
        self._load_gallery()
        
        logger.info(f"✅ Gallery Manager initialized")
        logger.info(f"   Gallery directory: {self.gallery_dir}")
        logger.info(f"   Similarity threshold: {self.similarity_threshold}")
        logger.info(f"   Auto-add threshold: {self.auto_add_threshold}")
        logger.info(f"   Loaded persons: {len(self.gallery_features)}")
    
    def _load_gallery(self) -> None:
        """Load gallery data from persistent storage"""
        try:
            # Load features
            features_file = self.gallery_dir / "features.pkl"
            if features_file.exists():
                with open(features_file, 'rb') as f:
                    self.gallery_features = pickle.load(f)
            
            # Load metadata
            metadata_file = self.gallery_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.gallery_metadata = json.load(f)
            
            # Load statistics
            stats_file = self.gallery_dir / "stats.json"
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    self.gallery_stats = json.load(f)
            
            # Load identification history
            history_file = self.gallery_dir / "history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    self.identification_history = json.load(f)
            
            # Load configuration if it exists (will override constructor parameters)
            config_file = self.gallery_dir / "config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    self.similarity_threshold = config_data.get('similarity_threshold', self.similarity_threshold)
                    self.auto_add_threshold = config_data.get('auto_add_threshold', self.auto_add_threshold)
                    self.max_features_per_person = config_data.get('max_features_per_person', self.max_features_per_person)
                    self.pca_components = config_data.get('pca_components', self.pca_components)
                logger.info(f"📋 Loaded configuration: similarity_threshold={self.similarity_threshold}, auto_add_threshold={self.auto_add_threshold}")
            
            # Update next person ID
            if self.gallery_features:
                existing_ids = [int(pid.split('_')[-1]) for pid in self.gallery_features.keys() 
                               if pid.startswith('person_')]
                if existing_ids:
                    self.next_person_id = max(existing_ids) + 1
            
            logger.info(f"📥 Loaded gallery: {len(self.gallery_features)} persons, "
                       f"{sum(len(features) for features in self.gallery_features.values())} total features")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to load gallery data: {e}")
            self._initialize_empty_gallery()
    
    def _initialize_empty_gallery(self) -> None:
        """Initialize empty gallery structures"""
        self.gallery_features = {}
        self.gallery_metadata = {}
        self.gallery_stats = {}
        self.identification_history = []
        self.next_person_id = 1
    
    def save_gallery(self) -> None:
        """Save gallery data to persistent storage"""
        try:
            with self.lock:
                # Save features
                features_file = self.gallery_dir / "features.pkl"
                with open(features_file, 'wb') as f:
                    pickle.dump(self.gallery_features, f)
                
                # Save metadata (convert numpy types to native Python types)
                metadata_file = self.gallery_dir / "metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(self._convert_numpy_types(self.gallery_metadata), f, indent=2)
                
                # Save statistics (convert numpy types to native Python types)
                stats_file = self.gallery_dir / "stats.json"
                with open(stats_file, 'w') as f:
                    json.dump(self._convert_numpy_types(self.gallery_stats), f, indent=2)
                
                # Save identification history (convert numpy types to native Python types)
                history_file = self.gallery_dir / "history.json"
                with open(history_file, 'w') as f:
                    json.dump(self._convert_numpy_types(self.identification_history[-1000:]), f, indent=2)  # Keep last 1000 entries
                
                # Save configuration
                config_file = self.gallery_dir / "config.json"
                config_data = {
                    'similarity_threshold': self.similarity_threshold,
                    'auto_add_threshold': self.auto_add_threshold,
                    'max_features_per_person': self.max_features_per_person,
                    'pca_components': self.pca_components
                }
                with open(config_file, 'w') as f:
                    json.dump(config_data, f, indent=2)
                
            logger.info(f"💾 Gallery saved to {self.gallery_dir}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save gallery: {e}")
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def add_person(self, 
                   person_id: Optional[str], 
                   features: np.ndarray, 
                   track_id: Optional[int] = None,
                   metadata: Optional[Dict] = None) -> str:
        """
        Add a person to the gallery
        
        Args:
            person_id: Person identifier (if None, auto-generates)
            features: Feature vector(s)
            track_id: Associated track ID
            metadata: Additional metadata
            
        Returns:
            Assigned person ID
        """
        with self.lock:
            # Auto-generate person ID if not provided
            if person_id is None:
                person_id = f"person_{self.next_person_id:03d}"
                self.next_person_id += 1
            
            # Ensure features is 1D
            if len(features.shape) > 1:
                features = features.flatten()
            
            # Initialize person if new
            if person_id not in self.gallery_features:
                self.gallery_features[person_id] = []
                self.gallery_metadata[person_id] = {
                    'created': datetime.now().isoformat(),
                    'track_ids': [],
                    'feature_count': 0,
                    'last_updated': datetime.now().isoformat()
                }
                self.gallery_stats[person_id] = {
                    'mean_feature': None,
                    'std_feature': None,
                    'feature_variance': 0.0,
                    'quality_score': 0.0
                }
            
            # Add features
            self.gallery_features[person_id].append(features)
            
            # Manage feature buffer
            if len(self.gallery_features[person_id]) > self.max_features_per_person:
                self.gallery_features[person_id].pop(0)  # Remove oldest
            
            # Update metadata
            if track_id is not None and track_id not in self.gallery_metadata[person_id]['track_ids']:
                self.gallery_metadata[person_id]['track_ids'].append(track_id)
            
            self.gallery_metadata[person_id]['feature_count'] = len(self.gallery_features[person_id])
            self.gallery_metadata[person_id]['last_updated'] = datetime.now().isoformat()
            
            if metadata:
                self.gallery_metadata[person_id].update(metadata)
            
            # Update statistics
            self._update_person_statistics(person_id)
            
            # Mark PCA as needing refit
            self.pca_fitted = False
            
            logger.info(f"👤 Added features for {person_id} (track {track_id}), "
                       f"total features: {len(self.gallery_features[person_id])}")
            
            return person_id
    
    def identify_person(self, 
                       query_features: np.ndarray, 
                       track_id: Optional[int] = None,
                       auto_add: bool = True) -> Tuple[Optional[str], float, Dict]:
        """
        Identify a person based on feature matching
        
        Args:
            query_features: Feature vector to match
            track_id: Associated track ID
            auto_add: Whether to auto-add as new person if no match found
            
        Returns:
            Tuple of (person_id, confidence, metadata)
        """
        with self.lock:
            if len(query_features.shape) > 1:
                query_features = query_features.flatten()
            
            # Normalize query features
            query_features = query_features / (np.linalg.norm(query_features) + 1e-8)
            
            # FIRST: Check if this track ID is already associated with an existing person
            if track_id is not None:
                for person_id, metadata in self.gallery_metadata.items():
                    if track_id in metadata.get('track_ids', []):
                        # Track already assigned to this person - update features and return
                        self.add_person(person_id, query_features, track_id)
                        result_metadata = {
                            'action': 'track_updated',
                            'reason': 'existing_track_assignment'
                        }
                        self._record_identification(person_id, 1.0, track_id, result_metadata)
                        return person_id, 1.0, result_metadata
            
            # SECOND: For new tracks, check if this track is already assigned to any person
            # If not, and this is a new track, create a new person regardless of similarity
            if track_id is not None:
                track_already_assigned = False
                for person_id, metadata in self.gallery_metadata.items():
                    if track_id in metadata.get('track_ids', []):
                        track_already_assigned = True
                        break
                
                if not track_already_assigned:
                    # This is a new track - create a new person for it
                    new_person_id = self.add_person(None, query_features, track_id)
                    result_metadata = {
                        'action': 'auto_added_new_track',
                        'reason': 'new_track_assignment'
                    }
                    self._record_identification(new_person_id, 1.0, track_id, result_metadata)
                    return new_person_id, 1.0, result_metadata
            
            if not self.gallery_features:
                # Empty gallery - auto-add if enabled
                if auto_add:
                    person_id = self.add_person(None, query_features, track_id)
                    result_metadata = {
                        'action': 'auto_added_new',
                        'reason': 'empty_gallery'
                    }
                    self._record_identification(person_id, 1.0, track_id, result_metadata)
                    return person_id, 1.0, result_metadata
                else:
                    return None, 0.0, {'action': 'no_match', 'reason': 'empty_gallery'}
            
            # Find best match
            best_person_id = None
            best_similarity = 0.0
            all_similarities = {}
            
            for person_id, person_features in self.gallery_features.items():
                similarities = []
                
                for stored_features in person_features:
                    # Normalize stored features
                    stored_features = stored_features / (np.linalg.norm(stored_features) + 1e-8)
                    
                    # Compute cosine similarity
                    similarity = np.dot(query_features, stored_features)
                    similarities.append(similarity)
                
                # Use maximum similarity for this person
                max_similarity = max(similarities) if similarities else 0.0
                avg_similarity = np.mean(similarities) if similarities else 0.0
                
                all_similarities[person_id] = {
                    'max': max_similarity,
                    'avg': avg_similarity,
                    'count': len(similarities)
                }
                
                if max_similarity > best_similarity:
                    best_similarity = max_similarity
                    best_person_id = person_id
            
            # Decision logic
            result_metadata = {
                'all_similarities': all_similarities,
                'best_similarity': best_similarity,
                'similarity_threshold': self.similarity_threshold,
                'auto_add_threshold': self.auto_add_threshold
            }
            
            if best_similarity >= self.similarity_threshold:
                # Positive identification
                result_metadata['action'] = 'identified'
                self._record_identification(best_person_id, best_similarity, track_id, result_metadata)
                
                # Update gallery with new features (continuous learning)
                self.add_person(best_person_id, query_features, track_id)
                
                return best_person_id, best_similarity, result_metadata
            
            elif auto_add and best_similarity < self.auto_add_threshold:
                # Auto-add as new person (dissimilar to all existing)
                new_person_id = self.add_person(None, query_features, track_id)
                result_metadata['action'] = 'auto_added_new'
                result_metadata['reason'] = 'below_auto_add_threshold'
                self._record_identification(new_person_id, 1.0, track_id, result_metadata)
                
                return new_person_id, 1.0, result_metadata
            
            else:
                # No confident match
                result_metadata['action'] = 'no_match'
                result_metadata['reason'] = 'below_similarity_threshold'
                self._record_identification(None, best_similarity, track_id, result_metadata)
                
                return None, best_similarity, result_metadata
    
    def _update_person_statistics(self, person_id: str) -> None:
        """Update statistics for a person"""
        if person_id not in self.gallery_features:
            return
        
        features_list = self.gallery_features[person_id]
        if not features_list:
            return
        
        # Convert to numpy array
        features_array = np.array(features_list)
        
        # Compute statistics
        mean_feature = np.mean(features_array, axis=0)
        std_feature = np.std(features_array, axis=0)
        feature_variance = np.mean(np.var(features_array, axis=0))
        
        # Quality score (inverse of variance - lower variance = higher quality)
        quality_score = 1.0 / (1.0 + feature_variance)
        
        self.gallery_stats[person_id] = {
            'mean_feature': mean_feature.tolist(),
            'std_feature': std_feature.tolist(),
            'feature_variance': float(feature_variance),
            'quality_score': float(quality_score)
        }
    
    def _record_identification(self, 
                              person_id: Optional[str], 
                              confidence: float, 
                              track_id: Optional[int],
                              metadata: Dict) -> None:
        """Record identification event in history"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'person_id': person_id,
            'confidence': confidence,
            'track_id': track_id,
            'metadata': metadata
        }
        
        self.identification_history.append(record)
        
        # Keep only recent history in memory
        if len(self.identification_history) > 10000:
            self.identification_history = self.identification_history[-5000:]
    
    def fit_pca_analysis(self) -> None:
        """Fit PCA for feature analysis and visualization"""
        with self.lock:
            if not self.gallery_features:
                logger.warning("⚠️  Cannot fit PCA: gallery is empty")
                return
            
            # Collect all features
            all_features = []
            person_labels = []
            
            for person_id, features_list in self.gallery_features.items():
                for features in features_list:
                    all_features.append(features)
                    person_labels.append(person_id)
            
            if len(all_features) < 2:
                logger.warning("⚠️  Cannot fit PCA: need at least 2 feature vectors")
                return
            
            # Convert to array
            feature_matrix = np.array(all_features)
            
            # Fit scaler and PCA
            try:
                feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
                self.pca.fit(feature_matrix_scaled)
                self.pca_fitted = True
                
                logger.info(f"✅ PCA fitted on {len(all_features)} features from {len(self.gallery_features)} persons")
                logger.info(f"   Explained variance ratio: {self.pca.explained_variance_ratio_}")
                
            except Exception as e:
                logger.error(f"❌ PCA fitting failed: {e}")
                self.pca_fitted = False
    
    def visualize_feature_space(self, 
                               save_path: Optional[str] = None,
                               show_plot: bool = True,
                               query_features: Optional[Dict[str, np.ndarray]] = None) -> Optional[str]:
        """
        Create PCA visualization of the feature space
        
        Args:
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            query_features: Optional query features to overlay (track_id -> features)
            
        Returns:
            Path to saved plot if save_path provided
        """
        if not self.pca_fitted:
            self.fit_pca_analysis()
        
        if not self.pca_fitted:
            logger.error("❌ Cannot create visualization: PCA not fitted")
            return None
        
        # Collect gallery features
        all_features = []
        person_labels = []
        feature_counts = []
        
        for person_id, features_list in self.gallery_features.items():
            for i, features in enumerate(features_list):
                all_features.append(features)
                person_labels.append(person_id)
                feature_counts.append(i + 1)  # Feature number for this person
        
        if len(all_features) < 2:
            logger.error("❌ Cannot create visualization: need at least 2 features")
            return None
        
        # Transform features
        feature_matrix = np.array(all_features)
        feature_matrix_scaled = self.scaler.transform(feature_matrix)
        pca_features = self.pca.transform(feature_matrix_scaled)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Get unique persons and assign colors
        unique_persons = list(set(person_labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_persons)))
        person_color_map = dict(zip(unique_persons, colors))
        
        # Plot gallery features
        for i, person_id in enumerate(unique_persons):
            person_mask = [label == person_id for label in person_labels]
            person_pca = pca_features[person_mask]
            
            plt.scatter(person_pca[:, 0], person_pca[:, 1], 
                       c=[person_color_map[person_id]], 
                       label=f'{person_id} ({np.sum(person_mask)} features)',
                       alpha=0.7, s=60)
            
            # Add convex hull or ellipse for each person
            if len(person_pca) > 2:
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(person_pca)
                    for simplex in hull.simplices:
                        plt.plot(person_pca[simplex, 0], person_pca[simplex, 1], 
                                color=person_color_map[person_id], alpha=0.3, linewidth=1)
                except:
                    pass  # Skip if convex hull fails
        
        # Plot query features if provided
        if query_features:
            for track_id, features in query_features.items():
                if len(features.shape) > 1:
                    features = features.flatten()
                
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                pca_query = self.pca.transform(features_scaled)
                
                plt.scatter(pca_query[0, 0], pca_query[0, 1], 
                           marker='*', s=200, c='red', 
                           label=f'Query Track {track_id}', 
                           edgecolors='black', linewidth=2)
        
        plt.xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('XGait Feature Space Visualization (PCA)\nFeature Separability Analysis')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        total_variance = np.sum(self.pca.explained_variance_ratio_)
        stats_text = f'Total Variance Explained: {total_variance:.1%}\n'
        stats_text += f'Gallery Persons: {len(unique_persons)}\n'
        stats_text += f'Total Features: {len(all_features)}'
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        saved_path = None
        if save_path:
            saved_path = save_path
            plt.savefig(saved_path, dpi=300, bbox_inches='tight')
            logger.info(f"💾 PCA visualization saved to {saved_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return saved_path
    
    def analyze_separability(self) -> Dict[str, Any]:
        """
        Analyze feature separability between different persons
        
        Returns:
            Dictionary with separability metrics
        """
        if not self.gallery_features or len(self.gallery_features) < 2:
            return {'error': 'Need at least 2 persons for separability analysis'}
        
        # Collect features by person
        person_features = {}
        for person_id, features_list in self.gallery_features.items():
            person_features[person_id] = np.array(features_list)
        
        # Compute inter-person similarities
        person_ids = list(person_features.keys())
        n_persons = len(person_ids)
        similarity_matrix = np.zeros((n_persons, n_persons))
        
        for i, person_a in enumerate(person_ids):
            for j, person_b in enumerate(person_ids):
                if i == j:
                    # Intra-person similarity (average within-person similarity)
                    features_a = person_features[person_a]
                    if len(features_a) > 1:
                        similarities = []
                        for k in range(len(features_a)):
                            for l in range(k+1, len(features_a)):
                                sim = cosine_similarity(features_a[k:k+1], features_a[l:l+1])[0, 0]
                                similarities.append(sim)
                        similarity_matrix[i, j] = np.mean(similarities) if similarities else 1.0
                    else:
                        similarity_matrix[i, j] = 1.0
                else:
                    # Inter-person similarity (average between-person similarity)
                    features_a = person_features[person_a]
                    features_b = person_features[person_b]
                    
                    similarities = []
                    for feat_a in features_a:
                        for feat_b in features_b:
                            sim = cosine_similarity(feat_a.reshape(1, -1), feat_b.reshape(1, -1))[0, 0]
                            similarities.append(sim)
                    
                    similarity_matrix[i, j] = np.mean(similarities) if similarities else 0.0
        
        # Compute separability metrics
        intra_person_similarities = np.diag(similarity_matrix)
        inter_person_similarities = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        
        # Separability score (higher is better)
        mean_intra = np.mean(intra_person_similarities)
        mean_inter = np.mean(inter_person_similarities)
        separability_score = (mean_intra - mean_inter) / (mean_intra + mean_inter + 1e-8)
        
        analysis = {
            'n_persons': n_persons,
            'total_features': sum(len(features) for features in person_features.values()),
            'similarity_matrix': similarity_matrix.tolist(),
            'person_ids': person_ids,
            'intra_person_similarity': {
                'mean': float(mean_intra),
                'std': float(np.std(intra_person_similarities)),
                'min': float(np.min(intra_person_similarities)),
                'max': float(np.max(intra_person_similarities))
            },
            'inter_person_similarity': {
                'mean': float(mean_inter),
                'std': float(np.std(inter_person_similarities)),
                'min': float(np.min(inter_person_similarities)),
                'max': float(np.max(inter_person_similarities))
            },
            'separability_score': float(separability_score),
            'quality_assessment': self._assess_quality(separability_score, mean_intra, mean_inter)
        }
        
        return analysis
    
    def _assess_quality(self, separability_score: float, mean_intra: float, mean_inter: float) -> Dict[str, Any]:
        """Assess the quality of feature separability"""
        quality = {}
        
        # Overall quality
        if separability_score > 0.3:
            quality['overall'] = 'Excellent'
        elif separability_score > 0.1:
            quality['overall'] = 'Good'
        elif separability_score > 0.0:
            quality['overall'] = 'Fair'
        else:
            quality['overall'] = 'Poor'
        
        # Specific assessments
        quality['intra_consistency'] = 'High' if mean_intra > 0.8 else 'Medium' if mean_intra > 0.6 else 'Low'
        quality['inter_separation'] = 'High' if mean_inter < 0.3 else 'Medium' if mean_inter < 0.5 else 'Low'
        
        # Recommendations
        recommendations = []
        if mean_intra < 0.7:
            recommendations.append("Consider collecting more features per person for better consistency")
        if mean_inter > 0.4:
            recommendations.append("Features may be too similar between persons - check feature extraction")
        if separability_score < 0.1:
            recommendations.append("Poor separability - consider different feature extraction or preprocessing")
        
        quality['recommendations'] = recommendations
        
        return quality
    
    def get_gallery_summary(self) -> Dict[str, Any]:
        """Get comprehensive gallery summary"""
        with self.lock:
            summary = {
                'persons': len(self.gallery_features),
                'total_features': sum(len(features) for features in self.gallery_features.values()),
                'avg_features_per_person': np.mean([len(features) for features in self.gallery_features.values()]) if self.gallery_features else 0,
                'person_details': {},
                'settings': {
                    'similarity_threshold': self.similarity_threshold,
                    'auto_add_threshold': self.auto_add_threshold,
                    'max_features_per_person': self.max_features_per_person
                },
                'identification_stats': self._get_identification_stats()
            }
            
            # Per-person details
            for person_id in self.gallery_features:
                summary['person_details'][person_id] = {
                    'feature_count': len(self.gallery_features[person_id]),
                    'track_ids': self.gallery_metadata.get(person_id, {}).get('track_ids', []),
                    'created': self.gallery_metadata.get(person_id, {}).get('created', 'unknown'),
                    'last_updated': self.gallery_metadata.get(person_id, {}).get('last_updated', 'unknown'),
                    'quality_score': self.gallery_stats.get(person_id, {}).get('quality_score', 0.0)
                }
            
            return summary
    
    def _get_identification_stats(self) -> Dict[str, Any]:
        """Get identification statistics from history"""
        if not self.identification_history:
            return {'total_identifications': 0}
        
        recent_history = self.identification_history[-1000:]  # Last 1000 identifications
        
        actions = [record['metadata']['action'] for record in recent_history]
        action_counts = {action: actions.count(action) for action in set(actions)}
        
        identified_records = [r for r in recent_history if r['metadata']['action'] == 'identified']
        confidences = [r['confidence'] for r in identified_records]
        
        stats = {
            'total_identifications': len(recent_history),
            'action_breakdown': action_counts,
            'identification_rate': len(identified_records) / len(recent_history) if recent_history else 0.0,
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'confidence_std': np.std(confidences) if confidences else 0.0
        }
        
        return stats
    
    def create_detailed_report(self, save_path: Optional[str] = None) -> str:
        """
        Create a detailed analysis report of the gallery
        
        Args:
            save_path: Optional path to save the report
            
        Returns:
            Report text
        """
        summary = self.get_gallery_summary()
        separability = self.analyze_separability()
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("XGAIT GALLERY ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Gallery Overview
        report_lines.append("📊 GALLERY OVERVIEW")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Persons: {summary['persons']}")
        report_lines.append(f"Total Features: {summary['total_features']}")
        report_lines.append(f"Average Features per Person: {summary['avg_features_per_person']:.1f}")
        report_lines.append("")
        
        # Settings
        report_lines.append("⚙️  CONFIGURATION")
        report_lines.append("-" * 40)
        for key, value in summary['settings'].items():
            report_lines.append(f"{key}: {value}")
        report_lines.append("")
        
        # Person Details
        report_lines.append("👥 PERSON DETAILS")
        report_lines.append("-" * 40)
        for person_id, details in summary['person_details'].items():
            report_lines.append(f"{person_id}:")
            report_lines.append(f"  Features: {details['feature_count']}")
            report_lines.append(f"  Tracks: {details['track_ids']}")
            report_lines.append(f"  Quality Score: {details['quality_score']:.3f}")
            report_lines.append(f"  Created: {details['created']}")
            report_lines.append("")
        
        # Separability Analysis
        if 'error' not in separability:
            report_lines.append("🔍 SEPARABILITY ANALYSIS")
            report_lines.append("-" * 40)
            report_lines.append(f"Separability Score: {separability['separability_score']:.3f}")
            report_lines.append(f"Overall Quality: {separability['quality_assessment']['overall']}")
            report_lines.append("")
            
            report_lines.append("Intra-person Similarity (consistency within persons):")
            intra = separability['intra_person_similarity']
            report_lines.append(f"  Mean: {intra['mean']:.3f}")
            report_lines.append(f"  Std:  {intra['std']:.3f}")
            report_lines.append(f"  Range: {intra['min']:.3f} - {intra['max']:.3f}")
            report_lines.append("")
            
            report_lines.append("Inter-person Similarity (separation between persons):")
            inter = separability['inter_person_similarity']
            report_lines.append(f"  Mean: {inter['mean']:.3f}")
            report_lines.append(f"  Std:  {inter['std']:.3f}")
            report_lines.append(f"  Range: {inter['min']:.3f} - {inter['max']:.3f}")
            report_lines.append("")
            
            if separability['quality_assessment']['recommendations']:
                report_lines.append("💡 RECOMMENDATIONS")
                report_lines.append("-" * 40)
                for rec in separability['quality_assessment']['recommendations']:
                    report_lines.append(f"• {rec}")
                report_lines.append("")
        
        # Identification Statistics
        id_stats = summary['identification_stats']
        if id_stats['total_identifications'] > 0:
            report_lines.append("📈 IDENTIFICATION STATISTICS")
            report_lines.append("-" * 40)
            report_lines.append(f"Total Identifications: {id_stats['total_identifications']}")
            report_lines.append(f"Identification Rate: {id_stats['identification_rate']:.1%}")
            report_lines.append(f"Average Confidence: {id_stats['avg_confidence']:.3f}")
            report_lines.append("")
            
            report_lines.append("Action Breakdown:")
            for action, count in id_stats['action_breakdown'].items():
                percentage = count / id_stats['total_identifications'] * 100
                report_lines.append(f"  {action}: {count} ({percentage:.1f}%)")
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"📄 Report saved to {save_path}")
        
        return report_text
    
    def cleanup(self) -> None:
        """Cleanup and save gallery data"""
        self.save_gallery()
        logger.info("🧹 Gallery Manager cleanup completed")


def create_gallery_manager(**kwargs) -> GalleryManager:
    """Factory function to create a GalleryManager"""
    return GalleryManager(**kwargs)


if __name__ == "__main__":
    # Test the gallery manager
    print("🧪 Testing Gallery Manager")
    
    # Create test gallery
    gallery = GalleryManager(gallery_dir="test_gallery")
    
    # Add some test persons
    for i in range(3):
        person_id = f"test_person_{i+1}"
        for j in range(5):
            # Generate synthetic features
            features = np.random.randn(256) * 0.1 + i  # Person-specific offset
            features = features / np.linalg.norm(features)  # Normalize
            
            gallery.add_person(person_id, features, track_id=i*10+j)
    
    # Test identification
    query_features = np.random.randn(256) * 0.1 + 1  # Similar to person 2
    query_features = query_features / np.linalg.norm(query_features)
    
    person_id, confidence, metadata = gallery.identify_person(query_features, track_id=999)
    print(f"🔍 Identification result: {person_id} (confidence: {confidence:.3f})")
    
    # Test analysis
    summary = gallery.get_gallery_summary()
    print(f"📊 Gallery summary: {summary['persons']} persons, {summary['total_features']} features")
    
    separability = gallery.analyze_separability()
    print(f"🎯 Separability score: {separability['separability_score']:.3f}")
    
    # Test visualization
    gallery.visualize_feature_space(save_path="test_pca.png", show_plot=False)
    
    # Generate report
    report = gallery.create_detailed_report("test_report.txt")
    print("📄 Report generated")
    
    # Cleanup
    gallery.cleanup()
    print("✅ Gallery Manager test completed")
