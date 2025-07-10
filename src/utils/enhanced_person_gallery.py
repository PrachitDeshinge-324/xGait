"""
Enhanced Person Gallery System with Movement State and Orientation Detection
Captures and stores embeddings for different movement patterns and orientations
"""

import json
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import logging
from collections import defaultdict, deque
from enum import Enum
import math

logger = logging.getLogger(__name__)

class MovementType(Enum):
    STEADY = "steady"
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"

class OrientationType(Enum):
    FRONT = "front"
    BACK = "back"
    LEFT_PROFILE = "left_profile"
    RIGHT_PROFILE = "right_profile"

@dataclass
class MovementProfile:
    """Stores movement analysis data"""
    movement_type: MovementType
    orientation_type: OrientationType
    velocity: float
    direction_angle: float
    confidence: float
    frame_count: int
    sequence_length: int

@dataclass
class PersonEmbedding:
    """Enhanced embedding with movement and orientation context"""
    embedding: np.ndarray
    movement_profile: MovementProfile
    quality: float
    timestamp: datetime
    track_id: int
    frame_number: int
    bounding_box: Tuple[int, int, int, int]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'embedding': self.embedding.tolist(),
            'movement_profile': {
                'movement_type': self.movement_profile.movement_type.value,
                'orientation_type': self.movement_profile.orientation_type.value,
                'velocity': float(self.movement_profile.velocity),
                'direction_angle': float(self.movement_profile.direction_angle),
                'confidence': float(self.movement_profile.confidence),
                'frame_count': int(self.movement_profile.frame_count),
                'sequence_length': int(self.movement_profile.sequence_length)
            },
            'quality': self.quality,
            'timestamp': self.timestamp.isoformat(),
            'track_id': int(self.track_id),  # Convert to regular int
            'frame_number': int(self.frame_number),  # Convert to regular int
            'bounding_box': tuple(int(x) for x in self.bounding_box)  # Convert to regular int
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PersonEmbedding':
        """Create from dictionary"""
        mp_data = data['movement_profile']
        movement_profile = MovementProfile(
            movement_type=MovementType(mp_data['movement_type']),
            orientation_type=OrientationType(mp_data['orientation_type']),
            velocity=mp_data['velocity'],
            direction_angle=mp_data['direction_angle'],
            confidence=mp_data['confidence'],
            frame_count=mp_data['frame_count'],
            sequence_length=mp_data['sequence_length']
        )
        
        return cls(
            embedding=np.array(data['embedding']),
            movement_profile=movement_profile,
            quality=data['quality'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            track_id=data['track_id'],
            frame_number=data['frame_number'],
            bounding_box=tuple(data['bounding_box'])
        )

@dataclass
class EnhancedPersonData:
    """Enhanced person data with movement-orientation categorization"""
    person_name: str
    embeddings_by_context: Dict[str, List[PersonEmbedding]]  # "movement_orientation" -> embeddings
    track_associations: List[int]
    creation_time: datetime
    last_update: datetime
    total_embeddings: int
    
    def get_context_key(self, movement: MovementType, orientation: OrientationType) -> str:
        """Generate context key for movement-orientation combination"""
        return f"{movement.value}_{orientation.value}"
    
    def get_all_embeddings(self) -> List[PersonEmbedding]:
        """Get all embeddings across all contexts"""
        all_embeddings = []
        for context_embeddings in self.embeddings_by_context.values():
            all_embeddings.extend(context_embeddings)
        return all_embeddings
    
    def get_context_summary(self) -> Dict[str, int]:
        """Get summary of embeddings per context"""
        return {context: len(embeddings) for context, embeddings in self.embeddings_by_context.items()}

class MovementOrientationAnalyzer:
    """Analyzes movement patterns and body orientations"""
    
    def __init__(self, history_length: int = 10):
        """
        Initialize the movement and orientation analyzer
        
        Args:
            history_length: Number of frames to keep in the history buffer
                           (should match XGait sequence length for best results)
        """
        self.history_length = history_length
        self.track_histories = defaultdict(lambda: deque(maxlen=history_length))
        self.track_crops = defaultdict(lambda: deque(maxlen=history_length))
        self.track_parsing_masks = defaultdict(lambda: deque(maxlen=history_length))
        
    def analyze_movement_and_orientation(self, track_id: int, bbox: Tuple[int, int, int, int], 
                                       crop: np.ndarray, frame_number: int, 
                                       parsing_mask: np.ndarray = None) -> MovementProfile:
        """
        Analyze movement type and orientation for a track
        
        Args:
            track_id: Track identifier
            bbox: Bounding box (x1, y1, x2, y2)
            crop: Person crop image (used for area analysis)
            frame_number: Current frame number
            parsing_mask: Human parsing mask for orientation detection
            
        Returns:
            MovementProfile with detected movement and orientation
        """
        # Store history
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        self.track_histories[track_id].append((center_x, center_y, frame_number))
        self.track_crops[track_id].append(crop)
        if parsing_mask is not None:
            self.track_parsing_masks[track_id].append(parsing_mask)
        
        # Analyze movement (uses bbox area for forward/backward and centroid for sideways)
        movement_type, velocity, direction_angle = self._analyze_movement(track_id)
        
        # Analyze orientation (uses only parsing mask, no crop-based fallback)
        orientation_type = self._analyze_orientation(crop, parsing_mask)
        
        # Calculate confidence based on history length and consistency
        confidence = self._calculate_confidence(track_id, movement_type, orientation_type)
        
        return MovementProfile(
            movement_type=movement_type,
            orientation_type=orientation_type,
            velocity=velocity,
            direction_angle=direction_angle,
            confidence=confidence,
            frame_count=frame_number,
            sequence_length=len(self.track_histories[track_id])
        )
    
    def _analyze_movement(self, track_id: int) -> Tuple[MovementType, float, float]:
        """
        Analyze movement pattern from position and bounding box history.
        Uses centroid trajectory for sideways movement and bounding box area changes for forward/backward movement.
        """
        history = list(self.track_histories[track_id])
        
        if len(history) < 3:
            return MovementType.STEADY, 0.0, 0.0
        
        # Calculate velocity vectors and bounding box areas
        velocities = []
        areas = []
        
        # Extract centroids and frame numbers from history
        centroids = [(pos[0], pos[1]) for pos in history]
        frame_numbers = [pos[2] for pos in history]
        
        # Calculate velocities from centroids
        for i in range(1, len(centroids)):
            dx = centroids[i][0] - centroids[i-1][0]
            dy = centroids[i][1] - centroids[i-1][1]
            dt = max(1, frame_numbers[i] - frame_numbers[i-1])  # Frame difference
            velocities.append((dx/dt, dy/dt))
        
        if not velocities:
            return MovementType.STEADY, 0.0, 0.0
        
        # Calculate average velocity and its components
        avg_vx = np.mean([v[0] for v in velocities])
        avg_vy = np.mean([v[1] for v in velocities])
        avg_speed = np.sqrt(avg_vx**2 + avg_vy**2)
        
        # Calculate direction angle
        direction_angle = math.atan2(avg_vy, avg_vx) * 180 / math.pi
        
        # Analyze bounding box area changes if we have bbox data (track_crops contains bbox crops)
        area_change_ratio = 0.0
        area_change_direction = 0  # 1 for growing (approaching), -1 for shrinking (moving away)
        
        if len(self.track_crops[track_id]) >= 3:
            crops = list(self.track_crops[track_id])
            # Calculate areas for the sequence
            areas = [crop.shape[0] * crop.shape[1] for crop in crops if crop.size > 0]
            
            # Skip empty or invalid crops
            valid_areas = [area for area in areas if area > 0]
            
            if len(valid_areas) >= 3:
                # Calculate area change trend
                # Use first and last valid areas for overall trend
                first_area = valid_areas[0]
                last_area = valid_areas[-1]
                
                if first_area > 0 and last_area > 0:
                    area_change_ratio = (last_area - first_area) / first_area
                    area_change_direction = 1 if area_change_ratio > 0 else -1
        
        # Calculate distance traveled (path length)
        total_displacement = 0
        for i in range(1, len(centroids)):
            dx = centroids[i][0] - centroids[i-1][0]
            dy = centroids[i][1] - centroids[i-1][1]
            total_displacement += np.sqrt(dx**2 + dy**2)
        
        # Lower threshold for identifying walking/movement
        # Most human walking will be detected at a much lower speed threshold
        walking_threshold = 0.8  # Reduced from 2.0 for more sensitivity to walking
        
        # If the person has moved a significant distance but speed is moderate,
        # it's likely walking rather than standing steady
        if total_displacement > len(centroids) * 1.5:  # Significant movement relative to frame count
            # Determine primary movement direction
            abs_vx = abs(avg_vx)
            abs_vy = abs(avg_vy)
            
            # Significant horizontal movement (sideways)
            if abs_vx > abs_vy * 1.2:  # Primarily horizontal
                if avg_vx > 0:
                    return MovementType.RIGHT, avg_speed, direction_angle
                else:
                    return MovementType.LEFT, avg_speed, direction_angle
                    
            # Significant area change (approaching/moving away)
            elif abs(area_change_ratio) > 0.1:  # Threshold for significant area change
                if area_change_direction > 0:
                    return MovementType.FORWARD, avg_speed, direction_angle  # Growing area means approaching/forward
                else:
                    return MovementType.BACKWARD, avg_speed, direction_angle  # Shrinking area means moving away/backward
                    
            # Hybrid movement patterns or inconclusive
            elif abs_vy > abs_vx:  # More vertical than horizontal movement
                if avg_vy > 0:
                    return MovementType.FORWARD, avg_speed, direction_angle  # Downward is forward (fallback)
                else:
                    return MovementType.BACKWARD, avg_speed, direction_angle  # Upward is backward (fallback)
            else:
                # Default to the strongest direction when movement is mixed
                if abs_vx > abs_vy:
                    return MovementType.RIGHT if avg_vx > 0 else MovementType.LEFT, avg_speed, direction_angle
                else:
                    return MovementType.FORWARD if avg_vy > 0 else MovementType.BACKWARD, avg_speed, direction_angle
        elif avg_speed > walking_threshold:  # Standard velocity-based detection as backup
            # Determine primary movement direction
            abs_vx = abs(avg_vx)
            abs_vy = abs(avg_vy)
            
            # Movement classification logic:
            # 1. For LEFT/RIGHT: Use the centroid trajectory (horizontal movement)
            # 2. For FORWARD/BACKWARD: Use the bounding box area changes
            
            # Significant horizontal movement (sideways)
            if abs_vx > abs_vy * 1.2:  # Primarily horizontal
                if avg_vx > 0:
                    return MovementType.RIGHT, avg_speed, direction_angle
                else:
                    return MovementType.LEFT, avg_speed, direction_angle
                    
            # Significant area change (approaching/moving away)
            elif abs(area_change_ratio) > 0.1:  # Threshold for significant area change
                if area_change_direction > 0:
                    return MovementType.FORWARD, avg_speed, direction_angle  # Growing area means approaching/forward
                else:
                    return MovementType.BACKWARD, avg_speed, direction_angle  # Shrinking area means moving away/backward
                    
            # Hybrid movement patterns or inconclusive
            elif abs_vy > abs_vx:  # More vertical than horizontal movement
                if avg_vy > 0:
                    return MovementType.FORWARD, avg_speed, direction_angle  # Downward is forward (fallback)
                else:
                    return MovementType.BACKWARD, avg_speed, direction_angle  # Upward is backward (fallback)
            else:
                # Default to the strongest direction when movement is mixed
                if abs_vx > abs_vy:
                    return MovementType.RIGHT if avg_vx > 0 else MovementType.LEFT, avg_speed, direction_angle
                else:
                    return MovementType.FORWARD if avg_vy > 0 else MovementType.BACKWARD, avg_speed, direction_angle
        else:
            return MovementType.STEADY, avg_speed, direction_angle
    
    def _analyze_orientation(self, crop: np.ndarray, parsing_mask: np.ndarray = None) -> OrientationType:
        """
        Analyze body orientation using only human parsing data
        
        Args:
            crop: Person crop image (not used for analysis, kept for API compatibility)
            parsing_mask: Human parsing mask with body part labels
                        (0=background, 1=head, 2=body, 3=r_arm, 4=l_arm, 5=r_leg, 6=l_leg)
        
        Returns:
            OrientationType: Detected orientation
        """
        # If no parsing mask provided, default to front orientation
        if parsing_mask is None or parsing_mask.size == 0:
            return OrientationType.FRONT
            
        h, w = parsing_mask.shape
        
        # Extract body part regions
        head_mask = (parsing_mask == 1)
        body_mask = (parsing_mask == 2)
        r_arm_mask = (parsing_mask == 3)
        l_arm_mask = (parsing_mask == 4)
        r_leg_mask = (parsing_mask == 5)
        l_leg_mask = (parsing_mask == 6)
        
        # Calculate centroids and areas of body parts
        def get_centroid_and_area(mask):
            if not np.any(mask):
                return None, 0
            y_coords, x_coords = np.where(mask)
            centroid_x = np.mean(x_coords)
            centroid_y = np.mean(y_coords)
            area = len(y_coords)
            return (centroid_x, centroid_y), area
        
        head_centroid, head_area = get_centroid_and_area(head_mask)
        body_centroid, body_area = get_centroid_and_area(body_mask)
        r_arm_centroid, r_arm_area = get_centroid_and_area(r_arm_mask)
        l_arm_centroid, l_arm_area = get_centroid_and_area(l_arm_mask)
        r_leg_centroid, r_leg_area = get_centroid_and_area(r_leg_mask)
        l_leg_centroid, l_leg_area = get_centroid_and_area(l_leg_mask)
        
        # Calculate confidence scores for different features
        confidence_factors = []
        
        # 1. Arm visibility and position analysis
        total_arm_area = r_arm_area + l_arm_area
        arm_visibility_ratio = total_arm_area / max(body_area, 1)
        
        # If arms are barely visible, likely back view
        if arm_visibility_ratio < 0.1:
            return OrientationType.BACK
        
        # 2. Arm asymmetry analysis for profile detection
        arm_asymmetry = 0
        if r_arm_area > 0 and l_arm_area > 0:
            arm_asymmetry = abs(r_arm_area - l_arm_area) / max(r_arm_area + l_arm_area, 1)
            confidence_factors.append(arm_asymmetry)
        
        # 3. Body part horizontal distribution analysis
        if body_centroid and head_centroid:
            # Check if head and body are well-aligned (front/back) or offset (profile)
            head_body_offset = abs(head_centroid[0] - body_centroid[0]) / w
            confidence_factors.append(head_body_offset)
        
        # 4. Arm position relative to body center
        arm_positions = []
        if body_centroid:
            body_center_x = body_centroid[0]
            
            if r_arm_centroid and r_arm_area > 0:
                r_arm_offset = (r_arm_centroid[0] - body_center_x) / w
                arm_positions.append(('right', r_arm_offset, r_arm_area))
            
            if l_arm_centroid and l_arm_area > 0:
                l_arm_offset = (l_arm_centroid[0] - body_center_x) / w
                arm_positions.append(('left', l_arm_offset, l_arm_area))
        
        # 5. Decide orientation based on arm analysis
        if len(arm_positions) >= 2:
            # Both arms visible - likely front view or 3/4 view
            if arm_asymmetry < 0.3:  # Similar arm sizes
                return OrientationType.FRONT
            else:
                # Significant arm asymmetry - determine which side is dominant
                if r_arm_area > l_arm_area * 1.5:
                    return OrientationType.RIGHT_PROFILE
                elif l_arm_area > r_arm_area * 1.5:
                    return OrientationType.LEFT_PROFILE
                else:
                    return OrientationType.FRONT
        
        elif len(arm_positions) == 1:
            # Only one arm visible - likely profile view
            arm_side, arm_offset, arm_area = arm_positions[0]
            
            # Check which side the visible arm is on
            if arm_offset > 0.1:  # Arm is to the right of body center
                return OrientationType.RIGHT_PROFILE
            elif arm_offset < -0.1:  # Arm is to the left of body center
                return OrientationType.LEFT_PROFILE
            else:
                # Arm is centered - could be front view with one arm hidden
                return OrientationType.FRONT
        
        # 6. Fallback analysis using head-body relationship
        if head_centroid and body_centroid:
            head_body_alignment = abs(head_centroid[0] - body_centroid[0]) / w
            
            if head_body_alignment < 0.1:  # Well-aligned
                # Check if face area is reasonable (front view has more visible head)
                if head_area > body_area * 0.1:
                    return OrientationType.FRONT
                else:
                    return OrientationType.BACK
            else:
                # Head-body misalignment suggests profile
                if head_centroid[0] > body_centroid[0]:
                    return OrientationType.RIGHT_PROFILE
                else:
                    return OrientationType.LEFT_PROFILE
        
        # Default fallback
        return OrientationType.FRONT
    
    def _calculate_confidence(self, track_id: int, movement_type: MovementType, 
                            orientation_type: OrientationType) -> float:
        """Calculate confidence score based on consistency and history length"""
        history_length = len(self.track_histories[track_id])
        
        # Base confidence from history length
        base_confidence = min(history_length / self.history_length, 1.0)
        
        # Movement consistency analysis
        movement_consistency = 1.0
        if history_length >= 3:
            history = list(self.track_histories[track_id])
            velocities = []
            for i in range(1, len(history)):
                dx = history[i][0] - history[i-1][0]
                dy = history[i][1] - history[i-1][1]
                dt = max(1, history[i][2] - history[i-1][2])
                velocities.append((dx/dt, dy/dt))
            
            if velocities:
                # Calculate consistency of movement direction
                speeds = [np.sqrt(vx**2 + vy**2) for vx, vy in velocities]
                if len(speeds) > 1:
                    speed_std = np.std(speeds)
                    max_speed = max(speeds) if speeds else 0.0
                    movement_consistency = max(0.3, 1.0 - (speed_std / max(max_speed, 1.0)))
        
        # Combine confidences with improved weighting
        final_confidence = 0.6 * base_confidence + 0.4 * movement_consistency
        
        # Ensure minimum confidence for testing
        return max(final_confidence, 0.3)

class EnhancedPersonGallery:
    """Enhanced person gallery with movement and orientation profiling"""
    
    def __init__(self, max_embeddings_per_context: int = 5, 
                 similarity_threshold: float = 0.91, min_confidence: float = 0.3,
                 history_length: int = 10):
        self.max_embeddings_per_context = max_embeddings_per_context
        self.similarity_threshold = similarity_threshold
        self.min_confidence = min_confidence
        self.history_length = history_length
        
        self.gallery: Dict[str, EnhancedPersonData] = {}
        self.track_to_person: Dict[int, str] = {}
        self.person_counter = 1
        
        # Movement and orientation analyzer (using XGait sequence length for history)
        self.movement_analyzer = MovementOrientationAnalyzer(history_length=history_length)
        
        # Statistics
        self.total_embeddings_stored = 0
        self.contexts_captured = set()
        
        logger.info("✅ Enhanced Person Gallery initialized")
        logger.info(f"   Max embeddings per context: {max_embeddings_per_context}")
        logger.info(f"   Similarity threshold: {similarity_threshold}")
        logger.info(f"   Min confidence: {min_confidence}")
        logger.info(f"   Movement history length: {history_length}")
    
    def add_person_embedding(self, person_name: str, track_id: int, 
                           embedding: np.ndarray, bbox: Tuple[int, int, int, int],
                           crop: np.ndarray, frame_number: int, quality: float,
                           parsing_mask: np.ndarray = None) -> bool:
        """
        Add an embedding for a person with movement and orientation analysis
        
        Args:
            person_name: Name of the person
            track_id: Track identifier
            embedding: XGait embedding vector
            bbox: Bounding box coordinates
            crop: Person crop image
            frame_number: Current frame number
            quality: Embedding quality score
            parsing_mask: Human parsing mask (optional, for better orientation detection)
            
        Returns:
            True if embedding was added, False otherwise
        """
        # Analyze movement and orientation with enhanced parsing data
        movement_profile = self.movement_analyzer.analyze_movement_and_orientation(
            track_id, bbox, crop, frame_number, parsing_mask
        )
        
        # Check confidence threshold
        if movement_profile.confidence < self.min_confidence:
            logger.debug(f"Low confidence movement analysis for {person_name} "
                        f"(confidence: {movement_profile.confidence:.3f})")
            return False
        
        # Create person embedding
        person_embedding = PersonEmbedding(
            embedding=embedding,
            movement_profile=movement_profile,
            quality=quality,
            timestamp=datetime.now(),
            track_id=track_id,
            frame_number=frame_number,
            bounding_box=bbox
        )
        
        # Get or create person data
        if person_name not in self.gallery:
            self.gallery[person_name] = EnhancedPersonData(
                person_name=person_name,
                embeddings_by_context={},
                track_associations=[],
                creation_time=datetime.now(),
                last_update=datetime.now(),
                total_embeddings=0
            )
        
        person_data = self.gallery[person_name]
        
        # Get context key
        context_key = person_data.get_context_key(
            movement_profile.movement_type, 
            movement_profile.orientation_type
        )
        
        # Initialize context if needed
        if context_key not in person_data.embeddings_by_context:
            person_data.embeddings_by_context[context_key] = []
        
        # Add embedding to context
        context_embeddings = person_data.embeddings_by_context[context_key]
        context_embeddings.append(person_embedding)
        
        # Manage context size - keep best quality embeddings
        if len(context_embeddings) > self.max_embeddings_per_context:
            # Sort by quality (descending) and keep only the best ones
            context_embeddings.sort(key=lambda x: x.quality, reverse=True)
            # Keep only the top max_embeddings_per_context
            person_data.embeddings_by_context[context_key] = context_embeddings[:self.max_embeddings_per_context]
        
        # Update person data
        person_data.total_embeddings += 1
        person_data.last_update = datetime.now()
        if track_id not in person_data.track_associations:
            person_data.track_associations.append(track_id)
        
        # Update mappings
        self.track_to_person[track_id] = person_name
        
        # Update statistics
        self.total_embeddings_stored += 1
        self.contexts_captured.add(context_key)
        
        logger.info(f"✅ Added embedding for {person_name} - {context_key} "
                   f"(quality: {quality:.3f}, confidence: {movement_profile.confidence:.3f})")
        
        return True
    
    def identify_person(self, embedding: np.ndarray, track_id: int, 
                       bbox: Tuple[int, int, int, int], crop: np.ndarray,
                       frame_number: int, parsing_mask: np.ndarray = None) -> Tuple[Optional[str], float, MovementProfile]:
        """
        Identify a person based on their embedding and current context
        
        Args:
            embedding: XGait embedding to identify
            track_id: Track identifier
            bbox: Bounding box coordinates
            crop: Person crop image
            frame_number: Current frame number
            parsing_mask: Human parsing mask (optional, for better orientation detection)
            
        Returns:
            Tuple of (person_name, confidence, movement_profile)
        """
        # Analyze current movement and orientation with enhanced parsing data
        movement_profile = self.movement_analyzer.analyze_movement_and_orientation(
            track_id, bbox, crop, frame_number, parsing_mask
        )
        
        if movement_profile.confidence < self.min_confidence:
            return None, 0.0, movement_profile
        
        # Get context key
        context_key = f"{movement_profile.movement_type.value}_{movement_profile.orientation_type.value}"
        
        best_person = None
        best_similarity = 0.0
        
        # Compare with all persons in gallery
        for person_name, person_data in self.gallery.items():
            # Get embeddings for this specific context
            context_embeddings = person_data.embeddings_by_context.get(context_key, [])
            
            if not context_embeddings:
                # If no embeddings for this context, compare with all contexts
                context_embeddings = person_data.get_all_embeddings()
            
            # Calculate similarity with embeddings in this context
            for person_embedding in context_embeddings:
                similarity = self._cosine_similarity(embedding, person_embedding.embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_person = person_name
        
        # Check if similarity meets threshold
        if best_person and best_similarity >= self.similarity_threshold:
            return best_person, best_similarity, movement_profile
        
        return None, best_similarity, movement_profile
    
    def create_new_person(self, track_id: int, embedding: np.ndarray, 
                         bbox: Tuple[int, int, int, int], crop: np.ndarray,
                         frame_number: int, quality: float, parsing_mask: np.ndarray = None, 
                         custom_name: str = None) -> str:
        """Create a new person with automatic or custom naming"""
        if custom_name:
            person_name = custom_name
        else:
            person_name = f"Person_{self.person_counter:03d}"
            self.person_counter += 1
        
        success = self.add_person_embedding(
            person_name, track_id, embedding, bbox, crop, frame_number, quality, parsing_mask
        )
        
        if success:
            logger.info(f"🆕 Created new person: {person_name}")
            return person_name
        else:
            logger.warning(f"Failed to create new person for track {track_id}")
            return None
    
    def merge_persons(self, person1_name: str, person2_name: str) -> bool:
        """
        Merge two persons in the gallery
        
        Args:
            person1_name: Name of the first person (will be kept)
            person2_name: Name of the second person (will be merged into first)
            
        Returns:
            True if merge was successful, False otherwise
        """
        if person1_name not in self.gallery or person2_name not in self.gallery:
            logger.error(f"Cannot merge - one or both persons not found: {person1_name}, {person2_name}")
            return False
        
        if person1_name == person2_name:
            logger.warning(f"Cannot merge person with themselves: {person1_name}")
            return False
        
        person1_data = self.gallery[person1_name]
        person2_data = self.gallery[person2_name]
        
        # Merge embeddings by context
        for context_key, embeddings in person2_data.embeddings_by_context.items():
            if context_key not in person1_data.embeddings_by_context:
                person1_data.embeddings_by_context[context_key] = []
            
            # Add embeddings from person2 to person1
            person1_data.embeddings_by_context[context_key].extend(embeddings)
            
            # Trim to max size if needed
            if len(person1_data.embeddings_by_context[context_key]) > self.max_embeddings_per_context:
                # Sort by quality and keep the best ones
                person1_data.embeddings_by_context[context_key].sort(
                    key=lambda x: x.quality, reverse=True
                )
                person1_data.embeddings_by_context[context_key] = \
                    person1_data.embeddings_by_context[context_key][:self.max_embeddings_per_context]
        
        # Merge track associations
        for track_id in person2_data.track_associations:
            if track_id not in person1_data.track_associations:
                person1_data.track_associations.append(track_id)
            # Update track mapping
            self.track_to_person[track_id] = person1_name
        
        # Update metadata
        person1_data.total_embeddings += person2_data.total_embeddings
        person1_data.last_update = datetime.now()
        
        # Remove person2 from gallery
        del self.gallery[person2_name]
        
        logger.info(f"🔗 Merged {person2_name} into {person1_name}")
        return True
    
    def get_person_summary(self, person_name: str) -> Dict:
        """Get detailed summary of a person's embeddings"""
        if person_name not in self.gallery:
            return {}
        
        person_data = self.gallery[person_name]
        context_summary = person_data.get_context_summary()
        
        return {
            'person_name': person_name,
            'total_embeddings': person_data.total_embeddings,
            'contexts_captured': len(context_summary),
            'context_breakdown': context_summary,
            'track_associations': person_data.track_associations,
            'creation_time': person_data.creation_time.isoformat(),
            'last_update': person_data.last_update.isoformat()
        }
    
    def get_gallery_statistics(self) -> Dict:
        """Get comprehensive gallery statistics"""
        if not self.gallery:
            return {
                'total_persons': 0,
                'total_embeddings': 0,
                'contexts_captured': 0,
                'contexts_list': [],
                'context_popularity': {},
                'avg_embeddings_per_person': 0.0,
                'avg_contexts_per_person': 0.0
            }
        
        total_persons = len(self.gallery)
        total_embeddings = sum(person.total_embeddings for person in self.gallery.values())
        
        # Analyze context coverage
        all_contexts = set()
        context_counts = defaultdict(int)
        
        for person_data in self.gallery.values():
            for context_key in person_data.embeddings_by_context.keys():
                all_contexts.add(context_key)
                context_counts[context_key] += 1
        
        return {
            'total_persons': total_persons,
            'total_embeddings': total_embeddings,
            'contexts_captured': len(all_contexts),
            'contexts_list': list(all_contexts),
            'context_popularity': dict(context_counts),
            'avg_embeddings_per_person': total_embeddings / total_persons if total_persons > 0 else 0,
            'avg_contexts_per_person': len(all_contexts) / total_persons if total_persons > 0 else 0
        }
    
    def save_gallery(self, filepath: str) -> bool:
        """Save gallery to JSON file"""
        try:
            # Convert track_to_person keys to regular int to avoid numpy int64 serialization issues
            track_to_person_serializable = {int(k): v for k, v in self.track_to_person.items()}
            
            data = {
                'gallery': {},
                'track_to_person': track_to_person_serializable,
                'person_counter': self.person_counter,
                'statistics': {
                    'total_embeddings_stored': self.total_embeddings_stored,
                    'contexts_captured': list(self.contexts_captured)
                },
                'metadata': {
                    'max_embeddings_per_context': self.max_embeddings_per_context,
                    'similarity_threshold': self.similarity_threshold,
                    'min_confidence': self.min_confidence,
                    'save_timestamp': datetime.now().isoformat()
                }
            }
            
            # Convert person data to serializable format
            for person_name, person_data in self.gallery.items():
                data['gallery'][person_name] = {
                    'person_name': person_data.person_name,
                    'embeddings_by_context': {
                        context: [emb.to_dict() for emb in embeddings]
                        for context, embeddings in person_data.embeddings_by_context.items()
                    },
                    'track_associations': [int(track_id) for track_id in person_data.track_associations],  # Convert to regular int
                    'creation_time': person_data.creation_time.isoformat(),
                    'last_update': person_data.last_update.isoformat(),
                    'total_embeddings': person_data.total_embeddings
                }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"✅ Enhanced gallery saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving enhanced gallery: {e}")
            return False
    
    def load_gallery(self, filepath: str, clear_track_associations: bool = True) -> bool:
        """
        Load gallery from JSON file
        
        Args:
            filepath: Path to the gallery JSON file
            clear_track_associations: If True, clears track associations to ensure 
                                    track independence between videos
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Load person data
            self.gallery = {}
            for person_name, person_dict in data['gallery'].items():
                embeddings_by_context = {}
                for context, embedding_dicts in person_dict['embeddings_by_context'].items():
                    embeddings_by_context[context] = [
                        PersonEmbedding.from_dict(emb_dict) for emb_dict in embedding_dicts
                    ]
                
                self.gallery[person_name] = EnhancedPersonData(
                    person_name=person_dict['person_name'],
                    embeddings_by_context=embeddings_by_context,
                    track_associations=[] if clear_track_associations else person_dict['track_associations'],
                    creation_time=datetime.fromisoformat(person_dict['creation_time']),
                    last_update=datetime.fromisoformat(person_dict['last_update']),
                    total_embeddings=person_dict['total_embeddings']
                )
            
            # Load mappings and metadata
            if clear_track_associations:
                # Clear track associations to ensure track independence between videos
                self.track_to_person = {}
                logger.info(f"🔄 Cleared track associations for enhanced gallery (track independence)")
            else:
                # Convert track_to_person keys from string (JSON) back to int
                self.track_to_person = {int(k): v for k, v in data['track_to_person'].items()}
            
            self.person_counter = data['person_counter']
            
            if 'statistics' in data:
                self.total_embeddings_stored = data['statistics']['total_embeddings_stored']
                self.contexts_captured = set(data['statistics']['contexts_captured'])
            
            if 'metadata' in data:
                metadata = data['metadata']
                self.max_embeddings_per_context = metadata.get('max_embeddings_per_context', 5)
                self.similarity_threshold = metadata.get('similarity_threshold', 0.7)
                self.min_confidence = metadata.get('min_confidence', 0.6)
            
            logger.info(f"✅ Enhanced gallery loaded from {filepath}")
            logger.info(f"   Loaded {len(self.gallery)} persons with {len(self.contexts_captured)} contexts")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading enhanced gallery: {e}")
            return False
    
    def print_gallery_report(self) -> None:
        """Print comprehensive gallery report"""
        stats = self.get_gallery_statistics()
        
        print("\n" + "="*60)
        print("ENHANCED PERSON GALLERY REPORT")
        print("="*60)
        
        print(f"Total Persons: {stats['total_persons']}")
        print(f"Total Embeddings: {stats['total_embeddings']}")
        print(f"Contexts Captured: {stats['contexts_captured']}")
        print(f"Average Embeddings per Person: {stats['avg_embeddings_per_person']:.1f}")
        print(f"Average Contexts per Person: {stats['avg_contexts_per_person']:.1f}")
        
        print("\nContext Breakdown:")
        for context, count in sorted(stats['context_popularity'].items()):
            print(f"  {context}: {count} persons")
        
        print("\nPerson Details:")
        for person_name in sorted(self.gallery.keys()):
            summary = self.get_person_summary(person_name)
            print(f"\n{person_name}:")
            print(f"  Total embeddings: {summary['total_embeddings']}")
            print(f"  Contexts: {summary['contexts_captured']}")
            print(f"  Tracks: {len(summary['track_associations'])}")
            for context, count in summary['context_breakdown'].items():
                print(f"    {context}: {count}")
        
        print("\n" + "="*60)
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        if a.size == 0 or b.size == 0:
            return 0.0
        
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)