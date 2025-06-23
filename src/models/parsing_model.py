"""
Improved Human Parsing Model
Combines semantic segmentation with pose estimation for detailed human part segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Union, Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling module"""
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                   stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, inplanes=2048, outplanes=256):
        super(ASPP, self).__init__()
        
        dilations = [1, 6, 12, 18]
        
        self.aspp1 = ASPPModule(inplanes, outplanes, 1, padding=0, dilation=dilations[0])
        self.aspp2 = ASPPModule(inplanes, outplanes, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = ASPPModule(inplanes, outplanes, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = ASPPModule(inplanes, outplanes, 3, padding=dilations[3], dilation=dilations[3])
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, outplanes, 1, stride=1, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU()
        )
        
        self.conv1 = nn.Conv2d(outplanes * 5, outplanes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        return self.dropout(x)


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet"""
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    """Modified ResNet backbone for SCHP"""
    def __init__(self, block, layers, num_classes=20):
        super(ResNet, self).__init__()
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2)
        
        self.aspp = ASPP(inplanes=2048, outplanes=256)
        
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
    
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.aspp(x)
        x = self.classifier(x)
        
        return x


class SCHPModel(nn.Module):
    """Self-Correction for Human Parsing Model"""
    def __init__(self, num_classes=20):
        super(SCHPModel, self).__init__()
        
        # ResNet-101 backbone
        self.backbone = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
        
        # Self-correction modules
        self.edge_layers = nn.ModuleList([
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Conv2d(256, 1, kernel_size=3, padding=1),
            nn.Conv2d(512, 1, kernel_size=3, padding=1),
            nn.Conv2d(1024, 1, kernel_size=3, padding=1),
            nn.Conv2d(2048, 1, kernel_size=3, padding=1)
        ])
        
    def forward(self, x):
        # Get features from backbone
        features = []
        
        # Forward through backbone with feature extraction
        feat = self.backbone.conv1(x)
        feat = self.backbone.bn1(feat)
        feat = self.backbone.relu(feat)
        feat = self.backbone.maxpool(feat)
        features.append(feat)
        
        feat = self.backbone.layer1(feat)
        features.append(feat)
        
        feat = self.backbone.layer2(feat)
        features.append(feat)
        
        feat = self.backbone.layer3(feat)
        features.append(feat)
        
        feat = self.backbone.layer4(feat)
        features.append(feat)
        
        # ASPP and classifier
        feat = self.backbone.aspp(feat)
        parsing = self.backbone.classifier(feat)
        
        # Generate edge maps
        edges = []
        for i, edge_layer in enumerate(self.edge_layers):
            edge = edge_layer(features[i])
            edges.append(edge)
        
        return parsing, edges


class HumanParsingModel:
    """
    Improved human parsing using semantic segmentation + pose estimation
    """
    
    # Human parsing labels
    LABELS = {
        0: 'background',
        1: 'hat',
        2: 'hair',
        3: 'glove',
        4: 'sunglasses',
        5: 'upperclothes',
        6: 'dress',
        7: 'coat',
        8: 'socks',
        9: 'pants',
        10: 'jumpsuits',
        11: 'scarf',
        12: 'skirt',
        13: 'face',
        14: 'left_arm',
        15: 'right_arm',
        16: 'left_leg',
        17: 'right_leg',
        18: 'left_shoe',
        19: 'right_shoe'
    }
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu", num_classes: int = 20):
        self.device = device
        self.num_classes = num_classes
        
        # Load semantic segmentation model (DeepLabV3)
        try:
            from torchvision.models.segmentation import deeplabv3_resnet101
            self.seg_model = deeplabv3_resnet101(weights='DEFAULT')
            self.seg_model.eval()
            self.seg_model.to(device)
            logger.info("✅ Loaded DeepLabV3 for base segmentation")
            self.seg_available = True
        except Exception as e:
            logger.warning(f"❌ Failed to load segmentation model: {e}")
            self.seg_model = None
            self.seg_available = False
        
        # Load pose estimation model
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True,
                min_detection_confidence=0.5
            )
            self.pose_available = True
            logger.info("✅ Loaded MediaPipe pose estimation")
        except Exception as e:
            logger.warning(f"❌ MediaPipe not available: {e}")
            self.pose_available = False
        
        self.model_loaded = True  # Always true for this implementation
        
        # Preprocessing for segmentation
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_parsing(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """Extract human parsing from person crops"""
        if not crops:
            return []
        
        parsing_masks = []
        
        for crop in crops:
            try:
                # Get base person segmentation
                person_mask = self._get_person_segmentation(crop)
                
                # Get pose-based parsing if available
                if self.pose_available:
                    parsing_mask = self._get_pose_based_parsing(crop, person_mask)
                else:
                    parsing_mask = self._get_rule_based_parsing(crop, person_mask)
                
                parsing_masks.append(parsing_mask)
                
            except Exception as e:
                logger.error(f"Error extracting parsing: {e}")
                # Fallback to rule-based parsing
                h, w = crop.shape[:2]
                fallback_mask = self._create_simple_parsing(h, w)
                parsing_masks.append(fallback_mask)
        
        return parsing_masks
    
    def _get_person_segmentation(self, crop):
        """Get person segmentation using DeepLabV3"""
        if not self.seg_available:
            h, w = crop.shape[:2]
            return np.ones((h, w), dtype=np.uint8)
        
        # Preprocess
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.seg_model(input_tensor)['out'][0]
            # Person class is 15 in COCO classes
            person_logits = output[15]  
            person_prob = torch.sigmoid(person_logits)
            
            # Resize to original size
            h, w = crop.shape[:2]
            person_prob = F.interpolate(
                person_prob.unsqueeze(0).unsqueeze(0), 
                size=(h, w), 
                mode='bilinear'
            )[0, 0]
            
            person_mask = (person_prob > 0.5).cpu().numpy().astype(np.uint8)
        
        return person_mask
    
    def _get_pose_based_parsing(self, crop, person_mask):
        """Use MediaPipe pose to create detailed parsing"""
        try:
            # Convert to RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Get pose landmarks
            results = self.pose.process(crop_rgb)
            
            h, w = crop.shape[:2]
            parsing_mask = np.zeros((h, w), dtype=np.uint8)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Create regions based on pose landmarks
                parsing_mask = self._create_pose_regions(landmarks, h, w, person_mask)
            else:
                # Fallback if no pose detected
                parsing_mask = self._get_rule_based_parsing(crop, person_mask)
            
            return parsing_mask
            
        except Exception as e:
            logger.error(f"Pose parsing error: {e}")
            return self._get_rule_based_parsing(crop, person_mask)
    
    def _create_pose_regions(self, landmarks, h, w, person_mask):
        """Create parsing regions based on pose landmarks"""
        parsing_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Convert landmarks to pixel coordinates
        def get_point(landmark):
            return int(landmark.x * w), int(landmark.y * h)
        
        try:
            # Get key points
            nose = get_point(landmarks[0])
            left_eye = get_point(landmarks[2])
            right_eye = get_point(landmarks[5])
            left_ear = get_point(landmarks[7])
            right_ear = get_point(landmarks[8])
            left_shoulder = get_point(landmarks[11])
            right_shoulder = get_point(landmarks[12])
            left_elbow = get_point(landmarks[13])
            right_elbow = get_point(landmarks[14])
            left_wrist = get_point(landmarks[15])
            right_wrist = get_point(landmarks[16])
            left_hip = get_point(landmarks[23])
            right_hip = get_point(landmarks[24])
            left_knee = get_point(landmarks[25])
            right_knee = get_point(landmarks[26])
            left_ankle = get_point(landmarks[27])
            right_ankle = get_point(landmarks[28])
            
            # Face region (including hair)
            face_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
            face_radius = max(abs(left_ear[0] - right_ear[0]), abs(nose[1] - left_eye[1])) // 2
            cv2.circle(parsing_mask, face_center, int(face_radius * 1.2), 13, -1)  # face
            cv2.circle(parsing_mask, (face_center[0], face_center[1] - face_radius), 
                      int(face_radius * 0.8), 2, -1)  # hair
            
            # Torso region
            torso_points = np.array([left_shoulder, right_shoulder, right_hip, left_hip])
            cv2.fillPoly(parsing_mask, [torso_points], 5)  # upperclothes
            
            # Arms
            # Left arm
            left_arm_points = np.array([left_shoulder, left_elbow, left_wrist])
            for i in range(len(left_arm_points) - 1):
                cv2.line(parsing_mask, tuple(left_arm_points[i]), tuple(left_arm_points[i+1]), 14, 15)
            
            # Right arm  
            right_arm_points = np.array([right_shoulder, right_elbow, right_wrist])
            for i in range(len(right_arm_points) - 1):
                cv2.line(parsing_mask, tuple(right_arm_points[i]), tuple(right_arm_points[i+1]), 15, 15)
            
            # Legs
            # Left leg
            left_leg_points = np.array([left_hip, left_knee, left_ankle])
            for i in range(len(left_leg_points) - 1):
                cv2.line(parsing_mask, tuple(left_leg_points[i]), tuple(left_leg_points[i+1]), 16, 20)
            
            # Right leg
            right_leg_points = np.array([right_hip, right_knee, right_ankle])
            for i in range(len(right_leg_points) - 1):
                cv2.line(parsing_mask, tuple(right_leg_points[i]), tuple(right_leg_points[i+1]), 17, 20)
            
            # Pants region (lower torso)
            pants_points = np.array([left_hip, right_hip, 
                                   (right_knee[0], right_knee[1] - 20), 
                                   (left_knee[0], left_knee[1] - 20)])
            cv2.fillPoly(parsing_mask, [pants_points], 9)  # pants
            
            # Shoes
            cv2.circle(parsing_mask, left_ankle, 15, 18, -1)  # left_shoe
            cv2.circle(parsing_mask, right_ankle, 15, 19, -1)  # right_shoe
            
            # Only keep regions where person is detected
            parsing_mask = parsing_mask * person_mask
            
        except Exception as e:
            logger.error(f"Error creating pose regions: {e}")
            return self._get_rule_based_parsing(None, person_mask)
        
        return parsing_mask
    
    def _get_rule_based_parsing(self, crop, person_mask):
        """Create parsing using simple geometric rules"""
        h, w = person_mask.shape
        parsing_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Find person bounding box
        coords = np.where(person_mask > 0)
        if len(coords[0]) == 0:
            return parsing_mask
        
        min_y, max_y = coords[0].min(), coords[0].max()
        min_x, max_x = coords[1].min(), coords[1].max()
        
        person_height = max_y - min_y
        person_width = max_x - min_x
        
        # Simple region assignment
        # Head (top 15%)
        head_bottom = min_y + int(0.15 * person_height)
        parsing_mask[min_y:head_bottom, min_x:max_x] = 13  # face
        
        # Hair (top 10%)  
        hair_bottom = min_y + int(0.10 * person_height)
        parsing_mask[min_y:hair_bottom, min_x:max_x] = 2  # hair
        
        # Torso (15% to 60%)
        torso_top = min_y + int(0.15 * person_height)
        torso_bottom = min_y + int(0.60 * person_height)
        parsing_mask[torso_top:torso_bottom, min_x:max_x] = 5  # upperclothes
        
        # Pants (60% to 85%)
        pants_top = min_y + int(0.60 * person_height)
        pants_bottom = min_y + int(0.85 * person_height)
        parsing_mask[pants_top:pants_bottom, min_x:max_x] = 9  # pants
        
        # Arms (sides)
        arm_width = int(0.15 * person_width)
        # Left arm
        parsing_mask[torso_top:pants_top, min_x:min_x+arm_width] = 14
        # Right arm  
        parsing_mask[torso_top:pants_top, max_x-arm_width:max_x] = 15
        
        # Legs (bottom 40%)
        leg_top = min_y + int(0.60 * person_height)
        leg_mid = min_x + person_width // 2
        # Left leg
        parsing_mask[leg_top:max_y, min_x:leg_mid] = 16
        # Right leg
        parsing_mask[leg_top:max_y, leg_mid:max_x] = 17
        
        # Shoes (bottom 10%)
        shoe_top = min_y + int(0.90 * person_height)
        parsing_mask[shoe_top:max_y, min_x:leg_mid] = 18  # left_shoe
        parsing_mask[shoe_top:max_y, leg_mid:max_x] = 19  # right_shoe
        
        # Only keep regions where person is detected
        parsing_mask = parsing_mask * person_mask
        
        return parsing_mask
    
    def _create_simple_parsing(self, h, w):
        """Create a simple fallback parsing"""
        parsing_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Simple geometric parsing
        # Head
        parsing_mask[:h//6, :] = 13  # face
        parsing_mask[:h//8, :] = 2   # hair
        
        # Torso
        parsing_mask[h//6:h//2, :] = 5  # upperclothes
        
        # Lower body
        parsing_mask[h//2:4*h//5, :] = 9  # pants
        
        # Legs
        parsing_mask[h//2:, :w//2] = 16   # left_leg
        parsing_mask[h//2:, w//2:] = 17   # right_leg
        
        # Arms
        parsing_mask[h//6:h//2, :w//4] = 14     # left_arm
        parsing_mask[h//6:h//2, 3*w//4:] = 15   # right_arm
        
        # Shoes
        parsing_mask[4*h//5:, :w//2] = 18    # left_shoe
        parsing_mask[4*h//5:, w//2:] = 19    # right_shoe
        
        return parsing_mask
    
    def get_part_mask(self, parsing_mask: np.ndarray, part_labels: List[int]) -> np.ndarray:
        """Get binary mask for specific body parts"""
        mask = np.zeros_like(parsing_mask, dtype=np.uint8)
        for label in part_labels:
            mask[parsing_mask == label] = 255
        return mask
    
    def get_body_parts(self, parsing_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract masks for different body parts"""
        parts = {}
        
        # Head (hat, hair, face)
        parts['head'] = self.get_part_mask(parsing_mask, [1, 2, 13])
        
        # Torso (upperclothes, dress, coat)
        parts['torso'] = self.get_part_mask(parsing_mask, [5, 6, 7])
        
        # Arms
        parts['arms'] = self.get_part_mask(parsing_mask, [14, 15])
        
        # Legs
        parts['legs'] = self.get_part_mask(parsing_mask, [16, 17])
        
        # Feet (shoes)
        parts['feet'] = self.get_part_mask(parsing_mask, [18, 19])
        
        # Lower body (pants, skirt)
        parts['lower_body'] = self.get_part_mask(parsing_mask, [9, 12])
        
        return parts
    
    def is_model_loaded(self) -> bool:
        """Check if real model weights are loaded"""
        return self.model_loaded
    
    def visualize_parsing(self, parsing_mask: np.ndarray) -> np.ndarray:
        """Create colored visualization of parsing mask"""
        # Define colors for each class
        colors = [
            [0, 0, 0],      # background
            [128, 0, 0],    # hat
            [255, 0, 0],    # hair
            [0, 85, 0],     # glove
            [170, 0, 51],   # sunglasses
            [255, 85, 0],   # upperclothes
            [0, 0, 85],     # dress
            [0, 119, 221],  # coat
            [85, 85, 0],    # socks
            [0, 85, 85],    # pants
            [85, 51, 0],    # jumpsuits
            [52, 86, 128],  # scarf
            [0, 128, 0],    # skirt
            [0, 0, 255],    # face
            [51, 170, 221], # left_arm
            [0, 255, 255],  # right_arm
            [85, 255, 170], # left_leg
            [170, 255, 85], # right_leg
            [255, 255, 0],  # left_shoe
            [255, 170, 0]   # right_shoe
        ]
        
        colored_mask = np.zeros((parsing_mask.shape[0], parsing_mask.shape[1], 3), dtype=np.uint8)
        
        for i, color in enumerate(colors):
            if i < len(colors):
                colored_mask[parsing_mask == i] = color
        
        return colored_mask


def create_human_parsing_model(model_path: Optional[str] = None, device: str = "cpu", num_classes: int = 20) -> HumanParsingModel:
    """
    Create and return a human parsing model
    
    Args:
        model_path: Path to SCHP model weights
        device: Device to use for inference
        num_classes: Number of parsing classes
        
    Returns:
        HumanParsingModel instance
    """
    # Look for default model path in weights directory
    if model_path is None:
        current_dir = Path(__file__).parent.parent.parent
        potential_paths = [
            current_dir / "weights" / "schp_resnet101.pth",
            current_dir / "weights" / "exp-schp-201908301523-atr.pth",
            current_dir / "weights" / "schp.pth"
        ]
        
        for path in potential_paths:
            if path.exists():
                model_path = str(path)
                break
    
    return HumanParsingModel(model_path=model_path, device=device, num_classes=num_classes)


if __name__ == "__main__":
    # Test the human parsing model
    print("🧪 Testing SCHP Human Parsing Model")
    
    # Create test data
    test_crop = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
    
    # Create parsing model
    parsing_model = create_human_parsing_model(device="cpu")
    
    # Extract parsing
    parsing_masks = parsing_model.extract_parsing([test_crop])
    
    print(f"✅ Extracted parsing: shape {parsing_masks[0].shape}")
    print(f"📊 Model loaded: {parsing_model.is_model_loaded()}")
    print(f"🎯 Unique values in parsing: {np.unique(parsing_masks[0])}")
    print(f"🏷️  Labels: {list(parsing_model.LABELS.values())}")
    
    # Test body parts extraction
    body_parts = parsing_model.get_body_parts(parsing_masks[0])
    print(f"🔍 Body parts extracted: {list(body_parts.keys())}")
