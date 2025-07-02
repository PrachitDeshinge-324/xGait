import torch
import torchvision.transforms as T
import cv2
import numpy as np
from ultralytics import YOLO
import sys
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F

# Add src directory to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from models.silhouette_model import SilhouetteExtractor
from models.isnet import ISNetDIS

# ===== Silhouette Extractors =====
# IS-Net, U²-Net, and YOLO-Seg implementations are included

class SilhouetteExtractorISNet:
    """
    IS-Net (Interactive Segmentation Network) implementation
    More accurate and modern approach for salient object detection
    """
    def __init__(self, device=None, input_size=1024):
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        # IS-Net uses higher resolution for better accuracy
        self.input_size = input_size
        
        # Load IS-Net model (we'll use a custom implementation)
        self.model = self._build_isnet_model()
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.input_size, self.input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def _build_isnet_model(self):
        """Build IS-Net architecture"""
        try:
            # Use the official IS-Net DIS model
            model = ISNetDIS(in_ch=3, out_ch=1)
            
            # Try to load pre-trained weights if available
            weight_path = "weights/isnet.pth"
            if Path(weight_path).exists():
                model.load_state_dict(torch.load(weight_path, map_location='cpu'))
                print("✅ Loaded IS-Net pre-trained weights")
            else:
                print("⚠️ No IS-Net weights found, using random initialization")
                
            return model
            
        except Exception as e:
            print(f"⚠️ Failed to load IS-Net: {e}")
            # Return a simplified version for fallback
            return self._build_simple_fallback()

    def _build_simple_fallback(self):
        """Simple fallback model"""
        class SimpleFallback(nn.Module):
            def __init__(self):
                super(SimpleFallback, self).__init__()
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                self.conv2 = nn.Conv2d(64, 1, 3, padding=1)
                self.relu = nn.ReLU(inplace=True)
                
            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = torch.sigmoid(self.conv2(x))
                return [x]  # Return as list to match IS-Net output format
                
        return SimpleFallback()

    def extract_silhouette(self, image: np.ndarray) -> np.ndarray:
        original_h, original_w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(image_rgb).to(self.device)

        with torch.no_grad():
            # IS-Net outputs list of predictions and features
            output = self.model(input_tensor.unsqueeze(0))
            
            # Handle IS-Net output format: [predictions, features]
            if isinstance(output, tuple) and len(output) == 2:
                predictions, _ = output
                pred = predictions[0]  # Use the first (main) prediction
            elif isinstance(output, list):
                pred = output[0]  # Use the first prediction
            else:
                pred = output
                
            # IS-Net predictions are already sigmoid activated
            if pred.max() <= 1.0:
                mask_np = pred.squeeze().cpu().numpy()
                mask_np = (mask_np * 255).astype(np.uint8)
            else:
                pred = torch.sigmoid(pred)
                mask_np = pred.squeeze().cpu().numpy()
                mask_np = (mask_np * 255).astype(np.uint8)
            
            # Resize back to original dimensions
            mask_resized = cv2.resize(mask_np, (original_w, original_h), 
                                    interpolation=cv2.INTER_CUBIC)
            return mask_resized

    def extract_silhouette_batch(self, images: list) -> list:
        """Process multiple images in a batch with IS-Net"""
        if not images:
            return []
            
        batch_tensors = []
        original_sizes = []
        valid_indices = []
        
        for i, image in enumerate(images):
            # Validate image
            if image is None or image.size == 0 or len(image.shape) < 2:
                original_sizes.append((1, 1))  # Minimum size for invalid images
                batch_tensors.append(None)
                continue
                
            original_h, original_w = image.shape[:2]
            
            # Skip images with invalid dimensions
            if original_h <= 0 or original_w <= 0 or original_h < 10 or original_w < 10:
                original_sizes.append((max(1, original_w), max(1, original_h)))
                batch_tensors.append(None)
                continue
                
            try:
                original_sizes.append((original_w, original_h))
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                input_tensor = self.transform(image_rgb)
                batch_tensors.append(input_tensor)
                valid_indices.append(i)
            except Exception as e:
                print(f"⚠️ Error processing image {i}: {e}")
                original_sizes.append((max(1, original_w), max(1, original_h)))
                batch_tensors.append(None)
        
        # Process only valid images
        valid_tensors = [tensor for tensor in batch_tensors if tensor is not None]
        
        if not valid_tensors:
            # Return empty masks for all images
            return [np.zeros((h, w), dtype=np.uint8) for w, h in original_sizes]
        
        try:
            batch_tensor = torch.stack(valid_tensors).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                
                # Handle IS-Net output format: [predictions, features]
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    predictions, _ = outputs
                    # predictions is a list of outputs from different scales
                    main_predictions = predictions[0]  # Use the first (main) prediction
                elif isinstance(outputs, list):
                    main_predictions = outputs[0]  # Use the first prediction
                else:
                    main_predictions = outputs
                    
                valid_masks = []
                
                for i in range(main_predictions.shape[0]):  # Iterate over batch
                    pred = main_predictions[i]
                    
                    # IS-Net predictions are already sigmoid activated
                    if pred.max() <= 1.0:
                        mask_np = pred.squeeze().cpu().numpy()
                        mask_np = (mask_np * 255).astype(np.uint8)
                    else:
                        pred = torch.sigmoid(pred)
                        mask_np = pred.squeeze().cpu().numpy()
                        mask_np = (mask_np * 255).astype(np.uint8)
                    
                    valid_masks.append(mask_np)
                    
        except Exception as e:
            print(f"⚠️ Error in IS-Net batch inference: {e}")
            # Return empty masks for all images
            return [np.zeros((h, w), dtype=np.uint8) for w, h in original_sizes]
        
        # Reconstruct results for all images
        results = []
        valid_idx = 0
        
        for i, (original_w, original_h) in enumerate(original_sizes):
            if i in valid_indices and valid_idx < len(valid_masks):
                # Resize valid mask
                try:
                    mask_resized = cv2.resize(valid_masks[valid_idx], (original_w, original_h), 
                                            interpolation=cv2.INTER_CUBIC)
                    results.append(mask_resized)
                except Exception as e:
                    print(f"⚠️ Error resizing mask {i}: {e}")
                    results.append(np.zeros((original_h, original_w), dtype=np.uint8))
                valid_idx += 1
            else:
                # Return empty mask for invalid images
                results.append(np.zeros((original_h, original_w), dtype=np.uint8))
                
        return results

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.extract_silhouette(image)


class SilhouetteExtractorYOLOSeg:
    """
    YOLO Segmentation implementation for person silhouette extraction
    Uses YOLO's instance segmentation capabilities to extract person masks
    """
    def __init__(self, device=None, model_path=None):
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        # Find available YOLO model
        if model_path is None:
            # Try to find YOLO segmentation model (preferred) or use auto-download
            possible_paths = [
                "weights/yolo11m-seg.pt",  # Preferred segmentation model
                "weights/yolo11s-seg.pt",  # Smaller segmentation model
                "weights/yolo11n-seg.pt",  # Nano segmentation model
                "yolo11m-seg.pt",          # Auto-download medium segmentation model
                "yolo11s-seg.pt",          # Auto-download small segmentation model
                "weights/yolo11m.pt",      # Detection model fallback
                "weights/yolov8n.pt"       # Old detection model fallback
            ]
            
            model_path = None
            for path in possible_paths:
                if Path(path).exists() or not path.startswith("weights/"):
                    model_path = path
                    break
                    
            if model_path is None:
                # Default to auto-download
                model_path = "yolo11m-seg.pt"
                print("📥 Will auto-download YOLO11m segmentation model")
        
        try:
            self.model = YOLO(model_path)
            print(f"✅ Loaded YOLO model from: {model_path}")
            
            # Check if the model supports segmentation
            # YOLO models with segmentation end with 'seg' or have segmentation tasks
            self.has_segmentation = self._check_segmentation_support()
            
        except Exception as e:
            print(f"⚠️ Failed to load YOLO model: {e}")
            raise

    def _check_segmentation_support(self):
        """Check if the loaded YOLO model supports segmentation"""
        try:
            # Check model name first - most reliable method
            model_name = str(self.model.ckpt_path if hasattr(self.model, 'ckpt_path') else "")
            if 'seg' in model_name.lower():
                print("✅ YOLO model supports segmentation (detected from filename)")
                return True
            
            # Try a test prediction to see if segmentation is available
            test_image = np.zeros((640, 640, 3), dtype=np.uint8)
            results = self.model.predict(test_image, verbose=False)
            
            # Check if the result object has the masks attribute, even if empty
            if hasattr(results[0], 'masks'):
                print("✅ YOLO model supports segmentation (detected from prediction)")
                return True
            else:
                print("⚠️ YOLO model does not support segmentation, using bounding boxes")
                return False
                
        except Exception as e:
            print(f"⚠️ Could not determine YOLO segmentation support ({e}), using bounding boxes")
            return False

    def extract_silhouette(self, image: np.ndarray) -> np.ndarray:
        """Extract silhouette using YOLO segmentation"""
        original_h, original_w = image.shape[:2]
        mask = np.zeros((original_h, original_w), dtype=np.uint8)
        
        try:
            # Run YOLO prediction
            results = self.model.predict(image, verbose=False, device=self.device)
            
            if not results or len(results) == 0:
                return mask
                
            result = results[0]
            
            # Check if we have detection results
            if result.boxes is None or len(result.boxes) == 0:
                return mask
            
            # Filter for person class (class 0 in COCO dataset)
            person_indices = []
            for i, cls in enumerate(result.boxes.cls):
                if int(cls) == 0:  # Person class
                    person_indices.append(i)
            
            if not person_indices:
                return mask
            
            if self.has_segmentation and result.masks is not None:
                # Use segmentation masks
                for idx in person_indices:
                    if idx < len(result.masks.data):
                        # Get mask data
                        mask_data = result.masks.data[idx].cpu().numpy()
                        
                        # Resize mask to original image size
                        mask_resized = cv2.resize(mask_data, (original_w, original_h), 
                                                interpolation=cv2.INTER_NEAREST)
                        
                        # Convert to binary mask
                        binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
                        
                        # Combine with existing mask
                        mask = np.maximum(mask, binary_mask)
            else:
                # Fallback: use bounding boxes to create rectangular masks
                for idx in person_indices:
                    if idx < len(result.boxes.xyxy):
                        x1, y1, x2, y2 = result.boxes.xyxy[idx].cpu().numpy().astype(int)
                        
                        # Ensure coordinates are within image bounds
                        x1 = max(0, min(x1, original_w - 1))
                        y1 = max(0, min(y1, original_h - 1))
                        x2 = max(x1 + 1, min(x2, original_w))
                        y2 = max(y1 + 1, min(y2, original_h))
                        
                        # Create rectangular mask
                        mask[y1:y2, x1:x2] = 255
                        
        except Exception as e:
            print(f"⚠️ Error in YOLO segmentation: {e}")
            return mask
            
        return mask

    def extract_silhouette_batch(self, images: list) -> list:
        """Process multiple images in a batch with YOLO"""
        if not images:
            return []
            
        results = []
        
        for i, image in enumerate(images):
            # Validate image
            if image is None or image.size == 0 or len(image.shape) < 2:
                original_h, original_w = 1, 1  # Minimum size for invalid images
                results.append(np.zeros((original_h, original_w), dtype=np.uint8))
                continue
                
            original_h, original_w = image.shape[:2]
            
            # Skip images with invalid dimensions
            if original_h <= 0 or original_w <= 0 or original_h < 10 or original_w < 10:
                results.append(np.zeros((max(1, original_h), max(1, original_w)), dtype=np.uint8))
                continue
                
            try:
                mask = self.extract_silhouette(image)
                results.append(mask)
            except Exception as e:
                print(f"⚠️ Error processing image {i} in YOLO batch: {e}")
                results.append(np.zeros((original_h, original_w), dtype=np.uint8))
        
        return results

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.extract_silhouette(image)
def process_video_comparison(video_path, output_path=None, show_preview=True, 
                            frame_skip=2, batch_size=4, max_frames=None):
    """
    Compare IS-Net and U²-Net silhouette extraction side by side
    
    Args:
        video_path: Path to input video
        output_path: Path to output video (optional)
        show_preview: Whether to show preview window
        frame_skip: Process every Nth frame (2 = every other frame)
        batch_size: Number of person crops to process in one batch
        max_frames: Maximum frames to process (None for all)
    """
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📹 Processing video: {total_frames} frames at {fps} FPS")
    print(f"⚡ Optimizations: frame_skip={frame_skip}, batch_size={batch_size}")
    print(f"🔍 Comparing IS-Net vs U²-Net models")

    # Optional output writer for comparison video
    if output_path:
        # Create side-by-side comparison video (width * 2)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * 2, height), isColor=False)
    else:
        out = None

    # Load models
    print("🔄 Loading models...")
    yolo_model = YOLO("../../weights/yolo11m.pt")
    isnet_model = SilhouetteExtractorISNet(input_size=256)
    
    print("🔄 Loading U²-Net model...")
    try:
        u2net_model = SilhouetteExtractor()
        print(f"✅ U²-Net model loaded: {u2net_model.get_model_info()}")
    except Exception as e:
        print(f"❌ Failed to load U²-Net model: {e}")
        return
    
    frame_count = 0
    processed_count = 0
    prev_mask_isnet = np.zeros((height, width), dtype=np.uint8)
    prev_mask_u2net = np.zeros((height, width), dtype=np.uint8)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Skip frames for speed
        if frame_count % frame_skip != 0:
            if out:
                # For skipped frames, use previous masks
                comparison_frame = np.hstack([prev_mask_isnet, prev_mask_u2net])
                out.write(comparison_frame)
            continue
            
        # Stop if max frames reached
        if max_frames and processed_count >= max_frames:
            break
            
        processed_count += 1

        # YOLO inference for person detection
        results = yolo_model(frame, verbose=False)[0]
        
        # Initialize masks
        isnet_mask_full = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        u2net_mask_full = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        # Collect all person crops for batch processing
        person_crops = []
        crop_positions = []
        
        for box in results.boxes:
            cls_id = int(box.cls.item())
            if cls_id != 0:  # Only person class (class 0 in COCO)
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            # Padding
            pad = 10
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(frame.shape[1], x2 + pad)
            y2 = min(frame.shape[0], y2 + pad)

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            person_crops.append(person_crop)
            crop_positions.append((x1, y1, x2, y2))

        # Process crops with both models
        if person_crops:
            # IS-Net batch processing
            for i in range(0, len(person_crops), batch_size):
                batch_crops = person_crops[i:i+batch_size]
                batch_positions = crop_positions[i:i+batch_size]
                
                # IS-Net processing
                isnet_masks = isnet_model.extract_silhouette_batch(batch_crops)
                
                # U²-Net processing
                u2net_masks = u2net_model(batch_crops)
                
                # Place IS-Net masks in full frame
                for mask, (x1, y1, x2, y2) in zip(isnet_masks, batch_positions):
                    isnet_mask_full[y1:y2, x1:x2] = np.maximum(
                        isnet_mask_full[y1:y2, x1:x2], mask)
                
                # Place U²-Net masks in full frame
                for mask, (x1, y1, x2, y2) in zip(u2net_masks, batch_positions):
                    # Resize U²-Net mask to crop size
                    mask_resized = cv2.resize(mask, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
                    u2net_mask_full[y1:y2, x1:x2] = np.maximum(
                        u2net_mask_full[y1:y2, x1:x2], mask_resized)

        # Add labels to distinguish the models
        cv2.putText(isnet_mask_full, "IS-Net", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(isnet_mask_full, f"Frame: {frame_count}/{total_frames}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(u2net_mask_full, "U2-Net", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(u2net_mask_full, f"Processed: {processed_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Create side-by-side comparison
        comparison_frame = np.hstack([isnet_mask_full, u2net_mask_full])
        
        # Show comparison
        if show_preview:
            cv2.imshow("Silhouette Comparison: IS-Net (Left) vs U²-Net (Right)", comparison_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if out:
            out.write(comparison_frame)
            
        prev_mask_isnet = isnet_mask_full.copy()
        prev_mask_u2net = u2net_mask_full.copy()
        
        # Progress update
        if processed_count % 10 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"⏳ Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")

    cap.release()
    if out:
        out.release()
    if show_preview:
        cv2.destroyAllWindows()
    print("✅ Comparison complete!")


def process_video_dual_comparison(video_path, output_path=None, show_preview=True, 
                                   frame_skip=2, batch_size=4, max_frames=None):
    """
    Compare IS-Net and U²-Net side by side
    
    Args:
        video_path: Path to input video
        output_path: Path to output video (optional)
        show_preview: Whether to show preview window
        frame_skip: Process every Nth frame (2 = every second frame)
        batch_size: Number of person crops to process in one batch
        max_frames: Maximum frames to process (None for all)
    """
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📹 Processing video: {total_frames} frames at {fps} FPS")
    print(f"⚡ Optimizations: frame_skip={frame_skip}, batch_size={batch_size}")
    print(f"🔍 Comparing IS-Net vs U²-Net")

    # Optional output writer for dual comparison video
    if output_path:
        # Create side-by-side comparison video (width * 2)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * 2, height), isColor=False)
    else:
        out = None

    # Load models
    print("🔄 Loading models...")
    yolo_model = YOLO("../../weights/yolo11m.pt")
    
    print("🔄 Loading IS-Net...")
    isnet_model = SilhouetteExtractorISNet(input_size=512)
    
    print("🔄 Loading U²-Net model...")
    try:
        u2net_model = SilhouetteExtractor()
        print(f"✅ U²-Net model loaded: {u2net_model.get_model_info()}")
    except Exception as e:
        print(f"❌ Failed to load U²-Net model: {e}")
        return
    
    frame_count = 0
    processed_count = 0
    prev_mask_isnet = np.zeros((height, width), dtype=np.uint8)
    prev_mask_u2net = np.zeros((height, width), dtype=np.uint8)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Skip frames for speed
        if frame_count % frame_skip != 0:
            if out:
                # For skipped frames, use previous masks
                comparison_frame = np.hstack([prev_mask_isnet, prev_mask_u2net])
                out.write(comparison_frame)
            continue
            
        # Stop if max frames reached
        if max_frames and processed_count >= max_frames:
            break
            
        processed_count += 1

        # YOLO inference for person detection
        results = yolo_model(frame, verbose=False)[0]
        
        # Initialize masks
        isnet_mask_full = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        u2net_mask_full = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        # Collect all person crops for batch processing
        person_crops = []
        crop_positions = []
        
        for box in results.boxes:
            cls_id = int(box.cls.item())
            if cls_id != 0:  # Only person class (class 0 in COCO)
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            # Padding
            pad = 15
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(frame.shape[1], x2 + pad)
            y2 = min(frame.shape[0], y2 + pad)

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            person_crops.append(person_crop)
            crop_positions.append((x1, y1, x2, y2))

        # Process crops with both models
        if person_crops:
            for i in range(0, len(person_crops), batch_size):
                batch_crops = person_crops[i:i+batch_size]
                batch_positions = crop_positions[i:i+batch_size]
                
                # Process with both models
                isnet_masks = isnet_model.extract_silhouette_batch(batch_crops)
                u2net_masks = u2net_model(batch_crops)
                
                # Place IS-Net masks in full frame
                for mask, (x1, y1, x2, y2) in zip(isnet_masks, batch_positions):
                    isnet_mask_full[y1:y2, x1:x2] = np.maximum(
                        isnet_mask_full[y1:y2, x1:x2], mask)
                
                # Place U²-Net masks in full frame
                for mask, (x1, y1, x2, y2) in zip(u2net_masks, batch_positions):
                    mask_resized = cv2.resize(mask, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
                    u2net_mask_full[y1:y2, x1:x2] = np.maximum(
                        u2net_mask_full[y1:y2, x1:x2], mask_resized)

        # Add labels to distinguish the models
        cv2.putText(isnet_mask_full, "IS-Net", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(isnet_mask_full, "Interactive Seg", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(isnet_mask_full, f"Frame: {frame_count}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.putText(u2net_mask_full, "U2-Net", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(u2net_mask_full, "Salient Object", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(u2net_mask_full, f"Processed: {processed_count}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Create dual side-by-side comparison
        comparison_frame = np.hstack([isnet_mask_full, u2net_mask_full])
        
        # Show comparison
        if show_preview:
            cv2.imshow("Dual Comparison: IS-Net (Left) | U²-Net (Right)", comparison_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if out:
            out.write(comparison_frame)
            
        prev_mask_isnet = isnet_mask_full.copy()
        prev_mask_u2net = u2net_mask_full.copy()
        
        # Progress update
        if processed_count % 10 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"⏳ Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")

    cap.release()
    if out:
        out.release()
    if show_preview:
        cv2.destroyAllWindows()
    print("✅ Dual comparison complete!")


def process_video_triple_comparison(video_path, output_path=None, show_preview=True, 
                                   frame_skip=2, batch_size=4, max_frames=None):
    """
    Compare IS-Net, U²-Net, and YOLO-Seg side by side (triple comparison)
    
    Args:
        video_path: Path to input video
        output_path: Path to output video (optional)
        show_preview: Whether to show preview window
        frame_skip: Process every Nth frame (2 = every second frame)
        batch_size: Number of person crops to process in one batch
        max_frames: Maximum frames to process (None for all)
    """
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📹 Processing video: {total_frames} frames at {fps} FPS")
    print(f"⚡ Optimizations: frame_skip={frame_skip}, batch_size={batch_size}")
    print(f"🔍 Comparing IS-Net vs U²-Net vs YOLO-Seg (Triple Comparison)")

    # Optional output writer for triple comparison video
    if output_path:
        # Create triple side-by-side comparison video (width * 3)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * 3, height), isColor=False)
    else:
        out = None

    # Load models
    print("🔄 Loading models...")
    yolo_detection_model = YOLO("../../weights/yolo11m.pt")  # For person detection
    
    print("🔄 Loading IS-Net...")
    isnet_model = SilhouetteExtractorISNet(input_size=512)
    
    print("🔄 Loading U²-Net...")
    try:
        u2net_model = SilhouetteExtractor()
        print(f"✅ U²-Net model loaded: {u2net_model.get_model_info()}")
    except Exception as e:
        print(f"❌ Failed to load U²-Net model: {e}")
        return
    
    print("🔄 Loading YOLO-Seg...")
    try:
        yolo_seg_model = SilhouetteExtractorYOLOSeg()
        print("✅ YOLO-Seg model loaded")
    except Exception as e:
        print(f"❌ Failed to load YOLO-Seg model: {e}")
        return
    
    frame_count = 0
    processed_count = 0
    prev_mask_isnet = np.zeros((height, width), dtype=np.uint8)
    prev_mask_u2net = np.zeros((height, width), dtype=np.uint8)
    prev_mask_yolo = np.zeros((height, width), dtype=np.uint8)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Apply frame skipping
        if frame_count % frame_skip != 0:
            continue
            
        # Apply max frames limit
        if max_frames and processed_count >= max_frames:
            break
            
        processed_count += 1

        # Detect persons using YOLO (for cropping)
        person_crops = []
        crop_positions = []
        
        results = yolo_detection_model(frame, verbose=False)
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Filter for person class (0) with reasonable confidence
                    if class_id == 0 and confidence > 0.5:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        
                        # Ensure coordinates are within frame bounds
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(width, x2)
                        y2 = min(height, y2)
                        
                        # Check if bounding box is reasonable
                        if x2 > x1 + 20 and y2 > y1 + 20:  # Minimum size filter
                            person_crop = frame[y1:y2, x1:x2]
                            person_crops.append(person_crop)
                            crop_positions.append((x1, y1, x2, y2))

        # Create masks for all three models
        isnet_mask_full = np.zeros((height, width), dtype=np.uint8)
        u2net_mask_full = np.zeros((height, width), dtype=np.uint8)
        yolo_mask_full = np.zeros((height, width), dtype=np.uint8)

        # Process crops with all three models
        if person_crops:
            for i in range(0, len(person_crops), batch_size):
                batch_crops = person_crops[i:i+batch_size]
                batch_positions = crop_positions[i:i+batch_size]
                
                # Process with all three models
                isnet_masks = isnet_model.extract_silhouette_batch(batch_crops)
                u2net_masks = u2net_model(batch_crops)
                yolo_masks = yolo_seg_model.extract_silhouette_batch(batch_crops)
                
                # Place IS-Net masks in full frame
                for mask, (x1, y1, x2, y2) in zip(isnet_masks, batch_positions):
                    isnet_mask_full[y1:y2, x1:x2] = np.maximum(
                        isnet_mask_full[y1:y2, x1:x2], mask)
                
                # Place U²-Net masks in full frame
                for mask, (x1, y1, x2, y2) in zip(u2net_masks, batch_positions):
                    mask_resized = cv2.resize(mask, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
                    u2net_mask_full[y1:y2, x1:x2] = np.maximum(
                        u2net_mask_full[y1:y2, x1:x2], mask_resized)
                
                # Place YOLO-Seg masks in full frame
                for mask, (x1, y1, x2, y2) in zip(yolo_masks, batch_positions):
                    yolo_mask_full[y1:y2, x1:x2] = np.maximum(
                        yolo_mask_full[y1:y2, x1:x2], mask)

        # Add labels to distinguish the models
        cv2.putText(isnet_mask_full, "IS-Net", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(isnet_mask_full, "Interactive Seg", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(isnet_mask_full, f"Frame: {frame_count}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.putText(u2net_mask_full, "U2-Net", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(u2net_mask_full, "Salient Object", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(u2net_mask_full, f"Processed: {processed_count}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.putText(yolo_mask_full, "YOLO-Seg", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(yolo_mask_full, "Instance Seg", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(yolo_mask_full, f"Batch: {batch_size}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Create triple side-by-side comparison
        comparison_frame = np.hstack([isnet_mask_full, u2net_mask_full, yolo_mask_full])
        
        # Show comparison
        if show_preview:
            cv2.imshow("Triple Comparison: IS-Net | U²-Net | YOLO-Seg", comparison_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if out:
            out.write(comparison_frame)
            
        prev_mask_isnet = isnet_mask_full.copy()
        prev_mask_u2net = u2net_mask_full.copy()
        prev_mask_yolo = yolo_mask_full.copy()
        
        # Progress update
        if processed_count % 10 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"⏳ Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")

    cap.release()
    if out:
        out.release()
    if show_preview:
        cv2.destroyAllWindows()
    print("✅ Triple comparison complete!")


# === Run it ===
if __name__ == "__main__":
    video_path = "/Users/prachit/Projects/Computer_Vision/Person_Identification/v_0/input/3c.mp4"
    
    # Choose your comparison mode:
    
    # 1. TRIPLE COMPARISON (IS-Net vs U²-Net vs YOLO-Seg) - NEW FEATURE
    output_path = "/Users/prachit/Projects/Computer_Vision/Person_Identification/v_0/output/triple_comparison_isnet_u2net_yolo_extended.mp4"
    process_video_triple_comparison(
        video_path=video_path, 
        output_path=output_path, 
        show_preview=True,
        frame_skip=2,     # Process every 2nd frame for speed
        batch_size=4,     # Process 4 person crops at once
        max_frames=100    # Extended evaluation with more frames
    )
    
    # 2. DUAL COMPARISON (IS-Net vs U²-Net) - Uncomment to use
    # output_path = "/Users/prachit/Projects/Computer_Vision/Person_Identification/v_0/output/dual_comparison_isnet_u2net.mp4"
    # process_video_dual_comparison(
    #     video_path=video_path, 
    #     output_path=output_path, 
    #     show_preview=True,
    #     frame_skip=2,     # Process every 2nd frame for speed
    #     batch_size=4,     # Process 4 person crops at once
    #     max_frames=150    # Limit for testing
    # )
    
    # 3. ALTERNATIVE DUAL COMPARISON (using different function) - Uncomment to use
    # output_path = "/Users/prachit/Projects/Computer_Vision/Person_Identification/v_0/output/comparison_isnet_u2net.mp4"
    # process_video_comparison(
    #     video_path=video_path, 
    #     output_path=output_path, 
    #     show_preview=True,
    #     frame_skip=2,     # Process every 2nd frame for speed
    #     batch_size=4,     # Process 4 person crops at once
    #     max_frames=200    # Limit for testing
    # )

    # 4. INDIVIDUAL MODEL TESTING - Uncomment to use
    # For IS-Net only:
    # isnet_model = SilhouetteExtractorISNet()
    # # Use similar processing logic with isnet_model
