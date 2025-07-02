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

class ISNetModel(nn.Module):
    """
    IS-Net (Interactive Segmentation Network) implementation
    Based on the official IS-Net architecture for salient object detection
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(ISNetModel, self).__init__()
        
        # Encoder (similar to U-Net but with IS-Net improvements)
        self.encoder1 = self._make_layer(in_ch, 64)
        self.encoder2 = self._make_layer(64, 128)
        self.encoder3 = self._make_layer(128, 256)
        self.encoder4 = self._make_layer(256, 512)
        
        # Bridge
        self.bridge = self._make_layer(512, 1024)
        
        # Decoder with skip connections
        self.decoder4 = self._make_layer(1024 + 512, 512)
        self.decoder3 = self._make_layer(512 + 256, 256)
        self.decoder2 = self._make_layer(256 + 128, 128)
        self.decoder1 = self._make_layer(128 + 64, 64)
        
        # Output layer
        self.final = nn.Conv2d(64, out_ch, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        
    def _make_layer(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        
        # Bridge
        bridge = self.bridge(self.pool(e4))
        
        # Decoder with skip connections
        d4 = F.interpolate(bridge, scale_factor=2, mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.decoder4(d4)
        
        d3 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)
        
        d2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)
        
        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)
        
        # Final output
        output = self.final(d1)
        
        return output

class SimpleISNet(nn.Module):
    """
    Simplified IS-Net implementation as fallback
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(SimpleISNet, self).__init__()
        
        # Simple encoder-decoder structure
        self.conv1 = nn.Conv2d(in_ch, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(64, out_ch, 2, stride=2)
        
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Encoder
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(self.pool(x1)))
        x3 = self.relu(self.conv3(self.pool(x2)))
        
        # Decoder
        x = self.up3(x3)
        x = self.up2(x)
        x = self.up1(x)
        
        return x

# ===== Silhouette Extractors =====
# Only IS-Net and U²-Net implementations are included

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
            # Try to use a pre-trained salient object detection model
            import torchvision.models as models
            
            # Use a modified U-Net like architecture for IS-Net
            model = ISNetModel()
            
            # Try to load pre-trained weights if available
            weight_path = "weights/isnet.pth"
            if Path(weight_path).exists():
                model.load_state_dict(torch.load(weight_path, map_location='cpu'))
                print("✅ Loaded IS-Net pre-trained weights")
            else:
                print("⚠️ No IS-Net weights found, using random initialization")
                
            return model
            
        except Exception as e:
            print(f"⚠️ Failed to load IS-Net, falling back to modified U-Net: {e}")
            return self._build_unet_backbone()

    def _build_unet_backbone(self):
        """Fallback U-Net backbone for IS-Net"""
        return SimpleISNet()

    def extract_silhouette(self, image: np.ndarray) -> np.ndarray:
        original_h, original_w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(image_rgb).to(self.device)

        with torch.no_grad():
            # IS-Net outputs probability maps
            output = self.model(input_tensor.unsqueeze(0))
            
            # Handle different output formats
            if isinstance(output, tuple):
                pred = output[0]  # Main prediction
            else:
                pred = output
                
            # Convert to binary mask
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
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Main predictions
                    
                valid_masks = []
                
                for output in outputs:
                    pred = torch.sigmoid(output)
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

# === Run it ===
if __name__ == "__main__":
    video_path = "/Users/prachit/Projects/Computer_Vision/Person_Identification/v_0/input/3c.mp4"
    
    # Choose your comparison mode:
    
    # 1. DUAL COMPARISON (IS-Net vs U²-Net) - RECOMMENDED
    output_path = "/Users/prachit/Projects/Computer_Vision/Person_Identification/v_0/output/dual_comparison_isnet_u2net.mp4"
    process_video_dual_comparison(
        video_path=video_path, 
        output_path=output_path, 
        show_preview=True,
        frame_skip=2,     # Process every 2nd frame for speed
        batch_size=4,     # Process 4 person crops at once
        max_frames=150    # Limit for testing
    )
    
    # 2. ALTERNATIVE DUAL COMPARISON (using different function) - Uncomment to use
    # output_path = "/Users/prachit/Projects/Computer_Vision/Person_Identification/v_0/output/comparison_isnet_u2net.mp4"
    # process_video_comparison(
    #     video_path=video_path, 
    #     output_path=output_path, 
    #     show_preview=True,
    #     frame_skip=2,     # Process every 2nd frame for speed
    #     batch_size=4,     # Process 4 person crops at once
    #     max_frames=200    # Limit for testing
    # )

    # 3. INDIVIDUAL MODEL TESTING - Uncomment to use
    # For IS-Net only:
    # isnet_model = SilhouetteExtractorISNet()
    # # Use similar processing logic with isnet_model
