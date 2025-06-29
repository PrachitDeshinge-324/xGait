import cv2
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from src.models.cdgnet_official import Res_Deeplab

# --- CONFIG ---
VIDEO_PATH = 'input/3c.mp4'  # Change as needed
FRAME_IDX = 100
CROP_SIZE = 256  # Size of the crop (center crop)
NUM_CLASSES = 12  # As per your CDGNet config
WEIGHTS_PATH = 'weights/cdgnet.pth'  # Path to your downloaded weights


crop = cv2.imread('input/demo_person_crop.png')
# --- Preprocess for CDGNet ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((CROP_SIZE, CROP_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = transform(crop).unsqueeze(0)  # Shape: [1, 3, H, W]

# --- Load model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Res_Deeplab(num_classes=NUM_CLASSES)
# --- Remove 'module.' prefix if present in state dict ---
state_dict = torch.load(WEIGHTS_PATH, map_location=device)
if any(k.startswith('module.') for k in state_dict.keys()):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    state_dict = new_state_dict
model.load_state_dict(state_dict)
model.eval()
model.to(device)

# --- Inference ---
with torch.no_grad():
    output = model(input_tensor.to(device))
    parsing_logits = output[0][-1]  # seg2
    parsing = parsing_logits.argmax(1).squeeze().cpu().numpy().astype(np.uint8)

# --- Visualization ---
# Assign a color to each class (12-class palette from official CDGNet-Parsing)
palette = np.array([
    [0, 0, 0],        # 0: background
    [128, 0, 0],     # 1
    [0, 128, 0],     # 2
    [128, 128, 0],   # 3
    [0, 0, 128],     # 4
    [128, 0, 128],   # 5
    [0, 128, 128],   # 6
    [128, 128, 128], # 7
    [64, 0, 0],      # 8
    [192, 0, 0],     # 9
    [64, 128, 0],    # 10
    [192, 128, 0],   # 11
], dtype=np.uint8)
parsing_rgb = palette[parsing]

# Resize parsing result to match crop size
parsing_rgb_resized = cv2.resize(parsing_rgb, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_NEAREST)

# Overlay parsing on crop
alpha = 0.5
crop_rgb = cv2.cvtColor(cv2.resize(crop, (CROP_SIZE, CROP_SIZE)), cv2.COLOR_BGR2RGB)
overlay = (alpha * crop_rgb + (1 - alpha) * parsing_rgb_resized).astype(np.uint8)

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title('Cropped Person')
plt.imshow(crop_rgb)
plt.axis('off')
plt.subplot(1, 3, 2)
plt.title('Parsing Result')
plt.imshow(parsing_rgb)
plt.axis('off')
plt.subplot(1, 3, 3)
plt.title('Overlay')
plt.imshow(overlay)
plt.axis('off')
plt.tight_layout()
plt.show()
