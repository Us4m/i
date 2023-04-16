import cv2
import numpy as np
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.plots import plot_one_box

# Load YOLOv8 model
model = attempt_load('yolov8.pt', map_location=torch.device('cpu'))

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load image and background
img_path = 'image.jpg'
bg_path = 'background.jpg'
img = cv2.imread(img_path)
bg = cv2.imread(bg_path)

# Resize image to model input size
img = cv2.resize(img, (640, 640))

# Convert image to tensor
img = torch.from_numpy(img).to(device).float()
img /= 255.0
img = img.unsqueeze(0)

# Make predictions
pred = model(img)[0]

# Apply non-maximum suppression
pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

# Get bounding box coordinates and class labels
boxes = []
for det in pred:
    if det is not None and len(det):
        det[:, :4] = det[:, :4].clone().cpu().detach().numpy()
        boxes.extend(det[:, :4])

# Create mask
mask = np.zeros_like(img.cpu().squeeze().numpy(), dtype=np.uint8)
for box in boxes:
    box = box.astype(np.int32)
    mask = cv2.rectangle(mask, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), -1)

# Resize background to match image size
bg = cv2.resize(bg, (640, 640))

# Apply mask to image
result = cv2.bitwise_and(img.cpu().squeeze().numpy(), mask)

# Invert mask and apply to background
inv_mask = cv2.bitwise_not(mask)
bg_masked = cv2.bitwise_and(bg, inv_mask)

# Combine masked image and background
result = cv2.add(result, bg_masked)

# Save result as PNG file
cv2.imwrite('result.png', result * 255.0)
