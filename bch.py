import cv2
import numpy as np
import torch

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', force_reload=True)

# Load image
img = cv2.imread("input_image.jpg")

# Detect objects using YOLOv5
results = model(img)

# Extract object bounding boxes
boxes = results.xyxy[0].cpu().numpy()

# Concatenate class IDs from all arrays in list
class_ids = np.concatenate([result[:, 5] for result in results.xyxy], axis=0).astype(int)

# Apply mask to remove background
mask = np.zeros_like(img)
for i in range(len(boxes)):
    xmin, ymin, xmax, ymax, confidence = boxes[i]
    x, y, w, h, confidence = xmin, ymin, xmax-xmin, ymax-ymin, confidence
    mask[int(y):int(y+h), int(x):int(x+w)] = 255
img_masked = cv2.bitwise_and(img, mask)

# Save masked image
cv2.imwrite("masked_image.png", img_masked)

# Load background image
background = cv2.imread("background_image.jpg")

# Resize background image to match masked image size
background = cv2.resize(background, (img.shape[1], img.shape[0]))

# Invert mask to create inverse mask
mask_inv = cv2.bitwise_not(mask)

# Apply inverse mask to background image
background_masked = cv2.bitwise_and(background, mask_inv)

# Merge masked image and background image
img_merged = cv2.addWeighted(img_masked, 1, background_masked, 1, 0)

# Save merged image
cv2.imwrite("merged_image.png", img_merged)
