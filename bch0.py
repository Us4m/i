import cv2
import numpy as np

# Load the input image
input_image = cv2.imread('input_image.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Apply a threshold to convert the grayscale image to a binary mask
ret, binary_mask = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY)

# Invert the binary mask
inverted_binary_mask = cv2.bitwise_not(binary_mask)

# Apply the binary mask to the input image to remove the background
foreground = cv2.bitwise_and(input_image, input_image, mask=inverted_binary_mask)

# Load the new background image
background_image = cv2.imread('background_image.jpg')

# Resize the new background image to the size of the input image
resized_background_image = cv2.resize(background_image, (input_image.shape[1], input_image.shape[0]))

# Apply the inverted binary mask to the new background image
background = cv2.bitwise_and(resized_background_image, resized_background_image, mask=binary_mask)

# Combine the background and foreground images using the bitwise OR operation
result = cv2.bitwise_or(background, foreground)

# Save the result as a new image
cv2.imwrite('output_image.jpg', result)

