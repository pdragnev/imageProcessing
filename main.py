import cv2
import numpy as np
import os

# Read the input
img = cv2.imread('images/ZV1zy.jpg')

# Check if the image was loaded correctly
if img is None:
    print("Error: Image not found")
    exit()

# Apply histogram equalization on the color image in the HSV space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
v_eq = cv2.equalizeHist(v)
hsv_eq = cv2.merge([h, s, v_eq])
img_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

# Apply a bilateral filter to reduce noise while keeping the edges sharp
filtered_img = cv2.bilateralFilter(img_eq, 9, 75, 75)

# Sharpen the image
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpened_image = cv2.filter2D(filtered_img, -1, kernel)

# Resize (upscale) if needed - change 'new_width' and 'new_height' to your desired dimensions
new_width, new_height = img.shape[1], img.shape[0]  # Keep original dimensions or set new ones
upscaled_image = cv2.resize(sharpened_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

# Calculate the center of the image (splitting in half)
center_x = upscaled_image.shape[1] // 2

# Split the image into left and right halves
left_half = upscaled_image[:, :center_x]
right_half = upscaled_image[:, center_x:]

# Save the full enhanced image
cv2.imwrite('processed/ZV1zy_v7.jpg', upscaled_image)

# Save each half
cv2.imwrite('processed/ZV1zy_v7_left_half.jpg', left_half)
cv2.imwrite('processed/ZV1zy_v7_right_half.jpg', right_half)


# Optionally, show the result
cv2.imshow('Enhanced Color Image', upscaled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



'''
v1
# read the input
img = cv2.imread('images/ZV1zy.jpg')

# stretch the dynamic range
stretch = skimage.exposure.rescale_intensity(img, in_range=(50,175), out_range=(0,255)).astype(np.uint8)

# save results
cv2.imwrite('processed/ZV1zy_v1.jpg', stretch)

v2 and v3 with/without resizing
import cv2
import numpy as np
import skimage.exposure

# Read the input
img = cv2.imread('images/ZV1zy.jpg')

# Check if image is loaded correctly
if img is None:
    print("Error: Image not found")
    exit()

# Stretch the dynamic range
stretch = skimage.exposure.rescale_intensity(img, in_range=(50,175), out_range=(0,255)).astype(np.uint8)

# Denoise
denoised_image = cv2.fastNlMeansDenoisingColored(stretch, None, 10, 10, 7, 21)

# Sharpen
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpened_image = cv2.filter2D(denoised_image, -1, kernel)

# Resize (upscale) if needed - change 'new_width' and 'new_height' to your desired dimensions
new_width, new_height = 1920, 1080
upscaled_image = cv2.resize(sharpened_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

# Save results
cv2.imwrite('processed/ZV1zy_v3.jpg', upscaled_image)

# Optionally, show the result
cv2.imshow('Processed Image', upscaled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

v4 v5
import cv2
import numpy as np
import skimage.exposure

# Read the input
img = cv2.imread('images/ZV1zy.jpg')

# Check if the image was loaded correctly
if img is None:
    print("Error: Image not found")
    exit()


# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Stretch the dynamic range and convert to 8-bit
stretch = skimage.exposure.rescale_intensity(gray, in_range=(50,175), out_range=(0,255)).astype(np.uint8)

# Apply adaptive thresholding
thresh = cv2.adaptiveThreshold(stretch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


# Denoise (might not be as necessary after thresholding, but can still be useful)
denoised_image = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)

# Sharpen
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpened_image = cv2.filter2D(denoised_image, -1, kernel)

# Resize (upscale) if needed - change 'new_width' and 'new_height' to your desired dimensions
new_width, new_height = img.shape[1], img.shape[0]  # Keep original dimensions or set new ones
upscaled_image = cv2.resize(sharpened_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

# Save results
cv2.imwrite('processed/ZV1zy_v5.jpg', upscaled_image)

# Optionally, show the result
cv2.imshow('Processed Image', upscaled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


v6
img = cv2.imread('images/ZV1zy.jpg')

# Check if the image was loaded correctly
if img is None:
    print("Error: Image not found")
    exit()

# Apply histogram equalization on the color image in the HSV space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
v_eq = cv2.equalizeHist(v)
hsv_eq = cv2.merge([h, s, v_eq])
img_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

# Apply a bilateral filter to reduce noise while keeping the edges sharp
filtered_img = cv2.bilateralFilter(img_eq, 9, 75, 75)

# Sharpen the image
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpened_image = cv2.filter2D(filtered_img, -1, kernel)

# Resize (upscale) if needed - change 'new_width' and 'new_height' to your desired dimensions
new_width, new_height = img.shape[1], img.shape[0]  # Keep original dimensions or set new ones
upscaled_image = cv2.resize(sharpened_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

# Save results
cv2.imwrite('processed/ZV1zy_v6.jpg', upscaled_image)

# Optionally, show the result
cv2.imshow('Enhanced Color Image', upscaled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
