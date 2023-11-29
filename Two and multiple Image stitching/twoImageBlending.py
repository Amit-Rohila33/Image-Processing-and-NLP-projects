import cv2
import numpy as np
from random import randrange

def blend_images(img1, img2, mask):
    # Generate Laplacian pyramids for the images
    L_pyr_img1 = cv2.buildLaplacianPyramid(img1, (3, 3))
    L_pyr_img2 = cv2.buildLaplacianPyramid(img2, (3, 3))

    # Generate Gaussian pyramid for the mask
    G_pyr_mask = cv2.buildPyramid(mask, (3, 3))

    # Combine the Laplacian pyramids using the mask
    blended_pyr = []
    for la1, la2, gm in zip(L_pyr_img1, L_pyr_img2, G_pyr_mask):
        blended_pyr.append(gm * la1 + (1 - gm) * la2)

    # Collapse the blended pyramid to get the final blended image
    blended_image = cv2.collapseLaplacianPyramid(blended_pyr)

    return blended_image

# Load your images
img1 = cv2.imread('uttower_left.JPG')
img2 = cv2.imread('uttower_right.JPG')

# Convert images to grayscale
gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Perform feature matching (your existing code)
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray_img1, None)
kp2, des2 = sift.detectAndCompute(gray_img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m in matches:
    if m[0].distance < 0.5 * m[1].distance:
        good.append(m)
matches = np.asarray(good)

# Check if enough keypoints are found
if len(matches[:, 0]) >= 4:
    src = np.float32([kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
else:
    raise AssertionError("Can't find enough keypoints.")

# Warp the images using the homography matrix
resultant_stitched_panorama = cv2.warpPerspective(img1, H, (img2.shape[1] + img1.shape[1], img2.shape[0]))

# Create a mask for blending
mask = np.zeros_like(resultant_stitched_panorama)
mask[:, :img2.shape[1]] = 1

# Blend the images
blended_result = blend_images(resultant_stitched_panorama, img2, mask)

# Display the blended result
cv2.imshow('Blended Result', blended_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
