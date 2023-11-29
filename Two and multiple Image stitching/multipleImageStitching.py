import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to stitch two images
def stitch_images(img1, img2):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m in matches:
        if m[0].distance < 0.5 * m[1].distance:
            good.append(m)
    matches = np.asarray(good)

    if len(matches[:, 0]) >= 4:
        src = np.float32([kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)

        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        print(H)
    else:
        raise AssertionError("Can't find enough keypoints.")

    dst = cv2.warpPerspective(img1, H, (img2.shape[1] + img1.shape[1], img2.shape[0]))

    dst[0:img2.shape[0], 0:img2.shape[1]] = img2

    return dst, H

# Load images
images = []
for i in range(175,170,-1):
    img = cv2.imread(f'DSC_0{i}.JPG')
    images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

# Stitch images iteratively
resultant_image = images[0]
homography_matrices = []

for i in range(1, len(images)):
    resultant_image, H = stitch_images(resultant_image, images[i])
    homography_matrices.append(H)
    
# Convert the resultant image to RGB color

# Display the resultant stitched image
plt.imshow(resultant_image, cmap="gray")
plt.title('Resultant Stitched Panorama')
plt.show()

# Save the resultant stitched image
cv2.imwrite('resultant_stitched_panorama.jpg', resultant_image)
