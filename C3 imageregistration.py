# -*- coding: utf-8 -*-
"""ImageRegistration.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1aLf1Ugod2V2V3kYF9-rSX1ySOgTpXhlJ
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import ORB, match_descriptors, plot_matches
from skimage.transform import AffineTransform, warp, resize
from skimage.measure import ransac

image2 = cv2.imread('/content/drive/MyDrive/CV Exam Datasets /image_registration_dataset/laptop_register.jpg', cv2.IMREAD_GRAYSCALE)
image1 = cv2.imread('/content/drive/MyDrive/CV Exam Datasets /image_registration_dataset/laptop_base.jpg', cv2.IMREAD_GRAYSCALE)
plot_image1 = cv2.imread('/content/drive/MyDrive/CV Exam Datasets /image_registration_dataset/laptop_base.jpg')
plot_image2 = cv2.imread('/content/drive/MyDrive/CV Exam Datasets /image_registration_dataset/laptop_register.jpg')

if image1 is None or image2 is None:
    raise ValueError("One or both images failed to load. Check the file paths.")

image1 = resize(image1, (500, 500), anti_aliasing=True)
image2 = resize(image2, (500, 500), anti_aliasing=True)
plot_image1 = resize(plot_image1, (500, 500), anti_aliasing=True)
plot_image2 = resize(plot_image2, (500, 500), anti_aliasing=True)

orb = ORB(n_keypoints=500)

orb.detect_and_extract(image1)
keypoints1 = orb.keypoints
descriptors1 = orb.descriptors

orb.detect_and_extract(image2)
keypoints2 = orb.keypoints
descriptors2 = orb.descriptors

matches = match_descriptors(descriptors1, descriptors2, cross_check=True)

src = keypoints1[matches[:, 0]]
dst = keypoints2[matches[:, 1]]

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
plt.gray()

plot_matches(ax, plot_image1, plot_image2, keypoints1, keypoints2, matches)
ax.axis('off')
ax.set_title("Keypoint Matches")
plt.show()

print(f"Number of matches: {len(matches)}")

if len(matches) < 4:
    raise ValueError("Not enough matches to compute a reliable transformation")

model_robust, inliers = ransac((dst, src),
                               AffineTransform, min_samples=4,
                               residual_threshold=2, max_trials=1000)

registered_image = warp(plot_image2, model_robust.inverse, output_shape=image2.shape)

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Image 1')
plt.imshow(plot_image1, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Image 2')
plt.imshow(plot_image2, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Registered Image')
plt.imshow(registered_image, cmap='gray')

plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path, size=(512, 512)):
    """Load and preprocess the image: load, resize, and convert to grayscale."""
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, size)
    gray_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    return img_resized, gray_img

def detect_and_match_keypoints(gray1, gray2):
    """Detect keypoints and match descriptors using ORB and brute force matcher."""
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Brute Force Matcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    return kp1, kp2, matches

def compute_affine_transformation(pts1, pts2):
    """Compute the affine transformation matrix and warp the image."""
    affine_matrix, _ = cv2.estimateAffine2D(pts2, pts1)
    return affine_matrix

def warp_image(image, transformation_matrix, size, is_homography=False):
    """Warp the image using the provided transformation matrix."""
    if is_homography:
        return cv2.warpPerspective(image, transformation_matrix, size)
    else:
        return cv2.warpAffine(image, transformation_matrix, size)

def visualize_results(img1, img2, matches, affine_aligned):
    """Visualize the original images, matched keypoints, and aligned results."""
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(10, 8))

    # Original Images
    plt.subplot(231), plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)), plt.title('Fossil Image 1')
    plt.subplot(232), plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)), plt.title('Fossil Image 2')

    # Matched Keypoints
    plt.subplot(233), plt.imshow(img_matches), plt.title('Matched Keypoints')

    # Affine Aligned Image
    plt.subplot(234), plt.imshow(cv2.cvtColor(affine_aligned, cv2.COLOR_BGR2RGB)), plt.title('Affine Aligned')

    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load and preprocess images
    img1, gray1 = load_and_preprocess_image('/content/drive/MyDrive/CV Exam Datasets /image_registration_dataset/laptop_base.jpg')
    img2, gray2 = load_and_preprocess_image('/content/drive/MyDrive/CV Exam Datasets /image_registration_dataset/laptop_register.jpg')

    # Detect and match keypoints
    kp1, kp2, matches = detect_and_match_keypoints(gray1, gray2)

    # Extract matched keypoints
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute transformations
    affine_matrix = compute_affine_transformation(pts1, pts2)

    # Warp images
    affine_aligned = warp_image(img2, affine_matrix, img1.shape[1::-1])

    # Visualize results
    visualize_results(img1, img2, matches, affine_aligned)