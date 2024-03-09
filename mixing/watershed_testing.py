import PIL
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_local
from scipy import signal
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util
)
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import data, transform, exposure
from skimage.util import compare_images

image_filename = r'/Users/lukebrown/Downloads/grimsvotn_middle.png'
image = PIL.Image.open(image_filename)
gray_image = image.convert('L')
gray_image = np.array(gray_image)
gray_image_equalized = exposure.equalize_hist(gray_image)
# image = np.zeros((256, 256), dtype=np.uint8)
# image[64:192, 64:192] = 255

distance = ndi.distance_transform_edt(gray_image)

peaks = peak_local_max(distance, footprint=np.ones((3, 3)), labels=gray_image)

mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(peaks.T)] = True

markers, _ = ndi.label(mask)
labels = watershed(-distance, markers=markers, mask=gray_image)

background_label = labels[-1, 0]
background = (labels == background_label)

from skimage.exposure import histogram

hist, hist_centers = histogram(gray_image_equalized)

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].imshow(gray_image_equalized, cmap=plt.cm.gray)
axes[0].axis('off')
axes[1].plot(hist_centers, hist, lw=2)
axes[1].set_title('histogram of gray values')

fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)

axes[0].imshow(gray_image_equalized > 0.6, cmap=plt.cm.gray)
axes[0].set_title('gray_image > 0.6')

axes[1].imshow(gray_image_equalized > 0.75, cmap=plt.cm.gray)
axes[1].set_title('gray_image > 0.75')

for a in axes:
    a.axis('off')

plt.tight_layout()

thresholds = filters.threshold_multiotsu(gray_image_equalized - np.mean(gray_image_equalized), classes=3)
regions = np.digitize(gray_image_equalized - np.mean(gray_image_equalized), bins=thresholds)

fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
ax[0].imshow(gray_image_equalized - np.mean(gray_image_equalized))
ax[0].set_title('Original')
ax[0].axis('off')
ax[1].imshow(regions)
ax[1].set_title('Multi-Otsu thresholding')
ax[1].axis('off')
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(13, 9), dpi=90)
axes[0].imshow(gray_image, cmap='gray')
axes[0].axis('off')
axes[1].imshow(gray_image_equalized, cmap='gray')
# ax.set_title('Contour plot of the same raw image')
axes[2].imshow(background, cmap=plt.cm.nipy_spectral)
axes[2].set_title('Identified Background')
axes[2].axis('off')
plt.show()

# image = data.page()

global_thresh = threshold_otsu(gray_image_equalized)
binary_global = gray_image_equalized > global_thresh

block_size = 35
local_thresh = threshold_local(gray_image_equalized, block_size, offset=0.1)
binary_local = gray_image_equalized > local_thresh

fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
ax = axes.ravel()
plt.gray()

ax[0].imshow(gray_image_equalized)
ax[0].set_title('Original')

ax[1].imshow(binary_global)
ax[1].set_title('Global thresholding')

ax[2].imshow(binary_local)
ax[2].set_title('Local thresholding')

for a in ax:
    a.axis('off')

plt.show()
