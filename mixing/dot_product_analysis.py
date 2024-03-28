# Importing the libraries
import numpy as np
# import cv2 as cv
# import cv2 as cv2
from PIL import Image
import matplotlib.pyplot as plt
import glob
from matplotlib.patches import Rectangle
import PIL
import os
import seaborn as sns
from skimage import filters
import skimage.measure
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
from skimage.morphology import disk

import PIL
import os
import seaborn as sns
from skimage import filters
import skimage.measure
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
from skimage.morphology import disk
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

sns.set_theme("notebook")

# Eyjafjallajökull 2010: example of unmixed plume
# read it in grayscale from copy A volcano images (To not ruin the images in Dropbox)

filename = r'/Users/lukebrown/Downloads/eyja_end.png'
# filename = r'/Users/lukebrown/Downloads/grimsvotn_middle.png'

# eyja = cv.imread(r'/Users/lukebrown/Downloads/eyja_end.png', cv.IMREAD_GRAYSCALE)

image = PIL.Image.open(filename)

gray_image = image.convert('L')
eyja = np.array(gray_image)
plt.imshow(eyja, cmap='gray')

disk_size = 11

# gray_grimsvotn = grim.convert('L')
gray_eyja = np.array(eyja)

# compute entropy to identify subject
footprint = disk(disk_size)
entropy_measure = skimage.filters.rank.entropy(gray_eyja, footprint)

# otsu threshold
thresholds = filters.threshold_multiotsu(entropy_measure, classes=2)
regions = np.digitize(entropy_measure, bins=thresholds)
regions = closing(regions, footprint)  # closing dark spots inside regions
eruption_mask_eyja = gray_eyja * regions
cropped_eyja = eruption_mask_eyja.astype(float)
# cropped_eyja[cropped_eyja.astype(float) == 0] = np.nan

## EYJA


# get the X&Y dimensions
imgheight = cropped_eyja.shape[0]
imgwidth = cropped_eyja.shape[1]
y1 = 0

# create a new list for storing the tiles
tiles_eyja = []

# cutting lengthwise or horizontal wise depending on which is longer

if imgheight > imgwidth:
    M = imgheight // 6
    N = imgwidth // 3
else:
    M = imgheight // 3
    N = imgwidth // 6

# for loop to split the image into the tiles
for y in range(0, imgheight, M):
    for x in range(0, imgwidth, N):
        y1 = y + M
        x1 = x + N
        tiles = cropped_eyja[y:y + M, x:x + N]
        tiles_eyja.append(tiles)

# plot all segments using a for loop

fig = plt.figure(figsize=(15, 9))

# setting values to rows and column variables
rows = 3
columns = 10

for i in range(len(tiles_eyja)):
    fig.add_subplot(rows, columns, i + 1)
    plt.imshow(tiles_eyja[i], cmap='gray')
    plt.axis('off')
    plt.title(str(i))

filter_eyja = tiles_eyja

plt.show()


# define dot product function

def dot_product(image1, image2):
    dtype_max = 255
    tolerance = 1e-5
    image1_float = image1.astype(float) / dtype_max
    image2_float = image2.astype(float) / dtype_max

    image1_norm = np.linalg.norm(image1_float)
    image2_norm = np.linalg.norm(image2_float)
    dot_product_norm = np.dot(image1_float.flatten(), image2_float.flatten())

    theta = np.arccos(dot_product_norm / (image1_norm * image2_norm))
    return dot_product_norm.round(2), theta.round(2)


# EYJA

# dot product for resized image using the artifical gray tile as a referance


# From this smaller seems to indicate more mixing????????
grays = [0, 80, 180, 255]

gray_refs = [Image.new('L', (56, 60), color=color) for color in grays]
grey_ref = np.array(Image.new('L', (56, 60), color=180))

# fig, ax = plt.subplots(1, len(grays), figsize=(10,5))
# for axes, gray in zip(ax, gray_refs):
#     print(np.array(gray))
# #     gray.show(grey_ref)

# plt.show()

# print(str(dot_product(grey_ref, grey_ref)) + ": " + "ref x ref")

for i in range(len(filter_eyja)):
    print(f"run: {i}")
    if filter_eyja[i].size < grey_ref.size:
        break
    for gray in gray_refs:
        print(str(dot_product(np.array(gray), filter_eyja[i])))  # + ": " + "gray ref by " + str(i))
