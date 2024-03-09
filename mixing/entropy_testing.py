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

"""
1. spatial subsets of image [X]
2. zero mean! i __mean__ it [X]
3. stretch using darks and lights (start with 2 categories and work up)
    a. use peak_local_max to create markers
4. tapering: "tukey" versus "hanning" [X]
"""
# grimsvotn_filname = r'/Users/lukebrown/Downloads/grimsvotn_middle.png'
grimsvotn_filname = r'/Users/lukebrown/Downloads/grimsvotn_end.png'
eyja_filename = r'/Users/lukebrown/Downloads/eyja_end.png'
filenames = [grimsvotn_filname, eyja_filename]


def analyze_eruption(
        filename,
        disk_size=11,
        make_plots=True,
        save_plots=False,
        welch_scaling='spectrum'
):
    image = PIL.Image.open(filename)

    gray_image = image.convert('L')
    gray_image = np.array(gray_image)

    # compute entropy to identify subject
    footprint = disk(disk_size)
    entropy_measure = skimage.filters.rank.entropy(gray_image, footprint)

    # otsu threshold
    thresholds = filters.threshold_multiotsu(entropy_measure, classes=2)
    regions = np.digitize(entropy_measure, bins=thresholds)
    regions = closing(regions, footprint)  # closing dark spots inside regions
    eruption_mask = gray_image * regions

    if make_plots:
        # fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
        #
        # axes[0].imshow(image, cmap=plt.cm.gray)
        # axes[0].set_title(f"{os.path.basename(filename).split('.')[0]}")
        # axes[1].imshow(entropy_measure, cmap=plt.cm.gray)
        # axes[1].set_title('entropy')
        #
        # for a in axes:
        #     a.axis('off')
        # plt.tight_layout()
        # plt.show()

        fig, ax = plt.subplots(ncols=4, figsize=(17, 5))
        ax[0].imshow(image, cmap=plt.cm.gray)
        ax[0].set_title('Original')
        ax[0].axis('off')
        ax[1].imshow(entropy_measure, cmap=plt.cm.gray)
        ax[1].set_title('Local Entropy')
        ax[1].axis('off')
        ax[2].imshow(regions)
        ax[2].set_title('Otsu thresholding')
        ax[2].axis('off')
        ax[3].imshow(eruption_mask, cmap=plt.cm.gray)
        ax[3].set_title('Masked Eruption')
        ax[3].axis('off')
        plt.tight_layout()
        if save_plots:
            plt.savefig(f"../figures/{os.path.basename(filename).split('.')[0]}_entropy_{disk_size}.png", dpi=300)
        plt.show()

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(eruption_mask, cmap='gray')
        plt.title('Grayscale Image')
        plt.axis('off')

        # Plot the histogram
        plt.subplot(1, 2, 2)
        plt.hist(eruption_mask[eruption_mask > 0].flatten(), range=[0, 256], color='gray', alpha=0.7)
        plt.title('Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    # subtract the mean before FFT
    corrected_mask = eruption_mask - np.mean(eruption_mask)

    # FFT using Welch Method for X and Y separately
    frequencies_x, psd_x = welch(corrected_mask, axis=0,
                                 window='hann', nperseg=128, scaling=welch_scaling)

    # Compute the power spectrum along the Y axis
    frequencies_y, psd_y = welch(corrected_mask, axis=1,
                                 window='hann', nperseg=128, scaling=welch_scaling)

    return {'gray_image': gray_image, 'entropy': entropy_measure,
            'regions': regions, 'mask': eruption_mask,
            'power_spectrum': [frequencies_x, frequencies_y, psd_x, psd_y]}


volcano_log = {'Grimsvötn': None, 'Eyjafjallajökull': None}
for eruption, image_file in zip(volcano_log.keys(), filenames):
    volcano_log[eruption] = analyze_eruption(image_file, disk_size=13)

# plot the power spectral density along the X axis
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(10, 5))
for eruption in volcano_log.keys():
    ax[0].semilogx(volcano_log[eruption]['power_spectrum'][0],
               volcano_log[eruption]['power_spectrum'][2].mean(axis=1),
               label=eruption, linestyle='dotted')
    ax[0].set_title('Power Spectrum (X axis)')
    ax[0].set_ylabel('Power')
    ax[0].grid(True)
    ax[1].semilogx(volcano_log[eruption]['power_spectrum'][0],
               volcano_log[eruption]['power_spectrum'][3].mean(axis=0).T,
               label=eruption, linestyle='dotted')
    ax[1].set_title('Power Spectrum (Y axis)')
    ax[1].set_xlabel('Frequency')
    ax[1].set_ylabel('Power')
    ax[1].grid(True)
    plt.suptitle(f"Welch Spectra")
    plt.tight_layout()
    plt.legend(loc='upper right')
plt.show()
