from __future__ import division
import PIL
import numpy as np
from entropy_testing import analyze_eruption
from matplotlib import pyplot

import pycwt as wavelet
from pycwt.helpers import find

grimsvotn_filname = r'/Users/lukebrown/Downloads/grimsvotn_middle.png'
eyja_filename = r'/Users/lukebrown/Downloads/eyja_end.png'
image = PIL.Image.open(grimsvotn_filname)
gray_image = image.convert('L')
gray_image = np.array(gray_image)

title = 'Grims and Eyja Wavelet Analysis'
label = 'grayscale intensity'

pixel_size = 1.
N = gray_image.size
X = np.arange(0, N) * pixel_size

# normalize by subtracting the mean
gray_image_analysis = analyze_eruption(grimsvotn_filname, make_plots=False)
gray_image_subject = gray_image_analysis['mask']
gray_image_norm = gray_image_subject - np.mean(gray_image_subject)

eruption_column = gray_image_norm / np.std(gray_image_norm)

mother = wavelet.Morlet(6)
s0 = 2 * pixel_size  # Starting scale, in this case 2 * 0.25 years = 6 months
dj = 1 / 12  # Twelve sub-octaves per octaves
J = 7 / dj  # Seven powers of two with dj sub-octaves
alpha, _, _ = wavelet.ar1(gray_image_subject)

wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(eruption_column, pixel_size, dj, s0, J,
                                                      mother)
iwave = wavelet.icwt(wave, scales, pixel_size, dj, mother) * np.std(gray_image_norm)

power = (np.abs(wave)) ** 2
fft_power = np.abs(fft) ** 2
period = 1 / freqs

power /= scales[:, None]

signif, fft_theor = wavelet.significance(1.0, pixel_size, scales, 0, alpha,
                                         significance_level=0.95,
                                         wavelet=mother)
sig95 = np.ones([1, N]) * signif[:, None]
sig95 = power / sig95


