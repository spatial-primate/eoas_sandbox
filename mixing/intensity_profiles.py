import PIL
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from scipy.signal import welch, hilbert
from scipy.signal.windows import tukey

sns.set_theme(style="dark")


def main():
    # filename = r'/Users/lukebrown/Downloads/grimsvotn_middle.png'
    # filename = r'/Users/lukebrown/Downloads/grimsvotn_end.png'
    filename = r'/Users/lukebrown/Downloads/eyja_end.png'

    stage = 'End'  # beginning, middle, end
    if 'eyja' in filename:
        eruption_name = 'Eyja'
    else:
        eruption_name = 'Grimsvötn'

    # window tapering and high pass filter
    tukey_alpha = 0.50
    filter_order = 5
    critical_frequency = 33

    image = PIL.Image.open(filename)

    gray_image = image.convert('L')
    gray_image = np.array(gray_image)

    if eruption_name == 'Eyja':
        gray_section_vertical = gray_image[:, gray_image.shape[0] // 2 - 30:gray_image.shape[0] // 2 + 80]
    else:
        # todo: maybe sum up 1-pixel slices to boost powah
        # todo: robust statistics to define noise-not noise
        # todo: hilbert transform because billows grow
        # todo: correct for pixel size (how?)
        # todo: pairing TGSD
        # eyja refs: magnusson et al 2010/12? Gudmundsson et al 2012
        # grims: prata separation of ash and so2; ash clouds at different heights --> H*
        gray_section_vertical = gray_image[:, gray_image.shape[0] // 2 - 80:gray_image.shape[0] // 2 + 30]

    gray_profile_x = np.mean(gray_section_vertical, axis=1)

    sos = signal.butter(filter_order, critical_frequency, btype='hp', fs=1000, output='sos')
    filtered = signal.sosfilt(sos, gray_profile_x)

    gray_profile_x_no_mean = filtered - np.mean(filtered)
    gray_profile_y = np.mean(gray_section_vertical, axis=0)

    # using the low-passed signal
    d_signal = np.diff(filtered)

    # window
    nx_d = d_signal.size  # get length of the differenced dataset
    window = tukey(nx_d, alpha=tukey_alpha)  # hanning window

    d_sig_new = (d_signal - np.mean(d_signal)) * window

    # tapering
    window = tukey(gray_profile_x_no_mean.size, alpha=tukey_alpha)  # tukey window
    sig_no_sec_w = gray_profile_x_no_mean * window

    print(f"Profile x: {gray_profile_x.shape}", f"Profile y: {gray_profile_y.shape}",
          "Original image shape: ", gray_image.shape)

    fig, ax = plt.subplots(nrows=2, sharey='row', figsize=(20, 10))
    ax2 = ax[0].twinx()
    lns1 = ax[0].plot(gray_profile_x_no_mean, label="Original")
    lns2 = ax[0].plot(sig_no_sec_w, label="Hanning, No Mean")
    lns3 = ax2.plot(d_sig_new, label="Tukey, Differenced", color='g')
    ax[0].set_title(f"{eruption_name} {stage} Intensity Profile: Eruption Centerline")
    ax[0].set_xlabel("x pixels")
    ax[0].set_ylabel("intensity")
    lines = lns1 + lns2 + lns3
    labels = [line.get_label() for line in lines]
    ax[0].legend(lines, labels, loc='upper right')

    # for the differenced
    frequencies_x, psd_x = welch(d_sig_new, axis=0,
                                 window='tukey', nperseg=128, scaling='density')
    # for the demeaned
    psd_x_hilbert = hilbert(d_sig_new, axis=0)

    # for the damaged
    psd_x_smoothed = sliding_window_view(psd_x, 3).mean(axis=-1)

    ax[1].semilogx(frequencies_x,
                   psd_x / psd_x.max(),
                   label='tukey differenced', linestyle='solid')

    ax[1].semilogx(frequencies_x[1:-1],
                   psd_x_smoothed / psd_x_smoothed.max(),
                   label='smoothed', linestyle='solid')

    # todo: how to interpret hilbert results
    # ax[1].semilogx(
    #     psd_x_hilbert / psd_x_hilbert.max(),
    #     label='hilbert', linestyle='solid')

    ax[1].set_title(f'{eruption_name} {stage} Power Spectrum')
    ax[1].set_xlabel("wavenumber")
    ax[1].set_ylabel('power')
    ax[1].grid(True)
    ax[1].legend(loc='upper right')
    plt.tight_layout()
    # plt.savefig(f"../figures/{eruption_name}_{stage}_intensity_fourier_profiles_v3.png", dpi=300)
    plt.show()

    print(f"lowest max frequency = {np.argmax(psd_x_smoothed)}")
    return


if __name__ == "__main__":
    main()
