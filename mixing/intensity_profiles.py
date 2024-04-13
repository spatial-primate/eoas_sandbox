import os

import PIL
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from scipy.signal import welch, hilbert
from scipy.signal.windows import tukey

sns.set_theme(style="dark")


# don't worry about changing billow sizes... yet!
def main():
    # todo make this smarter
    grims_mid_file = r'/Users/lukebrown/Downloads/grimsvotn_middle.png'
    # grims_end_file = r'/Users/lukebrown/Downloads/grimsvotn_end.png'
    # eyja_end_file = r'/Users/lukebrown/Downloads/eyja_end.png'
    new_volcano_file = r"/Users/lukebrown/Downloads/image39.png"
    another_volcano_file = r"/Users/lukebrown/Downloads/image41.png"
    # yet_another_file = r"/Users/lukebrown/Downloads/image53.png"
    # yet_yet_another_file = r"/Users/lukebrown/Downloads/image28.png"
    # yet_yet_yet_file = r"/Users/lukebrown/Downloads/image32.png"
    # tonga_end_file = r"/Users/lukebrown/Downloads/image66.png"
    filenames = [
        grims_mid_file,
        # grims_end_file,
        # eyja_end_file,
        new_volcano_file,
        another_volcano_file,
        # yet_another_file,
        # yet_yet_another_file,
        # yet_yet_yet_file,
        # tonga_end_file
    ]

    title = 'Eruption Spectra'
    eruption_spectra = []
    eruption_names = []
    for filename in filenames:
        # stage = 'End'  # beginning, middle, end
        if 'eyja' in filename:
            eruption_name = 'Eyja End'
        elif 'image' in filename:
            eruption_name = os.path.basename(filename).split('.')[0]
        else:
            if 'middle' in filename:
                eruption_name = 'Grimsvötn Middle'
            else:
                eruption_name = 'Grimsvötn End'
        eruption_names.append(eruption_name)
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
            # ideas:
            # todo: smooth input stripe to avoid ringing
            # todo: filter sectional wavenumbers to increase power in > 10**(-1)
            # todo: maybe sum up 1-pixel slices to boost powah
            # todo: robust statistics to define noise-not noise
            # todo: hilbert transform because billows grow
            # todo: correct for pixel size (how?)
            # todo: pairing TGSD
            # eyja refs: magnusson et al 2010/12? Gudmundsson et al 2012
            # grims: prata separation of ash and so2; ash clouds at different heights --> H*
            gray_section_vertical = gray_image[:, gray_image.shape[0] // 2 - 30:gray_image.shape[0] // 2 + 30]

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

        # fig, ax = plt.subplots(nrows=2, sharey='row', figsize=(20, 10))
        # ax2 = ax[0].twinx()
        # lns1 = ax[0].plot(gray_profile_x_no_mean, label="Original")
        # lns2 = ax[0].plot(sig_no_sec_w, label="Hanning, No Mean")
        # lns3 = ax2.plot(d_sig_new, label="Tukey, Differenced", color='g')
        # ax[0].set_title(f"{eruption_name} {stage} Intensity Profile: Eruption Centerline")
        # ax[0].set_xlabel("x pixels")
        # ax[0].set_ylabel("intensity")
        # lines = lns1 + lns2 + lns3
        # labels = [line.get_label() for line in lines]
        # ax[0].legend(lines, labels, loc='upper right')

        # for the differenced
        frequencies_x, psd_x = welch(d_sig_new, axis=0,
                                     window='tukey', nperseg=128, scaling='density')
        # for the demeaned
        psd_x_hilbert = hilbert(d_sig_new, axis=0)

        # for the damaged
        psd_x_smoothed = sliding_window_view(psd_x, 3).mean(axis=-1)

        eruption_spectra.append(psd_x_smoothed)

    # ax[1].semilogx(frequencies_x,
    #                np.log(psd_x / psd_x.max()),
    #                label='tukey differenced', linestyle='solid')

    fig, ax = plt.subplots(nrows=1, sharey='row', figsize=(20, 10))
    for spectrum, eruption_name in zip(eruption_spectra, eruption_names):
        ax.semilogx(frequencies_x[1:-1],
                    np.log(spectrum / spectrum.max()),
                    label=f'{eruption_name}', linestyle='solid')
        print(f"lowest max frequency = {np.argmax(spectrum)}")

    kolmogorov_linear = np.log(frequencies_x[4:-1]) * (-5 / 3)
    obi_linear = np.log(frequencies_x[4:-1]) * (-11 / 3)
    ax.semilogx(frequencies_x[4:-1], kolmogorov_linear - 3.33, 'r', linestyle='dotted', label='kolmogorov -5/3')
    ax.semilogx(frequencies_x[4:-1], kolmogorov_linear - 8, 'r', linestyle='dotted')
    ax.semilogx(frequencies_x[4:-1], obi_linear - 10, linestyle='dotted', label='obi -11/3')
    # todo: how to interpret hilbert results
    # ax[1].semilogx(
    #     psd_x_hilbert / psd_x_hilbert.max(),
    #     label='hilbert', linestyle='solid')

    ax.set_title(f'{eruption_name} Power Spectrum')
    ax.set_xlabel("wavenumber")
    ax.set_ylabel('power')
    ax.grid(True)
    ax.legend(loc='upper right')
    ax.set_title(f"{title}")
    plt.tight_layout()
    # plt.savefig(f"../figures/{title} kolmogorov.png", dpi=300)
    plt.show()

    return


if __name__ == "__main__":
    main()
