import os
from scipy.signal import welch
from scipy.signal.windows import tukey
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")


# Retrievals from particle plume experiments


class ParticlePlume:
    def __init__(self, trial, multiphase, diameter, mixture_density, cloud_height, overshoot_height):
                 # , video_file):
        # todo: parse video files and do PSD analysis
        # self.video_name = os.path.abspath(video_file)
        self.overshoot_height = overshoot_height
        self.cloud_height = cloud_height
        self.trial = trial
        self.multiphase = multiphase
        self.eddy_diameter = None  # from measurements (O(0.5 cm))
        self.plume_velocity = None  # from measurements
        self.entrainment_coefficient = 0.11  # for single phase plume
        self.outlet_diameter = 1.  # millimeters
        self.shape_factor = 1.  # [-]
        self.fluid_viscosity = 1e-3  # Pa-s
        self.diameter = diameter  # meters
        self.particle_density = 2500.  # kg/m3
        self.gravity = 9.81  # kg m/s2
        self.ambient_density = 1005.
        self.ambient_density_step = 1020.  # kg/m3
        self.mixture_density = mixture_density  # kg/m3
        self.particle_concentration = None  # g/L (cara)
        self.outlet_area = np.pi * (self.outlet_diameter / 2) ** 2
        self.discharge_rate = (
            self._calculate_discharge_rate())  # m3 / s  (== 1 cm^3/s for particle plume); 2x for single-phase plume
        self.settling_velocity = (self.diameter ** 2 * self.gravity *
                                  (self.particle_density - self.ambient_density) / (18 * self.fluid_viscosity))  # m/s
        self.initial_buoyancy_flux = self._calculate_source_buoyancy_flux()
        self.buoyancy_frequency = np.sqrt(-self.gravity / self.ambient_density *  # z increasing down
                                          -(self.ambient_density_step - self.ambient_density) / 15e-2)
        self.plume_characteristic_velocity = (self.initial_buoyancy_flux * self.buoyancy_frequency) ** (1 / 4)
        self.h_star = self.overshoot_height / self.cloud_height
        self.delta_h = self.overshoot_height - self.cloud_height
        # todo: v_s / self.plume_characteristic_velocity

    def calculate_mixture_density(self):
        # note: probably measured/calculated experimentally
        # todo: concentration either via volume fraction or solid mass per liter
        return

    def _calculate_discharge_rate(self):
        if self.multiphase:
            return 1e-6  # cm^3/s
        else:
            return 90e-6  # todo: this is not scaling correctly

    def calculate_initial_velocity(self):
        return self.discharge_rate / self.outlet_area

    def _calculate_source_buoyancy_flux(self):
        return (self.mixture_density - self.ambient_density) / self.ambient_density * self.gravity * self.discharge_rate

    def particle_inertial_response_time(self):
        # tau_p jessop and jellinek
        return (self.particle_density * self.diameter ** 2) / (18 * self.fluid_viscosity * self.shape_factor)

    def eddy_overturn_time(self):
        # tau_f jessop and jellinek
        # eddy diameter from measurements
        # plume velocity from measurements
        eddy_velocity = self.entrainment_coefficient * self.plume_velocity
        return self.eddy_diameter / eddy_velocity  # seconds

    def stokes_number(self):
        return self.particle_inertial_response_time() / self.eddy_overturn_time()

    def process_video_spectra(self):
        return
        # title = 'Eruption Spectra'
        # eruption_spectra = []
        # eruption_names = []
        # # for filename in filenames:
        # # stage = 'End'  # beginning, middle, end
        # if 'eyja' in self.video_name:
        #     eruption_name = 'Eyja End'
        # elif 'image' in self.video_name:
        #     eruption_name = os.path.basename(self.video_name).split('.')[0]
        # else:
        #     if 'middle' in self.video_name:
        #         eruption_name = 'Grimsvötn Middle'
        #     else:
        #         eruption_name = 'Grimsvötn End'
        # eruption_names.append(eruption_name)
        # # window tapering and high pass filter
        # tukey_alpha = 0.50
        # filter_order = 5
        # critical_frequency = 33
        #
        # image = PIL.Image.open(filename)
        #
        # gray_image = image.convert('L')
        # gray_image = np.array(gray_image)
        #
        # if eruption_name == 'Eyja':
        #     gray_section_vertical = gray_image[:, gray_image.shape[0] // 2 - 30:gray_image.shape[0] // 2 + 80]
        # else:
        #     # ideas:
        #     # todo: smooth input stripe to avoid ringing
        #     # todo: filter sectional wavenumbers to increase power in > 10**(-1)
        #     # todo: maybe sum up 1-pixel slices to boost powah
        #     # todo: robust statistics to define noise-not noise
        #     # todo: hilbert transform because billows grow
        #     # todo: correct for pixel size (how?)
        #     # todo: pairing TGSD
        #     # eyja refs: magnusson et al 2010/12? Gudmundsson et al 2012
        #     # grims: prata separation of ash and so2; ash clouds at different heights --> H*
        #     gray_section_vertical = gray_image[:, gray_image.shape[0] // 2 - 30:gray_image.shape[0] // 2 + 30]
        #
        # gray_profile_x = np.mean(gray_section_vertical, axis=1)
        #
        # sos = signal.butter(filter_order, critical_frequency, btype='hp', fs=1000, output='sos')
        # filtered = signal.sosfilt(sos, gray_profile_x)
        #
        # gray_profile_x_no_mean = filtered - np.mean(filtered)
        # gray_profile_y = np.mean(gray_section_vertical, axis=0)
        #
        # # using the low-passed signal
        # d_signal = np.diff(filtered)
        #
        # # window
        # nx_d = d_signal.size  # get length of the differenced dataset
        # window = tukey(nx_d, alpha=tukey_alpha)  # hanning window
        #
        # d_sig_new = (d_signal - np.mean(d_signal)) * window
        #
        # # tapering
        # window = tukey(gray_profile_x_no_mean.size, alpha=tukey_alpha)  # tukey window
        # sig_no_sec_w = gray_profile_x_no_mean * window
        #
        # print(f"Profile x: {gray_profile_x.shape}", f"Profile y: {gray_profile_y.shape}",
        #       "Original image shape: ", gray_image.shape)
        #
        # # fig, ax = plt.subplots(nrows=2, sharey='row', figsize=(20, 10))
        # # ax2 = ax[0].twinx()
        # # lns1 = ax[0].plot(gray_profile_x_no_mean, label="Original")
        # # lns2 = ax[0].plot(sig_no_sec_w, label="Hanning, No Mean")
        # # lns3 = ax2.plot(d_sig_new, label="Tukey, Differenced", color='g')
        # # ax[0].set_title(f"{eruption_name} {stage} Intensity Profile: Eruption Centerline")
        # # ax[0].set_xlabel("x pixels")
        # # ax[0].set_ylabel("intensity")
        # # lines = lns1 + lns2 + lns3
        # # labels = [line.get_label() for line in lines]
        # # ax[0].legend(lines, labels, loc='upper right')
        #
        # # for the differenced
        # frequencies_x, psd_x = welch(d_sig_new, axis=0,
        #                              window='tukey', nperseg=128, scaling='density')
        # for the demeaned
        # psd_x_hilbert = hilbert(d_sig_new, axis=0)

        # for the damaged
        # psd_x_smoothed = sliding_window_view(psd_x, 3).mean(axis=-1)
        #
        # eruption_spectra.append(psd_x_smoothed)
        #
        # # ax[1].semilogx(frequencies_x,
        # #                np.log(psd_x / psd_x.max()),
        # #                label='tukey differenced', linestyle='solid')
        #
        # fig, ax = plt.subplots(nrows=1, sharey='row', figsize=(20, 10))
        # for spectrum, eruption_name in zip(eruption_spectra, eruption_names):
        #     ax.semilogx(frequencies_x[1:-1],
        #                 np.log(spectrum / spectrum.max()),
        #                 label=f'{eruption_name}', linestyle='solid')
        #     print(f"lowest max frequency = {np.argmax(spectrum)}")
        #
        # kolmogorov_linear = np.log(frequencies_x[4:-1]) * (-5 / 3)
        # obi_linear = np.log(frequencies_x[4:-1]) * (-11 / 3)
        # ax.semilogx(frequencies_x[4:-1], kolmogorov_linear - 3.33, 'r', linestyle='dotted', label='kolmogorov -5/3')
        # ax.semilogx(frequencies_x[4:-1], kolmogorov_linear - 8, 'r', linestyle='dotted')
        # ax.semilogx(frequencies_x[4:-1], obi_linear - 10, linestyle='dotted', label='obi -11/3')
        # # todo: how to interpret hilbert results
        # # ax[1].semilogx(
        # #     psd_x_hilbert / psd_x_hilbert.max(),
        # #     label='hilbert', linestyle='solid')
        #
        # ax.set_title(f'{eruption_name} Power Spectrum')
        # ax.set_xlabel("wavenumber")
        # ax.set_ylabel('power')
        # ax.grid(True)
        # ax.legend(loc='upper right')
        # ax.set_title(f"{title}")
        # plt.tight_layout()
        # # plt.savefig(f"../figures/{title} kolmogorov.png", dpi=300)
        # plt.show()
        #
        # return
