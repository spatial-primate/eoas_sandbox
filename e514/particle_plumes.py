import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")


# Retrievals from particle plume experiments


class ParticlePlume:
    def __init__(self, trial, multiphase, diameter, mixture_density):
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
