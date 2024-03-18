import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme('paper')

diffusion_so2 = 1.089e-5  # m2/s
diffusion_so2 = np.mean([diffusion_so2, 1.122e-5, 1.087e-5])

ash_density = 2.7e3
diameters_ash = np.logspace(-8, -5, 100)  # m

flow_speed = 300.  # m/s
dynamic_viscosity_air = 18.03e-6  # Pa-s
kinematic_viscosity_air = 14.88e-6  # m2/s

particle_reynolds_numbers = flow_speed * diameters_ash/kinematic_viscosity_air

time_scale = np.nan * np.ones_like(diameters_ash)

for ii, number in enumerate(particle_reynolds_numbers):
    if number < 10:
        time_scale[ii] = ash_density * diameters_ash[ii]**2 / (18 * dynamic_viscosity_air)

stokes_numbers = time_scale * flow_speed / np.sqrt(kinematic_viscosity_air * time_scale)

# plt.plot(diameters_ash, particle_reynolds_numbers)
# plt.xlabel('Diameter')
# plt.ylabel('Particle Reynolds number')
# plt.show()
#
# plt.plot(diameters_ash, stokes_numbers)
# plt.xlabel('Diameter')
# plt.ylabel('Stokes Number')
# plt.show()

# timescale for diffusion

diffusion_times = diameters_ash**2 / diffusion_so2

plt.plot(diameters_ash, diffusion_times, label='diffusion times (s)')
plt.legend()
plt.xlabel('ash diameter (m)')
plt.show()

# timescale for advection (not long)

tropopause_heights = np.array([8., 11., 16.]) * 1e3  # meters

advection_times = None

# timescale for mixing (large eddy size + velocity gradient)
