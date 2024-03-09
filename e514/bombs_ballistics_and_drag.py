import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


# buoyancy and drag both fight inertia


def velocity_components(
        initial_velocity, timesteps,
        muzzle_angle=np.pi / 4, shape_drag=3.,
        density_rock=2.7e3, density_air=1.225,
        diameter_particle=0.1
):
    kappa = density_air / (density_rock * diameter_particle) * shape_drag
    u0 = initial_velocity * np.sin(muzzle_angle)
    w0 = initial_velocity * np.cos(muzzle_angle)
    u_component = u0 / (1 + kappa * u0 * timesteps)
    w_component = (np.sqrt(effective_gravity / kappa) *
                   np.arctan(w0 / np.sqrt(effective_gravity / kappa)) -
                   kappa * timesteps)
    return u_component, w_component


density_air = 1.225  # kg/m^3
density_rock = 2.7e3  # kg/m^3

gravity = 9.81  # m/s^2

diameter_particles = np.logspace(-2, 0., 5)  # 10 cm
shape_drags = 30.  # np.linspace(1., 100, 10)

effective_gravity = (density_rock - density_air) / density_rock * gravity

kappas = density_air / (density_rock * diameter_particles) * shape_drags

v_min = 0.
v_max = 700.

muzzle_angles = np.pi / 4  # np.linspace(np.pi / 4, np.pi / 2)
initial_mag_velocities = np.linspace(v_min, v_max)  # what order of magnitude of variation?

# plotting horizontal displacement for a range of initial velocities
u_initial = initial_mag_velocities * np.sin(muzzle_angles)
w_initial = initial_mag_velocities * np.cos(muzzle_angles)

fig, ax = plt.subplots(figsize=(13, 9))
l_maxes = []
for jj, kappa in enumerate(kappas):
    for ii in [0, ]:
        times = 1 / np.sqrt(effective_gravity * kappa) * (
                2 * np.arctan(w_initial / np.sqrt(effective_gravity / kappa)) - ii * np.pi)
        l_maxes.append(1 / kappa * np.log(1 + kappa * u_initial / np.sqrt(effective_gravity * kappa) * times))
    ax.plot(initial_mag_velocities, l_maxes[jj],
            label=f'{diameter_particles[jj]:.2} (m), {muzzle_angles:.2}')
    ax.title.set_text(f'horizontal displacement considering ballistics and aerodynamics')
    ax.set_xlabel('initial velocity (m/s)')
    ax.set_ylabel('maximum displacement (m)')
    ax.legend(title=f'particle diameter, muzzle angle')
plt.savefig("../figures/lmax_versus_v0_700_v2.png", dpi=300, bbox_inches='tight')
plt.show()

# z = 1 / kappa * (
#     np.log(
#         np.cos(
#             np.arctan(w_initial / np.sqrt(effective_gravity / kappa)) - np.sqrt(effective_gravity * kappa * times)
#         )
#     )
# ) - np.log(
#     np.cos(
#         np.arctan(w_initial / np.sqrt(effective_gravity / kappa))
#     )
# )
