# import cython
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import seaborn as sns

# sns.set_context("talk")
sns.set_theme(style="white")
# sns.color_palette("Paired")
# plt.rcParams['text.usetex'] = True

# steady-state volcanic plume model (MTT 1956)
# demonstrates role of entrainment velocity parametrization

# todo: fix normalizations
volume_fraction = 0.5  # of gas
density_air = 1.225  # kg/m**3
g0 = 9.81  # m/s/s
Rg0 = 461.  # J/kgK
Rg = Rg0
Ra = 287.
# Girault et al. 2014
# Rg = Ra + (Rg0 - Ra) * (1-volume_fraction)volume_fraction*(volume_fraction/(1 - volume_fraction))
specific_volume = (1 - volume_fraction) / density_air + volume_fraction * Rg * 300. / 100e3
density_plume = 1 / specific_volume
reduced_gravity = (density_air - density_plume) / density_air * g0
w0 = 0.01  # m/s
v0 = 0.0  # m/s

# depends strongly on plume mixture density
f0 = reduced_gravity * w0  # m**4/s**3
# f0 = 1.

altitude_span = np.linspace(0, 100., 100)  # meters (hydrostatic scale height)
initial_conditions = np.array([w0, v0, f0])  # upward velocity, horizontal velocity, buoyancy flux (heat flux)
entrainment_coefficients = [0.01, 0.07, 0.11]  # , 0.2]  # , 0.8, 0.99]  # MTT experimental, constant
buoyancy_with_height = False


def density_hydrostatic(z, reference_density=1.225, hydrostatic_scale_height=8.4e3):
    # hydrostatic balance and ideal gas law
    return reference_density * np.exp(-z / hydrostatic_scale_height)


def dy_dz(height, x, mtt_entrainment=0.11, f_z=False):
    # system of coupled ODEs for MTT plume model
    # X = [vertical velocity, radial velocity, buoyancy flux]
    # print(f"current solution: {x}")
    if f_z:
        buoyancy_frequency = -g0 / density_air * density_hydrostatic(height, hydrostatic_scale_height=1.)
    else:
        buoyancy_frequency = 0.01

    # todo: generate (rho_0 - rho) for plotting NBH
    return np.array([2 * mtt_entrainment * x[1],  # ** (1 / 4),  # dw/dz
                     4 * x[2] * x[0],  # dv/dz
                     -2 * buoyancy_frequency * x[0]])  # df*/dz buoyancy_frequency already squared
    # return np.array([x[1],  # dw/dz
    #                  x[2] * x[0],  # dv/dz
    #                  -x[0]])  # df*/dz buoyancy_frequency already squared


# solve ODE system for velocity and buoyancy flux
solutions = []
for entrainment_coefficient in entrainment_coefficients:
    solution = solve_ivp(
        dy_dz, t_span=[altitude_span[0], altitude_span[-1]],
        # t_eval=altitude_span,
        y0=initial_conditions,
        args=[entrainment_coefficient, buoyancy_with_height], method='LSODA'
    )
    solutions.append(solution)

# plotting
fig, ax = plt.subplots(1, 3, sharey='all')
for jj, solution in enumerate(solutions):
    for ii in range(solution.y.shape[0]):
        ax[ii].plot(solution.y.T[:, ii], solution.t, label=entrainment_coefficients[jj])

ax[0].set_title("vertical\nvelocity")
ax[1].set_title("radial\nvelocity")
ax[2].set_title("buoyancy\nflux")
ax[0].set_ylabel("altitude (km)")
plt.legend(title="entrainment\ncoefficients", bbox_to_anchor=(2.25, 0.01), loc="lower right")
plt.subplots_adjust(right=0.8)
plt.suptitle("MTT plume model")
plt.tight_layout()
# plt.savefig(
#     f"../figures/mtt_plume_model_delta_F_{buoyancy_with_height}_volumefraction_{volume_fraction}_v5.png", dpi=300
# )
plt.show()
#
# heights = np.linspace(0., 3e4, 100)
# buoyancy_frequencies = -g0 / density_air * density_hydrostatic(heights)
# plt.plot(buoyancy_frequencies, heights)
# plt.title("buoyancy frequency with hydrostatic density decrease")
# plt.xlabel(r'$N^2$')
# plt.ylabel("Altitude (m)")
# plt.show()
