import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from solvers.odes import dtemperature_dtime
import numpy as np


def plot_integrator_results(title_string, filename_string, args: tuple):
    # Time interval for integration
    years = 50
    t_min = 0
    t_max = years * 3.154e+7  # seconds in a year (multiplied by years)

    max_step = 50000  # dial max_step down for stiff problems
    initial_temperatures, coefficients, compute_couplings, \
        volcano_model, volcano_onset, volcano_duration, solvers, snowball_scenario, save_figure = args

    y0_reshaped = initial_temperatures.reshape(6)

    # # setup lists to collect results for plotting
    # all_t = []
    # all_temperatures = []
    # delta_t = []
    # # plot_titles = []

    # Loop over each solver and emission type, and add the time array and mass array to all_t, all_M
    for solver in solvers:
        t_start = time.time()
        sol = solve_ivp(fun=dtemperature_dtime, t_span=(t_min, t_max),
                        y0=y0_reshaped, method=solver, max_step=max_step,
                        args=(volcano_model, volcano_onset, volcano_duration,
                              compute_couplings, snowball_scenario)
                        )
        t_end = time.time()
        t = sol.t
        temperature = sol.y.T

        # all_t.append(t)
        # all_temperatures.append(temperature)
        # delta_t.append(t_end - t_start)  # the times taken to compute the integration
        # plot_titles.append(solver + ", " + model.replace("_", " "))

    # Plotting
    fig = plt.figure(figsize=(9, 4), dpi=150)

    colors = ["#000000", "#33658a", "#86bbd8", "#588157", "#f6ae2d", "#f26419"]

    for i in range(np.shape(temperature)[1]):
        plt.plot(t / 3.154e+7, temperature[:, 5 - i] - 273.15, color=colors[i])  # convert to years and celsius
    plt.xlabel('Time (y)')
    plt.ylabel(r"Temperature ($^\degree$C)")

    plt.suptitle(title_string)
    fig.legend(['60-90N', '30-60N', '0-30N', '0-30S', '30-60S', '60-90S'
                ], loc="center right")  # bbox_to_anchor=(1.04, 0.5), title='Zone')
    plt.tight_layout()

    fig_filename = "plots/" + filename_string + ".png"
    if save_figure:
        plt.savefig(fig_filename, bbox_inches='tight')
    plt.show()


# WIP
def plot_volcano_models():
    return
