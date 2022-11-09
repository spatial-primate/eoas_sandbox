import os.path

import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.integrate import solve_ivp
from solvers.odes import dtemperature_dtime
# from utilities.func_rk4 import rk4
# from utilities.euler_method import euler_method

# todo: create module for volcano models:
# from solvers.volcanoes import emissions


def plot_integrator_results(title_string, args: tuple):
    # Time interval for integration
    years = 5
    t_min = 0
    t_max = years * 3.154e+7  # seconds in a year (multiplied by years)

    # Number of points in time array (used for rk4, euler)
    # n = 10000
    # max_step = 10
    # todo: turn this down to understand whether it's affecting bad graphs
    max_step = 10  # dial max_step down for stiff problems
    initial_temperatures, coefficients, compute_couplings, volcano_models, solvers = args

    y0_reshaped = initial_temperatures.reshape(6)

    # setup lists to collect results for plotting
    all_t = []
    all_temperatures = []
    delta_t = []
    # plot_titles = []

    # Loop over each solver and emission type, and add the time array and mass array to all_t, all_M
    for solver in solvers:
        # for model in volcano_models:
        t_start = time.time()
        sol = solve_ivp(fun=dtemperature_dtime, t_span=(t_min, t_max),
                        y0=y0_reshaped, method=solver, max_step=max_step,
                        # args=(compute_couplings, model)
                        )
        t_end = time.time()
        t = sol.t
        temperature = sol.y.T

        all_t.append(t)
        all_temperatures.append(temperature)
        delta_t.append(t_end - t_start)  # the times taken to compute the integration
        # plot_titles.append(solver + ", " + model.replace("_", " "))

    # Plotting
    fig = plt.figure(figsize=(9, 4), dpi=150)
    plt.plot(all_t[0], all_temperatures[0])
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (K)')
    # plt.title(plot_titles[0] + ", delta t = " + "{:.2E}".format(delta_t[0]) + "s")

    plt.suptitle(title_string)
    # fig.legend(['steady-state solution',
    #             # 'volcanic forcing',
    #             ], loc="center right")
    plt.tight_layout()
    plt.savefig("./plots/steady_state_no_couplings.png")

    plt.show()


def plot_volcano_models():
    return