import os.path

import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.integrate import solve_ivp
from solvers.odes import dtemperature_dtime
# from utilities.func_rk4 import rk4
# from utilities.euler_method import euler_method

# todo: create module for volcano models:
# from solvers.emissions import emissions


def plot_integrator_results(title_string, args: tuple):
    # Time interval for integration
    t_min = 0
    t_max = 1

    # Number of points in time array (used for rk4, euler)
    n = 10000
    # max_step = 10
    max_step = 1e-3  # dialed max_step down for nine boxes
    initial_temperatures, coefficients, volcano_models, solvers = args

    if os.path.isfile(coefficients):
        # todo: read coefficients from file
        pass

    # setup lists to collect results for plotting
    all_t = []
    all_temperatures = []
    delta_t = []
    plot_titles = []

    # Loop over each solver and emission type, and add the time array and mass array to all_t, all_M
    for solver in solvers:
        for model in volcano_models:
            if solver == "rk4":
                t_start = time.time()
                # t, temperature = rk4(fxy=dtemperature_dtime, x0=t_min, xf=t_max,
                #            y0=m_init, N=n, args=(k, model))
                t_end = time.time()
            elif solver == "euler":
                t_start = time.time()
                # t, temperature = euler_method(fxy=dtemperature_dtime, x0=t_min, xf=t_max,
                #                     y0=m_init, N=n, args=(k, model))
                t_end = time.time()
            else:
                t_start = time.time()
                sol = solve_ivp(fun=dtemperature_dtime, t_span=(t_min, t_max),
                                y0=initial_temperatures, method=solver, max_step=max_step,
                                args=(coefficients, model))
                t_end = time.time()
                t = sol.t
                temperature = sol.y.T

            all_t.append(t)
            all_temperatures.append(temperature)
            delta_t.append(t_end - t_start)  # The times taken to compute the integration
            plot_titles.append(solver + ", " + model.replace("_", " "))

    # Plotting
    # n_plots = len(plot_titles)
    # if n_plots > 1:
    #     fig, ax = plt.subplots()
    #     if n_plots <= 3:
    #         n_rows = 1
    #         n_cols = n_plots
    #     else:
    #         n_cols = 3
    #         n_rows = int(np.ceil(n_plots / n_cols))
    #     n_rows = 3
    #     n_cols = 1
    #     for ii in range(n_plots):
    #         plt.subplot(n_rows, n_cols, ii + 1)
    #         # plotting [0, 1] for just forced atmosphere and surface ocean water
    #         plt.plot(all_t[ii], all_temperatures[ii][:, [0, 1]])
    #         plt.xlabel('Time (yr)')
    #         plt.ylabel('Mass (Gt)')
    #         plt.title(plot_titles[ii] + ", delta t = " + "{:.2E}".format(delta_t[ii]) + "s")
    #
    # else:
    #     fig = plt.figure(figsize=(9, 4), dpi=150)
    #     plt.plot(all_t[0], all_temperatures[0][:, [0, 1]])
    #     plt.xlabel('Time (yr)')
    #     plt.ylabel('Mass (Gt)')
    #     plt.title(plot_titles[0] + ", delta t = " + "{:.2E}".format(delta_t[0]) + "s")
    #
    # plt.suptitle(title_string)
    # fig.legend(['atmosphere',
    #             'surface water',
    #             ], loc="center right")
    # plt.tight_layout()
    #
    # plt.show()
