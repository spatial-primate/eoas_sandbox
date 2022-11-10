import numpy as np
from solvers.plotters import plot_integrator_results

# model configuration
title_string = "test_no_save_figure"  # "equal zones: steady-state without couplings"
filename_string = "test_no_save_figure"  # "steadystate_no_coupling_equal_zones_v2"
save_figure = False
initial_temperatures = 288 * np.ones((6,))  # avg temp today: 288 K
coefficients = np.ones((6,))
volcano_model = [None]  # can iterate over [None]
solvers = ['LSODA']  # 'DOP853', cut-paste if desired
compute_couplings = True
# todo: unequal zone selection


def main():
    plot_integrator_results(
        title_string, filename_string,
        args=(
            initial_temperatures, coefficients,
            compute_couplings,
            volcano_model, solvers, save_figure
        )
    )
    return


if __name__ == "__main__":
    main()
