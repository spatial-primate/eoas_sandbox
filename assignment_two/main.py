import numpy as np
from solvers.plotters import plot_integrator_results

# os
title_string = r'Average temperature today with snowball albedo, coupling and large volcanic forcing'
filename_string = "coupling_n50_deg_snowball"
save_figure = False

# model configuration
solvers = ['LSODA']  # 'DOP853', cut-paste if desired
celsius0 = 0.01  # human-readable initial temperature: [change this one!]
initial_temperatures = (celsius0 + 273.15) * np.ones((6,))  # avg temp today: 15degC
coefficients = np.ones((6,))
compute_couplings = True
snowball_scenario = True
# volcano options
volcano_model = 'medium'  # can iterate over [None, 'small', 'medium', 'large']
volcano_onset = 1.5  # in years after start
volcano_duration = 1.5  # in years
# unequal zones
unequal_zones = False


def main():
    """
    generate plots of radiative energy balance for:
        1. uncoupled six-zone system with equal latitude spacing
        2. heat transfer coupled six-zone system with gulf stream
        3. volcanic forcing on climate
        4. modified definition for arctic circle (75 degrees)
    :return:
    figures showing time evolution of system
    """
    plot_integrator_results(
        title_string, filename_string,
        args=(
            initial_temperatures, coefficients,
            compute_couplings,
            volcano_model, volcano_onset, volcano_duration,
            solvers, snowball_scenario,
            unequal_zones,
            save_figure
        )
    )
    return


if __name__ == "__main__":
    main()
    print("... and Bob's your uncle")
