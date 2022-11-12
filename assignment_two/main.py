import numpy as np
from solvers.plotters import plot_integrator_results

# model configuration
title_string = r'Average/snowball earth, coupling'
filename_string = "coupling_n50_deg_snowball"
celsius0 = 0  # human-readable initial temperature
initial_temperatures = (celsius0 + 273) * np.ones((6,))  # avg temp today: 15degC
coefficients = np.ones((6,))
solvers = ['LSODA']  # 'DOP853', cut-paste if desired
compute_couplings = True
snowball_scenario = False
# volcano options
volcano_model = 'large'  # can iterate over [None, 'small', 'medium', 'large']
volcano_onset = 0  # in years after start
volcano_duration = 15  # in years


# todo: unequal zone selection

def main():
    """
    generate plots of radiative energy balance for:
        1. uncoupled six-zone system with equal latitude spacing
        2. heat transfer coupled six-zone system with gulf stream
        3. volcanic forcing on climate
        4. modified definition for arctic circle (75 degrees)
    :return:
    figures showing time-dependent system
    """
    plot_integrator_results(title_string, filename_string,
                            args=(
                                initial_temperatures, coefficients,
                                compute_couplings,
                                volcano_model, volcano_onset, volcano_duration,
                                solvers, snowball_scenario
                                )
                            )
    return


if __name__ == "__main__":
    main()
    print("... and Bob's your uncle")
