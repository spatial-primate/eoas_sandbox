import numpy as np
from solvers.plotters import plot_integrator_results

# model configuration
title_string = r' Average/snowball earth, coupling'
filename_string = "coupling_15_deg_snowball"
initial_temperatures = (-200 + 273)  * np.ones((6,))  # avg temp today: 15degC
coefficients = np.ones((6,))
solvers = ['LSODA']  # 'DOP853', cut-paste if desired
compute_couplings = False
snowball_scenario = False

volcano_model = [None]  # can iterate over [None, 'small', 'medium', 'large']
volcano_onset = 10 # in years after start
volcano_duration =  2 #in years
# todo: unequal zone selection

def main():
    plot_integrator_results(title_string, filename_string,
                            args=(initial_temperatures, coefficients,
                                  compute_couplings,
                                  volcano_model, volcano_onset, volcano_duration,
                                   solvers,snowball_scenario
                                  )
                            )
    return


if __name__ == "__main__":
    main()
