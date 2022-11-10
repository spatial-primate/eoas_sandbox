import numpy as np
from solvers.plotters import plot_integrator_results

# model configuration
title_string = "equal zones: -70$\deg$C start"
filename_string = "steadystate_coupling_equal_zones_neg70"
initial_temperatures = (-70 + 273)  * np.ones((6,))  # avg temp today: 15degC
coefficients = np.ones((6,))
volcano_model = [None]  # can iterate over [None]
solvers = ['LSODA']  # 'DOP853', cut-paste if desired
compute_couplings = True
# todo: unequal zone selection

def main():
    plot_integrator_results(title_string, filename_string,
                            args=(initial_temperatures, coefficients,
                                  compute_couplings,
                                  volcano_model, solvers,
                                  )
                            )
    return


if __name__ == "__main__":
    main()
