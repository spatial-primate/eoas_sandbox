import numpy as np
from solvers.plotters import plot_integrator_results

# model configuration
title_string = r'Modern Earth, coupling, 20% surface radiation trapped' #, $\alpha _{sky}$ = 0.2'
filename_string = "coupling_sky_traps_20_perc_15_deg"
initial_temperatures = (15 + 273)  * np.ones((6,))  # avg temp today: 15degC
coefficients = np.ones((6,))
volcano_model = [None]  # can iterate over [None]
solvers = ['LSODA']  # 'DOP853', cut-paste if desired
compute_couplings = True
sky_reflects = False
# todo: unequal zone selection

def main():
    plot_integrator_results(title_string, filename_string,
                            args=(initial_temperatures, coefficients,
                                  compute_couplings,
                                  volcano_model, solvers,
                                  sky_reflects
                                  )
                            )
    return


if __name__ == "__main__":
    main()
