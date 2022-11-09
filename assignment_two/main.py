import numpy as np
from solvers.plotters import plot_integrator_results

title_string = "steady-state, no couplings"
initial_temperatures = 288 * np.ones([6, 1])  # avg temp today: 288 K
coefficients = np.ones([6, 1])
volcano_model = [None]  # can iterate over [None]
solvers = ['DOP853']  # 'LSODA', cut-paste if desired
compute_couplings = True


def main():
    plot_integrator_results(title_string,
                            args=(initial_temperatures, coefficients,
                                  compute_couplings,
                                  volcano_model, solvers))
    return


if __name__ == "__main__":
    main()
