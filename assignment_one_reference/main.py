import argparse
import os

import numpy as np
from utilities.plotters import plot_integrator_results, plot_emissions_models

# Plotting variables
n_boxes = 9
'''
possible emissions_models: "short_sine", "long_sine", "short_exp", "long_exp", "IPCC-A2", "GFDL-ESM2G_esmrcp85",
   "CNRM-ESM2-1_esm-ssp585", "MPI-ESM1-2-LR_esm-ssp585", "UKESM1-0-LL_esm-ssp585"
'''
emissions_models = ["IPCC-A2", "GFDL-ESM2G_esmrcp85", "UKESM1-0-LL_esm-ssp585"]
integrators = ["LSODA"]

if n_boxes == 4:
    title_string = "four box model"
    print("computing four box model")
    initial_masses = np.genfromtxt('data/four_initial_masses.csv',
                                   delimiter=',')
    initial_fluxes = np.genfromtxt('data/four_box_fluxes.csv',
                                   delimiter=','
                                   )
else:
    title_string = "nine box model"
    print("computing nine box model")
    initial_masses = np.genfromtxt('data/initial_masses.csv',
                                   delimiter=',')
    initial_fluxes = np.genfromtxt('data/nine_box_fluxes.csv',
                                   delimiter=','
                                   )

# Get rate coefficients from steady-state flux and initial mass
k = np.divide(initial_fluxes, initial_masses)


def main():
    plot_integrator_results(title_string, args=(initial_masses, k, emissions_models, integrators))
    # plot_emissions_models()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="plot time-dependent carbon cycle scenarios from IPCC emissions")
    # parser.add_argument("n_boxes",
    #                     help='choose how many boxes for carbon box model; default is 4',
    #                     type=int)
    # parser.add_argument("initial_masses", type=os.PathLike)
    # parser.add_argument("initial_fluxes", type=os.PathLike)
    # # todo: option to add either sinusoidal or exponential damp forcing
    # parser.add_argument("--add_flux_C", action='store_true', type=bool)
    # parser.add_argument("--add_emissions", action='store_true', type=bool)
    # parser.parse_args()
    main()
