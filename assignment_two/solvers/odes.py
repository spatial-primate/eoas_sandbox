import numpy as np
from utilities.emissions import emissions


def dtemperature_dtime(time: float, temperatures: np.ndarray,
                       gamma: np.ndarray, sigma, tau, epsilon,
                       albedo_sky: np.ndarray, albedo_surface: np.ndarray,
                       solar=None, volcano_model=None, kwargs=None):
    """
    t: float (scalar), time.
    temperature_array: zonally-averaged surface temperature (1 by 6)

    dtemperature_dtime: (1 by 6) vector of changes to temperature
    """
    # compute pre-factor on temperature
    # sigma is constant
    k = sigma * tau * epsilon

    # todo: compute coupling terms

    # Update values for fluxes from mass:
    flux_in = np.multiply(gamma, (1 - albedo_sky), (1 - albedo_surface), solar)  # add coupling terms
    flux_out = np.multiply(k, temperatures ** 4)

    dtemperature = flux_in - flux_out

    # todo: volcano-climate addition
    if volcano_model:
        emission_array = emissions([time], model=volcano_model)[0]
    else:
        emission_array = 0

    # add forcing fluxes to first slot: the atmosphere box
    dtemperature[0] += emission_array

    return dtemperature
