import numpy as np
from utilities.emissions import emissions
from data.constants import *


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
    #changed name to b as k is taken for transfer coefficients
    b = sigma * tau * epsilon 

    # compute coupling terms (Luke - feel free to do this with your nice matrices if you prefer)
    coup_pre = np.divide(1,(area*densities*thermals*specific_heats*)) # not sure if this will find the constants?...
    coup_pos = np.ndarray((1,6))
    coup_neg = np.ndarray((1,6))
    for i in range(0,6): 
            coup_pos[i] = k[i]*L[i]*(temperatures[i+1] - temperatures[i])
            coup_neg[i+1] = k[i]*L[i]*(temperatures[i+1] - temperatures[i])
            
    couplings = coup_pre * (coup_pos - coup_neg)

    # Update values for fluxes from mass:
    flux_in = np.multiply(gamma, (1 - albedo_sky), (1 - albedo_surface), solar)
    flux_out = np.multiply(b, temperatures ** 4)

    dtemperature = flux_in - flux_out + couplings 

    # todo: volcano-climate addition
    if volcano_model:
        emission_array = emissions([time], model=volcano_model)[0]
    else:
        emission_array = 0

    # add forcing fluxes to first slot: the atmosphere box
    dtemperature[0] += emission_array

    return dtemperature
