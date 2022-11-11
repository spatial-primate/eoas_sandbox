# from solvers.volcanoes import emissions
# import numpy as np
from data.constants import *


def dtemperature_dtime(time: float, temperatures: np.ndarray,
                       volcano_model=None, compute_couplings=False):
    """
    inputs:
    t: scalar time, y'know?
    temperature_array: zonally-averaged surface temperature (1 by 6)
    volcano_models: reference model of volcanic forcing
    compute_couplings: choice to compute without inter-zonal heat transfer

    returns:
    dtemperature_dtime: (1 by 6) vector of changes to temperature
    """
    # compute pre-factor on temperature term
    # sigma and tau are constant, epsilon usually just 1 (one)
    b = sigma * tau * epsilon

    noncoupling_prefactors = np.divide(1,
                                       densities * thermals * specific_heats
                                       )
    # Update values for fluxes from new temperatures:
    # todo: pickle the flux-in to save on computation
    # todo: time-dependent albedo_sky term will plug-in here:
    step1 = np.multiply(gamma.reshape(6), (1 - albedo_sky))
    step2 = np.multiply(step1, (1 - albedo_surface))
    flux_in = np.multiply(step2, S_0)
    flux_out = np.multiply(b, temperatures ** 4)

    if compute_couplings:  # include heat transfer between zones
        coupling_prefactors = np.divide(1,
                                        area.reshape(6) * densities * thermals * specific_heats
                                        )
        # transfer couplings between zones saved as csv, read in as constants
        couplings = np.dot(k_matrix, temperatures)

    else:  # ignore heat transfer between zones
        couplings = 0.0
        coupling_prefactors = 0.0

    # scale couplings by zonally-averaged prefactors
    couplings = np.multiply(coupling_prefactors, couplings)
    # finally, calculate change in temperature
    dtemperature = noncoupling_prefactors * (flux_in - flux_out) + couplings

    # todo: volcano-climate addition
    # if volcano_model is not None:
    #     emission_array = emissions([time], model=volcano_models)[0]
    # else:
    #     emission_array = 0

    # add forcing fluxes to first slot: the atmosphere box
    # dtemperature[0] += emission_array

    return dtemperature


# for debugging standalone
# time = 1.0
# temperature0 = np.zeros((6,))
# dtemperature_dtime(time, temperature0, compute_couplings=True)


def albedo_sky_stepwise(time: float, t_onset=False):
    a = ':)'
    return a
