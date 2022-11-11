from solvers.volcanoes import volcanic_clouds
from solvers.volcanoes import snowball_surface
from data.constants import *
# import numpy as np


def dtemperature_dtime(time: float, temperatures: np.ndarray,
                       volcano_model=None, volcano_onset = 10, volcano_duration = 2,
                        compute_couplings=False, snowball_scenario = False):
    """
    inputs:
    t: scalar time, y'know?
    temperatures: zonally-averaged surface temperature (1 by 6)
    volcano_model: reference model of volcanic forcing
    volcano_onset: start time for volcano (years)
    volcano_duration: lenth of volcano effect (years)
    compute_couplings: choice to compute without inter-zonal heat transfer
    snowball_scenario: choice to flip to version of model where T<0 = ice albedo

    returns:
    dtemperature_dtime: (1 by 6) vector of changes to temperature
    """

    # compute pre-factor on temperature term
    # sigma and tau are constant, epsilon usually just 1 (one)
    b = sigma * tau * epsilon

    noncoupling_prefactors = np.divide(1,
                                       densities * thermals * specific_heats)

    # include heat transfer between zones
    if compute_couplings:
        coupling_prefactors = np.divide(1,
                                        area.reshape(6) * densities * thermals * specific_heats)
        couplings_plus = np.zeros((6,))
        couplings_minus = np.zeros((6,))

        # transfer couplings between zones saved as csv read in as constants
        couplings = np.dot(k_matrix, temperatures)

        # # do the north and south separately cuz they're built different
        # couplings_plus[0] = k[0] * L[0] * (temperatures[1] - temperatures[0])
        # couplings_minus[-1] = k[-2] * L[-2] * (temperatures[-1] - temperatures[-2])
        # # now compute the inner zones:
        # for zone in range(1, len(k) - 1):
        #     couplings_plus[zone] = k[zone] * L[zone] * (temperatures[zone + 1] - temperatures[zone])
        #     couplings_minus[zone] = k[zone - 1] * L[zone - 1] * (temperatures[zone] - temperatures[zone - 1])
        # couplings = couplings_plus - couplings_minus
    else:  # ignore heat transfer between zones
        couplings = 0.0
        coupling_prefactors = 0.0

    # scale couplings by zonally-averaged prefactors
    couplings = np.multiply(coupling_prefactors, couplings)


    # # volcano-climate interaction by adding volcanic clouds to change albedo_sky
    if volcano_model:
        while (time >= (volcano_onset * 3.154e+7)) and (time <= ((volcano_onset + volcano_duration) * 3.154e+7)):
            albedo_sky = volcanic_clouds(time, model=volcano_model)


    # snowball-earth melting by changing surface albedo from ice to land/water
    if snowball_scenario:
        albedo_surface = snowball_surface(temperatures)


    # Update values for fluxes from new temperatures:
    # todo: time-dependent albedo_sky term will plug-in here:
    step1 = np.multiply(gamma.reshape(6), (1 - albedo_sky))
    step2 = np.multiply(step1, (1 - albedo_surface))
    flux_in = np.multiply(step2, S_0)
    flux_out = (np.multiply(b, temperatures ** 4))*(1-albedo_sky) #added a term to reflect back to earth

    dtemperature = noncoupling_prefactors * (flux_in - flux_out) + couplings

    return dtemperature


# for debugging standalone
# time = 1.0
# temperature0 = np.zeros((6,))
# dtemperature_dtime(time, temperature0, compute_couplings=True)