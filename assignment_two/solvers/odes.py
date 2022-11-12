from solvers.volcanoes import volcanic_clouds
from solvers.volcanoes import snowball_surface
from data.constants import *


def dtemperature_dtime(time: float, temperatures: np.ndarray,
                       volcano_model=None, volcano_onset=10, volcano_duration=2,
                       compute_couplings=False, snowball_scenario=False):
    """
    inputs:
    t: scalar time, y'know?
    temperatures: zonally-averaged surface temperature (1 by 6)
    volcano_model: reference model of volcanic forcing
    volcano_onset: start time for volcano (years)
    volcano_duration: length of volcano effect (years)
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

    if compute_couplings:  # include heat transfer between zones
        coupling_prefactors = np.divide(1,
                                        area.reshape(6) * densities * thermals * specific_heats)

        # transfer couplings between zones saved as csv read in as constants
        couplings = np.dot(k_matrix, temperatures)

    else:  # ignore heat transfer between zones
        couplings = 0.0
        coupling_prefactors = 0.0

    # scale couplings by zonally-averaged prefactors
    couplings = np.multiply(coupling_prefactors, couplings)

    flux = calculate_flux_terms(time, temperatures,
                                b, GAMMA,
                                volcano_model,
                                volcano_onset,
                                volcano_duration,
                                snowball_scenario
                                )
    dtemperature = noncoupling_prefactors * flux + couplings

    return dtemperature


# for debugging standalone
# time = 1.0
# temperature0 = np.zeros((6,))
# dtemperature_dtime(time, temperature0, compute_couplings=True)


def calculate_flux_terms(time, temperatures,
                         b,
                         gamma=GAMMA,
                         volcano_model=None,
                         volcano_onset=10,
                         volcano_duration=2,
                         snowball_scenario=False
                         ):

    # volcano-climate interaction by adding volcanic clouds to change albedo_sky
    if volcano_model is not None\
            and (volcano_onset * 3.154e+7) <= time <= ((volcano_onset + volcano_duration) * 3.154e+7):
        albedo_sky = volcanic_clouds(time, model=volcano_model)
    else:
        albedo_sky = ALBEDO_SKY

    # snowball-earth melting by changing surface albedo from ice to land/water
    if snowball_scenario:
        albedo_surface = snowball_surface(temperatures)
    else:
        albedo_surface = ALBEDO_SURFACE

    # Update values for fluxes from new temperatures:
    step1 = np.multiply(gamma.reshape(6), (1 - albedo_sky))
    step2 = np.multiply(step1, (1 - albedo_surface))
    flux_in = np.multiply(step2, S_0)
    flux_out = np.multiply(b, temperatures ** 4)
    if volcano_model is not None:
        # added a term to reflect back to earth
        flux_out = np.multiply(flux_out, (1 - albedo_sky))

    return flux_in - flux_out
