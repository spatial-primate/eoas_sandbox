import numpy as np
from utilities.volcanoes import volcanoes_scenario


def dtemp_dt(time, temp_array, k, volcano_model=None):
    """
    t: float (scalar), time.
    T: size 6 array, temperature of boxes 1, 2, 3, 4, 5, 6.

    dtempdt: size 6 array, change in temperature for boxes 1, 2, 3, 4, 5, 6.
    """

    # Update values for fluxes from mass:
    flux_array = np.multiply(k, temp_array)

    flux_in = np.sum(flux_array, axis=1)  # row-wise for in-flux
    flux_out = np.sum(flux_array, axis=0)  # column-wise for out-flux

    dtempdt = flux_in - flux_out

    if volcanoes_model:
        volcano_array = volcanoes_scenario([time], model=volcanoes_model)[0]
    else:
        volcano_array = 0

    # add forcing fluxes to first slot: the atmosphere box
    dtempdt[0] += volcano_array

    return dtempdt
