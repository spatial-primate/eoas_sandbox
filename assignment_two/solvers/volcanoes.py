import numpy as np
import pandas as pd
from data.constants import albedo_snowball, albedo_not_snowball


# volcanic_clouds causes change to albedo_sky (increase) and adds trapping to flux_out

def volcanic_clouds(yr, model, onset=10, duration=2):
    """
    Function defining the changes produced by onset of volcanic activity

    model parameter is one of: "small", "medium", "large"
    onset in years
    duration in years
    """
    yr = np.asarray(yr)
    if model.lower() == "small":  # small amount of volcanoes, small increase in albedo
        albedo_sky = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) * 1.5

    elif model.lower() == "medium":
        albedo_sky = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) * 2

    elif model.lower() == "large":
        albedo_sky = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) * 4
    else:
        raise ValueError(
            "please check that volcano model is one of [None, 'small', 'medium', 'large']"
        )

    return albedo_sky


def snowball_surface(temperatures: np.ndarray):
    """
    Function deciding if the surface is ice or not and defining a surface albedo 

    inputs:
    temperatures: zonally-averaged surface temperature (1 by 6)

    returns:
    albedo_surface: (1 by 6) vector of surface reflectance based off temperature
    """
    # loop through the 6 temperatures, if < 0 deg C, all ice, otherwise 70% water 30% land
    temperatures = np.array([270, 270, 270, 280, 280, 280])
    ice = (temperatures < 273.15) * albedo_snowball  # boolean of whether temps are < 0
    not_ice = (temperatures > 273.15) * albedo_not_snowball

    return ice + not_ice
