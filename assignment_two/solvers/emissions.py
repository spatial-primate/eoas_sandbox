# todo: host volcano models here

import numpy as np
import pandas as pd

# model data stored in a csv
df = pd.read_csv("./data/model-data.csv")


def emissions(yr, model):
    """
    reusing assignment one as much as possible
    more general models than just emissions can go here
    """
    yr = np.asarray(yr)
    if model == "short_sine": # short wavelength sine wave forcing
        T = 0.1
        e = (20 / T) * (np.sin((2*np.pi*yr) / T))
    elif model == "long_sine":
        T = 100
        e = (20 / T) * (np.sin((2*np.pi*yr) / T)) # long wavelength sine wave forcing
    elif model == "short_exp":
        T = 0.1
        e = np.exp(-yr/2100) * (np.sin((2*np.pi*yr) / T))  # short wavelength exponential decay
    elif model == "long_exp":
        T = 100  # long wavelength
        e = np.exp(-yr/2100) * (np.sin((2*np.pi*yr) / T))  # long wavelength exponential decay
    else:
        # get model data from pandas dataframe
        t_yr = df[model + "_times"]
        e_GtC_yr = df[model + "_emissions"]
        # interpolate yr from model data
        e = np.interp(yr, t_yr, e_GtC_yr)

    return e
