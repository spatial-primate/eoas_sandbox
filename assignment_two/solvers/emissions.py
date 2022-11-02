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
        # do model one
        pass
    elif model == "long_exp":
        # do model two
        pass
    else:
        # get model data from pandas dataframe
        pass
        # interpolate yr from model data
        # e = np.interp(yr, t_yr, e_GtC_yr)

    return  # e
