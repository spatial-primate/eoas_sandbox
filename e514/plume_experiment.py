import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from particle_plumes import ParticlePlume as Plume

sns.set_theme(style="whitegrid")

# todo: tabulate plume experiments into a csv
experiments = pd.read_csv('plume_experiments_e514.csv')

plumes = []
for index, experiment in experiments.iterrows():
    plumes.append(
        Plume(trial=experiment['trial'],
              multiphase=experiment['multiphase [-]'],
              diameter=experiment['particle_diameter (m)'],
              mixture_density=experiment['mixture_density (kg/m3)']
              )
    )

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(plumes[0].plume_velocity, plumes[0].initial_buoyancy_flux, label='Plume')
