import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

# lake_depth = np.loadtxt("lake_depth.txt")
kinematic_viscosity = np.logspace(2, 11)
# timescales = lake_depth**2 / kinematic_viscosity

times = np.logspace(-2, 2, 10)
depths = np.linspace(0, 1)
N = 100000

fourier_sum = np.zeros_like(depths)

for time in times:
    for ii in range(N):
        fourier_sum += np.exp(-(ii*np.pi)**2*time)*np.sin(ii*np.pi*depths)
    shudder_transmission = (1 - depths) - 2/np.pi * fourier_sum
    plt.plot(shudder_transmission, -depths)

plt.show()
