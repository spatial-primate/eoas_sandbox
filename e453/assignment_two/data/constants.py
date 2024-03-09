import numpy as np
import pandas as pd

# standalone
sigma = np.multiply(5.6696, 10 ** (-8))  # boltzmann constant
S_0 = 1368  # solar constant
R = np.multiply(6371, np.power(10, 3))  # radius earth
A = np.multiply((np.pi * 4), np.power(R, 2))  # total SA earth
epsilon = 1  # emissivity
tau = 0.63  # transmissivity
b = sigma * tau * epsilon
alpha = np.array([0.4, 0.1, 0.6])  # albedos
rho = np.array([2500, 1028, 900])  # densities
Z = np.array([1, 70, 1])  # thermal scale depth
c = np.array([790, 4187, 2060])  # specific heat capacity

# unequal zones parameters

# calculate zonal averages
zone_params = pd.read_csv('./data/equal_zones.csv')
GAMMA = np.array(zone_params[["gamma"]])  # spatial radiation factors
area = np.array(zone_params[["frac area"]] * A)
land_types = np.array(zone_params[["land", "ocean", "ice"]])
ALBEDO_SURFACE = np.sum((land_types * alpha), axis=1)  # surface albedos
albedo_snowball = np.ones(6)*alpha[2]  # all zones ice
albedo_not_snowball = np.ones(6)*((alpha[0]*0.3) + (alpha[1]*0.7))  # 70perc water 30perc land
ALBEDO_SKY = np.ones(6)*0.2
densities = np.sum((land_types * rho), axis=1)  # densities
specific_heats = np.sum((land_types * c), axis=1)  # specific heat capacities (c_k)
thermals = np.sum((land_types * Z), axis=1)  # thermal scale depth
noncoupling_prefactors = np.divide(1,
                                   densities * thermals * specific_heats
                                   )

# boundary parameters
df = pd.read_csv('./data/boundary_equal_zones.csv')
k = df['k (W m-1 K-1)'].values
L = df['L (m)'].values
boundary_params = pd.read_csv('data/kl_heat_transfer_equal_zones.csv', header=None)
k_matrix = np.array(boundary_params)
