import numpy as np
import pandas as pd


# standalone
sigma = np.multiply(5.6696, 10**(-8))  # boltzmann constant
S_0 = 1368  # solar constant
R = np.multiply(6371, np.power(10,3))  # radius earth
A = np.multiply(np.pi, np.power(R,2))  # total SA earth
epsilon = 1  # emissivity
tau = 0.63  # transmissivity
alpha = np.array([0.4, 0.1, 0.6])  # albedos
rho = np.array([2500, 1028, 900])  # densitites
Z = np.array([1, 70, 1])  # thermal scale depth
c = np.array([790, 4187, 2060])  # specific heat capacity


#calculate zonal averages
zone_params = pd.read_csv('equal_zones.csv')

gamma = np.array(zone_params[["gamma"]])  # spatial radiation factors
area = np.array(zone_params[["frac area"]]*A)
land_types = np.array(zone_params[["land","ocean","ice"]])
albedo_surface = np.sum((land_types * alpha), axis=1)  # surface albedos
albedo_sky = np.array([0.2,0.2,0.2,0.2,0.2,0.2])
densities = np.sum((land_types * rho), axis=1)  # densities
specific_heats = np.sum((land_types * c), axis=1)  # specific heat capacities (c_k)
thermals = np.sum((land_types * Z), axis=1)  # thermal scale depth

#boundary parameters
boundary_params = pd.read_csv('boundary_equal_zones.csv')
L = np.array(boundary_params[["L (m)"]]) 
k = np.array(boundary_params[["k (W m-1 K-1)"]]) 
