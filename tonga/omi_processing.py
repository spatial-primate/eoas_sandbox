import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import h5py

root_directory = r"/Users/lukebrown/Library/CloudStorage/Dropbox/S_scavenging_2023/papers/observations/lidar/datafiles"
file = "OMI-Aura_L2G-OMTO3G_2022m0115_v003-2022m0116t063032.he5"

filename = os.path.join(root_directory, file)


def array_with_nans(h5var):
    """ Extracts the array and replaces fillvalues and missing values with Nans
    """
    array = h5var[:]  # not very efficient

    # _FillValue and MissingValue attributes are lists
    for value in h5var.attrs['MissingValue']:
        array[array == value] = np.nan

    for value in h5var.attrs['_FillValue']:
        array[array == value] = np.nan

    return array


with h5py.File(filename, "r") as f:
    # Print all root level object names (aka keys)
    # these can be group or dataset names
    print("Keys: %s" % f.keys())
    # get first object name/key; may or may NOT be a group
    a_group_key = list(f.keys())[0]

    # get the object type for a_group_key: usually group or dataset
    print(type(f[a_group_key]))
    group_key = 'GRIDS'
    omi_product = 'OMI Column Amount O3'
    data_fields = f[a_group_key][group_key][omi_product]['Data Fields']
    # geolocation_fields = f['HDFEOS']['GRIDS']['OMI Column Amount O3']['Geolocation Fields']
    data = data_fields['ColumnAmountO3']
    # data = dataset[:]

    offset = data.attrs['Offset'][0]
    print(f"offset: {offset}")
    scale = data.attrs['ScaleFactor'][0]
    print(f"scale: {scale}")

    candidate = 0

    dataArray = array_with_nans(data)[candidate]

    data_mask = np.ma.masked_array(dataArray, np.isnan(dataArray))

    map_label = data.attrs['Units'].decode()

    # Define the range of latitudes and longitudes
    min_latitude = -90.0
    max_latitude = 90.0
    min_longitude = -180.0
    max_longitude = 180.0

    # Create arrays of latitudes and longitudes
    latitudes = np.linspace(min_latitude, max_latitude, num=720)
    longitudes = np.linspace(min_longitude, max_longitude, num=1440)

    # Create a meshgrid of latitudes and longitudes
    grid_long, grid_lat = np.meshgrid(longitudes, latitudes)

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.contourf(grid_long, grid_lat, data[candidate, :, :], cmap='viridis')  # Adjust cmap as needed
    plt.colorbar(label='Data')  # Add colorbar
    plt.title('Plot of Data over Grid of Latitudes and Longitudes')  # Add title
    plt.xlabel('Longitude')  # Add x-axis label
    plt.ylabel('Latitude')  # Add y-axis label
    plt.grid(True)  # Add gridlines
    plt.show()

    offset = data.attrs['Offset'][0]
    print(f"offset: {offset}")
    scale = data.attrs['ScaleFactor'][0]
    print(f"scale: {scale}")
    #
    # candidate = 0
    #
    # dataArray = array_with_nans(data)[candidate]
    # dataArray = scale * (dataArray - offset)
    #
    # lat = array_with_nans(data_fields['Latitude'])[candidate]
    # lon = array_with_nans(data_fields['Longitude'])[candidate]
    #
    # data_mask = np.ma.masked_array(dataArray, np.isnan(dataArray))
    #
    # fig = plt.figure()
    #
    # m = Basemap(projection='cyl', resolution='l',
    #             llcrnrlat=-90, urcrnrlat=90,
    #             llcrnrlon=-180, urcrnrlon=180)
    #
    # m.drawcoastlines(linewidth=0.5)
    # m.drawparallels(np.arange(-90., 60., 30.), labels=[1, 0, 0, 0])
    # m.drawmeridians(np.arange(-180, 180., 60.), labels=[0, 0, 0, 1])
    # my_cmap = mpl.colormaps.get_cmap('gist_stern_r')
    # my_cmap.set_under('w')
    # m.pcolormesh(lon, lat, data_mask, latlon=True, cmap=my_cmap, shading='gouraud')
    # cb = m.colorbar()
    # cb.set_label(map_label)
    # plt.autoscale()
    # plt.show()
