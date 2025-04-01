
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import cm
from matplotlib.colors import Normalize



user_specified_directory = 'combined_us_GSOY_data.csv'

data = pd.read_csv(user_specified_directory)

data.dropna(subset=['STATE'], inplace=True)
data.dropna(subset=['TMAX'], inplace=True)

latmin  = 25.0
latmax  = 49.0
longmin = -125.0
longmax = -65.0


# Specify desired grid lengths
nlat         = 10
nlong        = 10

'''
nlat         = 15
nlong        = 15
'''

# Creating latitude and longitude grids
latitude  = np.linspace(latmin,  latmax,  nlat+1)
longitude = np.linspace(longmin, longmax, nlong+1)



# Labelling latitude and longitude coordinates from the data with a grid label.
data['LATITUDE_BIN']  = pd.cut(x=data['LATITUDE'],  bins=latitude)
data['LONGITUDE_BIN'] = pd.cut(x=data['LONGITUDE'], bins=longitude)

locs = data.groupby(['LATITUDE_BIN','LONGITUDE_BIN','DATE'])


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

ax.set_extent([-125.1, -67, 23.3 ,50.5], crs=ccrs.PlateCarree())

ax.stock_img()
ax.coastlines()


data['tmax_avg'] = locs['TMAX'].transform('mean')
data['lat_avg'] = locs['LATITUDE_BIN'].transform(lambda x:  (x.values[0].left + x.values[0].right)/2 )
data['lon_avg'] = locs['LONGITUDE_BIN'].transform(lambda x:  (x.values[0].left + x.values[0].right)/2 )

print("all data")
print(data)


data = data.query('DATE == 2000')

print("data only in 2000")
print(data)


locs2 = data.groupby(['lat_avg','lon_avg'])
print("locs2")
print(locs2.head())



vals = data['tmax_avg']
norm = Normalize(vmin=min(vals), vmax=max(vals)) #Normalize values
cmap = plt.get_cmap('magma') #Choose colormap

ax.scatter(data['lon_avg'], data['lat_avg'], c = cmap(norm(vals)), transform=ccrs.PlateCarree())
cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax, orientation='horizontal')
cbar.set_label('Average Annual Maximum Temperature (Celsius) in the Year 2000')

plt.show()


'''
vals = np.array(locs2['tmax_avg'])
norm = Normalize(vmin=min(vals), vmax=max(vals)) #Normalize values
cmap = plt.get_cmap('magma') #Choose colormap

ax.scatter(locs2['lon_avg'], locs2['lat_avg'], c = cmap(norm(locs2)), transform=ccrs.PlateCarree())
cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax, orientation='horizontal')
cbar.set_label('Average Annual Maximum Temperature (Celsius) in the Year 2000')

plt.show()
'''