
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import cm
from matplotlib.colors import Normalize


allstations = pd.read_csv('combined_us_GSOY_data.csv',delimiter=',',on_bad_lines='skip')




#station['state'].replace('', np.nan, inplace=True)

#station.dropna(subset=['state'], inplace=True)
#station = station[station['code'].str.contains("US1")]
#station = allstation[allstation['code'].str.contains("US1MD")]


#station = allstations[allstations['STATE'].str.contains("CA")]
station = allstations

#print(station)



locs = station.groupby('STATION')
print("part2")
'''
for name, group in locs:


	#print(name)
	plt.plot(group['DATE'],group["TMAX"])

plt.xlabel("Year")
plt.ylabel("TMAX")
plt.title("Stations in Maryland (Tawny File)")
plt.show()
'''




'''
print("part3")

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
#maryland
#ax.set_extent([-79.6, -74.9, 37 ,39.85], crs=ccrs.PlateCarree())


ax.stock_img()
ax.coastlines()



locs = station.groupby('STATION')





group_means = station.groupby('STATION')['TMAX'].transform('mean')
filtered_station = station[group_means.notna()]
filtered_locs = filtered_station.groupby('STATION')



vals = filtered_locs['TMAX'].mean()
norm = Normalize(vmin=min(vals), vmax=max(vals)) #Normalize values
cmap = plt.get_cmap('magma') #Choose colormap

ax.scatter(filtered_locs['LONGITUDE'].mean(), filtered_locs['LATITUDE'].mean(), c = cmap(norm(vals)), transform=ccrs.PlateCarree())
#cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax, orientation='horizontal')
cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax, orientation='horizontal')
cbar.set_label('Average Annual Maximum Temperature (Celsius)')

plt.show()

'''


print("part3")

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
#maryland
#ax.set_extent([-79.6, -74.9, 37 ,39.85], crs=ccrs.PlateCarree())
ax.set_extent([-160, -80, 17 ,73], crs=ccrs.PlateCarree())#Whole Country

ax.stock_img()
ax.coastlines()



locs = station.groupby('STATION')





group_means = station.groupby('STATION')['T_DIFF'].transform('mean')
filtered_station = station[group_means.notna()]
filtered_locs = filtered_station.groupby('STATION')



vals = filtered_locs['T_DIFF'].mean()
norm = Normalize(vmin=min(vals), vmax=max(vals)) #Normalize values
cmap = plt.get_cmap('magma') #Choose colormap

ax.scatter(filtered_locs['LONGITUDE'].mean(), filtered_locs['LATITUDE'].mean(), c = cmap(norm(vals)), transform=ccrs.PlateCarree())
#cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax, orientation='horizontal')
cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax, orientation='horizontal')
cbar.set_label('Average Annual Maximum and Annual Minimum Difference (Celsius)')

plt.show()