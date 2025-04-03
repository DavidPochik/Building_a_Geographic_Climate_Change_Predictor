
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import cm
from matplotlib.colors import Normalize
import sklearn as sk


user_specified_directory = 'combined_us_GSOY_data.csv'

data = pd.read_csv(user_specified_directory)

data.dropna(subset=['STATE'], inplace=True)
data.dropna(subset=['TMAX'], inplace=True)

data2 = data.query('DATE >= 1950')
data2 = data2.query('DATE < 2010')


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
data2['LATITUDE_BIN']  = pd.cut(x=data['LATITUDE'],  bins=latitude)
data2['LONGITUDE_BIN'] = pd.cut(x=data2['LONGITUDE'], bins=longitude)

#locs = data.groupby(['LATITUDE_BIN','LONGITUDE_BIN','DATE'])
locs = data2.groupby(['LATITUDE_BIN','LONGITUDE_BIN','DATE'])


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

ax.set_extent([-125.1, -67, 23.3 ,50.5], crs=ccrs.PlateCarree())

ax.stock_img()
ax.coastlines()


data2['tmax_avg'] = locs['TMAX'].transform('mean')
data2['lat_avg'] = locs['LATITUDE_BIN'].transform(lambda x:  (x.values[0].left + x.values[0].right)/2 )
data2['lon_avg'] = locs['LONGITUDE_BIN'].transform(lambda x:  (x.values[0].left + x.values[0].right)/2 )

#print("all data")
#print(data2)


#data = data.query('DATE == 2000')

#print("data only in 2000")
#print(data)




locs2 = data2.groupby(['lat_avg','lon_avg'])
#print("locs2")
#print(locs2.head())


'''
vals = data['tmax_avg']
norm = Normalize(vmin=min(vals), vmax=max(vals)) #Normalize values
cmap = plt.get_cmap('magma') #Choose colormap

ax.scatter(data['lon_avg'], data['lat_avg'], c = cmap(norm(vals)), transform=ccrs.PlateCarree())
cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax, orientation='horizontal')
cbar.set_label('Average Annual Maximum Temperature (Celsius) in the Year 2000')

plt.show()
'''



i = 0 

m_fit = []
b_fit = []
lats = []
lons = []
errs = []

for key, item in locs2:

	if len(item['DATE'].unique())>1:


		#print(locs2.get_group(key), "\n\n")
		#print("LATITUDE_BIN")
		#latbin = item['LATITUDE_BIN'].values[0]
		#lonbin = item['LONGITUDE_BIN'].values[0]
		stuff  = [item['DATE'].values,item['tmax_avg'].values]

		z = np.polyfit(stuff[0], stuff[1], 1)

		#print("z[0]")
		#print(z[0])
		#print("z[1]")
		#print(z[1])
		p = np.poly1d(z)

		#print("Date and tmax_avg")
		#print(stuff)
		#print("print(len(item['DATE'].unique()))")
		#print(len(item['DATE'].unique()))


		'''
		plt.plot(stuff[0],stuff[1],'o')
		plt.plot(stuff[0],p(stuff[0]),'-',color='red')
		plt.title(f" Lon: {item['lon_avg'].values[0]} and Lat: {item['lat_avg'].values[0]}")
		plt.xlabel("year")
		plt.ylabel("tmax_avg")
		plt.show()
		'''


		err = sk.metrics.root_mean_squared_error(stuff[1],p(stuff[0]))

		b_fit.append(z[1])
		m_fit.append(z[0])
		lats.append(item['lat_avg'].values[0])
		lons.append(item['lon_avg'].values[0])
		errs.append(err)

		#i = i +1

		#if i ==3:
		#	break


	#print(latbin)
	#print(latbin.left)
	#print(latbin.right)
	#print(item['TMAX'])
	#print("item['TMAX'].mean()")
	#print(item['TMAX'].mean())




inside = {'lons':lons, 'lats':lats, 'b term':b_fit, "m term": m_fit, "train rms error": errs}
df = pd.DataFrame(inside)

print(df)





###########################
#Testing!
###########################

data3 = data.query('DATE >= 2010')


# Labelling latitude and longitude coordinates from the data with a grid label.
data3['LATITUDE_BIN']  = pd.cut(x=data3['LATITUDE'],  bins=latitude)
data3['LONGITUDE_BIN'] = pd.cut(x=data3['LONGITUDE'], bins=longitude)


locs = data3.groupby(['LATITUDE_BIN','LONGITUDE_BIN','DATE'])


data3['tmax_avg'] = locs['TMAX'].transform('mean')
data3['lat_avg'] = locs['LATITUDE_BIN'].transform(lambda x:  (x.values[0].left + x.values[0].right)/2 )
data3['lon_avg'] = locs['LONGITUDE_BIN'].transform(lambda x:  (x.values[0].left + x.values[0].right)/2 )



locs3 = data3.groupby(['lat_avg','lon_avg'])

#print(locs3.head())


train_err = []

for key, item in locs3:

	if len(item['DATE'].unique())>1:

		model = df[   (df['lons'] == item['lon_avg'].values[0] )  &  (df['lats'] == item['lat_avg'].values[0] ) ] 
		
		stuff  = [item['DATE'].values,item['tmax_avg'].values]

		z = [model['m term'].values[0], model['b term'].values[0]]

		p = np.poly1d(z)

		err = sk.metrics.root_mean_squared_error(stuff[1],p(stuff[0]))

		train_err.append(err)




df.insert(len(df.columns),'train error', train_err)

print(df)


df.to_csv("alison_model_test.csv")

