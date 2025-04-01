# Load in some modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import plotly.express as px
import plotly.io as pio
from sklearn.linear_model import LinearRegression
sns.set_style("whitegrid")

# Specify directory string in which you have the data file stored
user_specified_directory = '/mnt/d/Erdos_DIR/project_data_TAWNYFILE_DIR/combined_us_GSOY_data.csv'

# Read in data
data = pd.read_csv(user_specified_directory)

######################## INPUTS ###################################
# Specify desired grid lengths
nlat         = 10
nlong        = 10

# lat/long indices for test plot, should be between 1 and nlat or nlong
ilat  = 4
ilong = 7

# Specify minimum number of data points per grid cell
# Ideally choose anything greater than or equal to one
min_data_pts = 10

# Specify years you want to look at
years    = np.array([1960, 1962, 1964, 1966, 1968, 1970, 1972, 1974, 1976, 1978, 1980])

# Specify how many elements at the end of the years array you want to use as your horizon.
# e.g., if years = [1990, 2000, 2010, 2020], then nHorizon=2 picks out 2010 and 2020 as horizon (or test) data.
nHorizon = 5 # This must be <= len(years)-2

# Specify regression model
reg_model = LinearRegression()

# plot parameters
plotzoom  = 3.0
plotlat   = 37
plotlong  = -95
plotrad   = 20.0
soln_type = np.array(['TEMP', 'SNOW', 'PRCP'])

# Option to generate sample plots at final year (for EDA purposes)
plotdata  = False
##################################################################

# min/max latitutude and longitude coordinates (set by hand to exclude Hawaii/Alaska)
latmin  = 25.0
latmax  = 49.0
longmin = -125.0
longmax = -65.0

# Creating latitude and longitude grids
latitude  = np.linspace(latmin,  latmax,  nlat+1)
longitude = np.linspace(longmin, longmax, nlong+1)

# Creating bin labels for latitude and longitude. 
# Note: pd.cut requires these to be 1 less in length than latitude and longitude, hence nlat-1 and nlong-1
latlabel  = [i+1 for i in range(0,len(latitude)-1)]
longlabel = [i+1 for i in range(0,len(longitude)-1)]

# Labelling latitude and longitude coordinates from the data with a grid label.
data['LATITUDE_BIN']  = pd.cut(x=data['LATITUDE'],  bins=latitude,  labels=latlabel)
data['LONGITUDE_BIN'] = pd.cut(x=data['LONGITUDE'], bins=longitude, labels=longlabel)

def data_clean_split(min_data_pts, years, nHorizon, plotzoom, nlat, nlong, soln_type, plotdata):
    # Initializing train-test split
    temp_train = np.zeros(shape=(len(years)-nHorizon, nlat, nlong))
    prcp_train = np.zeros(shape=(len(years)-nHorizon, nlat, nlong))
    snow_train = np.zeros(shape=(len(years)-nHorizon, nlat, nlong))
    lat_train  = np.zeros(shape=(len(years)-nHorizon, nlat))
    long_train = np.zeros(shape=(len(years)-nHorizon, nlong))

    temp_test = np.zeros(shape=(nHorizon, nlat, nlong))
    prcp_test = np.zeros(shape=(nHorizon, nlat, nlong))
    snow_test = np.zeros(shape=(nHorizon, nlat, nlong))
    lat_test  = np.zeros(shape=(nHorizon, nlat))
    long_test = np.zeros(shape=(nHorizon, nlong))

    # Yearly grid count
    grid_count = []

    # Begin cleaning data + creating time series train-test split
    for year in range(len(years)):
        print('\nYear = ', years[year])

        # Arrays in which model features will be stored
        temp_array = []
        prec_array = []
        snow_array = []
        size_array = []

        # Arrays in which mean latitude/longitude coordinates for all data belonging to a grid cell will be stored
        bin_lat_avg    = []
        bin_long_avg   = []

        # Arrays for bin coordinates.
        bin_lat_B  = []
        bin_long_B = []

        # Keeps track of number of grid points per year that meet the min_data_pts threshold
        yearly_grid_number = 0
        for lat in latlabel:
            for long in longlabel:
                # Creating temporary dataframe that includes data only in the grid cell specified by lat and long
                df_filtered = data[(data['LATITUDE_BIN'] == latlabel[lat-1]) & 
                                   (data['LONGITUDE_BIN'] == longlabel[long-1]) & 
                                   (data['DATE'] == years[year])]

                # Removing any rows that have NaNs in any of the model features.
                df_new = df_filtered.dropna(subset=['TMAX', 'PRCP', 'SNOW']) # can improve on this: only look at one variable at a time, not all at once
                num_stations_per_cell = df_new.shape[0]

                # Conditional statement checks to see that there are at least an 'min_data_pts' amount of non-NaN rows in this grid cell
                if(num_stations_per_cell+1>min_data_pts): # pretty sure this can just be greater than or equal to instead of the +1 thingy
                    yearly_grid_number += 1
                    # Compute mean model features within this grid cell
                    temp_array.append(float(np.mean(df_new['TMAX'])))
                    prec_array.append(float(np.mean(df_new['PRCP'])))
                    snow_array.append(float(np.mean(df_new['SNOW'])))
                    size_array.append(float(df_new.shape[0]))

                    # Compute mean lat/long coordinate of all data in the cell to obtain a 'representative' coordinate per grid cell
                    bin_lat_avg.append(float(np.mean(df_new['LATITUDE'])))
                    bin_long_avg.append(float(np.mean(df_new['LONGITUDE'])))

                    # Store actual bin coordinates
                    # NOTE: I THINK THIS IS HOW IT WORKS, BUT IM NOT 100% SURE
                    bin_lat_B.append((latitude[lat] + latitude[lat-1]) / 2.0)
                    bin_long_B.append((longitude[long] + longitude[long-1]) / 2.0)   

                    # form train-test split
                    if(year<len(years)-nHorizon):
                        temp_train[year, lat-1, long-1] = float(np.mean(df_new['TMAX']))
                        snow_train[year, lat-1, long-1] = float(np.mean(df_new['PRCP']))
                        prcp_train[year, lat-1, long-1] = float(np.mean(df_new['SNOW']))
                        lat_train[year, lat-1]          = float(np.mean(df_new['LATITUDE']))
                        long_train[year, long-1]        = float(np.mean(df_new['LATITUDE']))
                    else:
                        temp_test[year-(len(years)-nHorizon), lat-1, long-1] = float(np.mean(df_new['TMAX']))
                        snow_test[year-(len(years)-nHorizon), lat-1, long-1] = float(np.mean(df_new['PRCP']))
                        prcp_test[year-(len(years)-nHorizon), lat-1, long-1] = float(np.mean(df_new['SNOW']))
                        lat_test[year-(len(years)-nHorizon), lat-1]          = float(np.mean(df_new['LATITUDE']))
                        long_test[year-(len(years)-nHorizon), long-1]        = float(np.mean(df_new['LATITUDE']))

        # Keep track of number of grid cells per year
        grid_count.append(int(yearly_grid_number))

        # Compute average number of stations per grid cell
        avg_cell_size = np.mean(size_array)
        print('Average number of stations per grid cell (rounded down) = ', np.floor(avg_cell_size))
        print('Number of grid cells for current year = ', yearly_grid_number)

        # Build new dataframe with grid-cell-averaged data
        coarse_df = pd.DataFrame.from_dict({'TEMP' : temp_array, 'PRCP' : prec_array, 'SNOW' : snow_array, 
                                            'LATITUDE' : bin_lat_avg, 'LONGITUDE' : bin_long_avg, 
                                            'LATITUDE_B' : bin_lat_B, 'LONGITUDE_B' : bin_long_B, 
                                            'SIZE' : size_array})

        # Plot all solution types for the current year in the loop
        if(plotdata and year==len(years)-1):
            for i in range(0,len(soln_type)):
                print('Plotting quantity: ', str(soln_type[i]))
                fig = px.scatter_map(coarse_df, lat='LATITUDE', lon='LONGITUDE', color=str(soln_type[i]), 
                                     zoom=plotzoom, width=1000, height=700, size='SIZE', 
                                     title='Year = ' + str(years[year]) + ', min_data_pts = ' + str(min_data_pts) + ', nlat x nlong = ' + str(nlat) + 'x' + str(nlong))
                # fig.show()
                pio.write_image(fig, str(soln_type[i]) + '_plot_' + str(years[year]) + '_minpts_' + 
                                str(min_data_pts) + '_nlat_x_nlong_' + str(nlat) + 'x' + str(nlong) + '.png')

                # clear plot
                fig.data   = []
                fig.layout = {}

    return temp_train, prcp_train, snow_train, lat_train, long_train, \
           temp_test,  prcp_test,  snow_test,  lat_test,  long_test, \
           grid_count

def forecasting(years, nlat, nlong, nHorizon, traindata, reg_model):
    data_forecast = np.zeros(shape=(nHorizon, nlat, nlong))
    # Loop over grid cells
    for j in range(0,nlat):
        for k in range(0, nlong):
            data_array = []
            year_array = []
            for i in range(0,len(years)-nHorizon):
                # Find non-zero entries for a given grid cell.
                if(traindata[i,j,k]!=0.0):
                    data_array.append(float(traindata[i,j,k]))
                    year_array.append(float(years[i]))
            if(len(data_array)!=0):
                data_forecast[:,j,k] = model_formation(time_train=np.array(year_array).reshape(-1,1),
                                                       data_train=np.array(data_array),
                                                       model=reg_model,
                                                       time_forecast=years[len(years)-nHorizon:len(years)].reshape(-1,1))
    return data_forecast

def model_formation(time_train, data_train, model, time_forecast):
    model.fit(time_train, data_train)
    data_forecast = model.predict(time_forecast)
    return data_forecast

# Clean data + form train/test split
temptrain, prcptrain, snowtrain, lattrain, longtrain, \
temptest, prcptest, snowtest, lattest, longtest, gridcount = \
    data_clean_split(min_data_pts=min_data_pts,
                     years=years, 
                     nHorizon=nHorizon, 
                     plotzoom=plotzoom, 
                     nlat=nlat, 
                     nlong=nlong, 
                     soln_type=soln_type, 
                     plotdata=plotdata)

# Forecast data
temp_forecast = forecasting(years=years, nlat=nlat, nlong=nlong, 
                              nHorizon=nHorizon, traindata=temptrain, 
                              reg_model=reg_model)

fs_label  = 15
fs_tick   = 14
fs_legend = 13

plt.figure(0)
plt.plot(years[0:len(years)-nHorizon], temptrain[:,ilat,ilong] * 9./5. + 32., linewidth=3.0, color='black', label='train data')
plt.plot(years[len(years)-nHorizon:len(years)], temp_forecast[:,ilat,ilong] * 9./5. + 32., linewidth=1.5, color='red', linestyle='dashed', label=r'forecast')
plt.plot(years[len(years)-nHorizon:len(years)], temptest[:,ilat,ilong] * 9./5. + 32, linewidth=3.0, color='blue', label='test data')
plt.title('lat x long = ' + '{:.3f}'.format(lattrain[0,ilat]) + ' x ' + '{:.3f}'.format(longtrain[0,ilong]))
plt.tick_params(axis='x', labelsize=fs_tick)
plt.xlabel(r'Year', fontsize = fs_label)
plt.ylabel(r'$T$ [degrees Fahrenheit]', fontsize = fs_label)
plt.legend(fontsize = fs_legend)
plt.savefig('temp_vs_year_nHorizon_'+str(nHorizon)+'_mindatapts_'+str(min_data_pts)+'_minYear_'+str(years[0])+'_maxYear_'+str(years[len(years)-1])+'.png')