# Load in some modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import plotly.express as px
import plotly.io as pio
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import csv
import sys
sns.set_style("whitegrid")

# Specify directory string in which you have the data file stored
user_specified_directory = '/mnt/d/Erdos_DIR/project_data_TAWNYFILE_DIR/combined_us_GSOY_data.csv'

# Read in data
data = pd.read_csv(user_specified_directory)

######################## INPUTS ###################################
# Specify desired grid lengths
nlat         = 10
nlong        = 10

# Specify desired feature: TMAX, SNOW, or PRCP
user_feature = 'TMAX'

# lat/long indices for test plot, should be between 1 and nlat or nlong
ilat  = 5
ilong = 8

# Specify minimum number of data points per grid cell
# Ideally choose anything greater than or equal to one
min_data_pts = 10

# Specify years you want to look at
years    = np.array([2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024])

ltrain   = 2  # length of train set
lhorizon = 2  # length of validation set
iterate  = 1  # step size of validation set

# Specify regression model
reg_model = LinearRegression()

# Specify options for (1) creating RMS error csv files for each validation set and (2) creating plots at each validation set
save_errors          = False
plot_each_validation = False
##################################################################
           
def data_clean_split_single_feature(min_data_pts, years, nTrain, nHorizon, nlat, nlong, feature):
    # Initializing train-test split
    x_train    = np.zeros(shape=(nTrain, nlat, nlong))
    lat_train  = np.zeros(shape=(nTrain, nlat))
    long_train = np.zeros(shape=(nTrain, nlong))

    x_test    = np.zeros(shape=(nHorizon, nlat, nlong))
    lat_test  = np.zeros(shape=(nHorizon, nlat))
    long_test = np.zeros(shape=(nHorizon, nlong))

    lat_grid  = np.zeros(shape=(nlat, nlong))
    long_grid = np.zeros(shape=(nlat, nlong))

    # Yearly grid count
    grid_count = []

    # Begin cleaning data + creating time series train-test split
    for year in range(len(years)):
        if(year<nTrain):
            print('\nYear (train) = ', years[year])
        else:
            print('\nYear (test)  = ', years[year])

        # Arrays in which model features will be stored
        x_array    = []
        size_array = []

        # Arrays in which mean latitude/longitude coordinates for all data belonging to a grid cell will be stored
        bin_lat_avg   = []
        bin_long_avg  = []

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
                df_new = df_filtered.dropna(subset=[feature]) # can improve on this: only look at one variable at a time, not all at once
                num_stations_per_cell = df_new.shape[0]

                # Conditional statement checks to see that there are at least an 'min_data_pts' amount of non-NaN rows in this grid cell
                if(num_stations_per_cell+1>min_data_pts): # pretty sure this can just be greater than or equal to instead of the +1 thingy
                    yearly_grid_number += 1
                    # Compute mean model features within this grid cell
                    x_array.append(float(np.mean(df_new[feature])))
                    size_array.append(float(df_new.shape[0]))

                    # Compute mean lat/long coordinate of all data in the cell to obtain a 'representative' coordinate per grid cell
                    bin_lat_avg.append(float(np.mean(df_new['LATITUDE'])))
                    bin_long_avg.append(float(np.mean(df_new['LONGITUDE'])))

                    # Store actual bin coordinates
                    # NOTE: I THINK THIS IS HOW IT WORKS, BUT IM NOT 100% SURE
                    bin_lat_B.append((latitude[lat] + latitude[lat-1]) / 2.0)
                    bin_long_B.append((longitude[long] + longitude[long-1]) / 2.0)

                    lat_grid[lat-1, long-1]  = float(np.mean(df_new['LATITUDE']))
                    long_grid[lat-1, long-1] = float(np.mean(df_new['LONGITUDE']))

                    # form train-test split
                    if(year<nTrain):
                        x_train[year, lat-1, long-1] = float(np.mean(df_new[feature]))
                        lat_train[year, lat-1]       = float(np.mean(df_new['LATITUDE']))
                        long_train[year, long-1]     = float(np.mean(df_new['LATITUDE']))
                    else:
                        x_test[year-nTrain, lat-1, long-1] = float(np.mean(df_new[feature]))
                        lat_test[year-nTrain, lat-1]       = float(np.mean(df_new['LATITUDE']))
                        long_test[year-nTrain, long-1]     = float(np.mean(df_new['LATITUDE']))

        # Keep track of number of grid cells per year
        grid_count.append(int(yearly_grid_number))

        # Compute average number of stations per grid cell
        avg_cell_size = np.mean(size_array)
        print('Average number of stations per grid cell (rounded down) = ', np.floor(avg_cell_size))
        print('Number of grid cells for current year = ', yearly_grid_number)

    return x_train, x_test, lat_grid, long_grid, grid_count

def forecasting(years, nlat, nlong, nTrain, nHorizon, traindata, reg_model):
    data_forecast  = np.zeros(shape=(nHorizon, nlat, nlong))
    data_coef      = np.zeros(shape=(nlat, nlong))
    data_intercept = np.zeros(shape=(nlat, nlong))
    # Loop over grid cells
    for j in range(0,nlat):
        for k in range(0, nlong):
            data_array = []
            year_array = []
            for i in range(0,nTrain):
                # Find non-zero entries for a given grid cell.
                if(traindata[i,j,k]!=0.0):
                    data_array.append(float(traindata[i,j,k]))
                    year_array.append(float(years[i]))
            if(len(data_array)!=0):
                data_forecast[:,j,k], data_coef[j,k], data_intercept[j,k] = \
                                        model_formation(time_train=np.array(year_array).reshape(-1,1),
                                                       data_train=np.array(data_array),
                                                       model=reg_model,
                                                       time_forecast=years[nTrain:len(years)].reshape(-1,1))
    return data_forecast, data_coef, data_intercept

def model_formation(time_train, data_train, model, time_forecast):
    model.fit(time_train, data_train)
    data_forecast = model.predict(time_forecast)
    return data_forecast, model.coef_[0], model.intercept_

def RMSe(test, forecast, nH, nLat, nLong):
    errors = np.zeros(shape=(nLat, nLong))
    for i in range(0,nLat):
        for j in range(0,nLong):
            errors[i,j] = root_mean_squared_error(test[0:nH,i,j], forecast[0:nH,i,j])
    return errors 

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

nCross   = int(np.floor(len(years) - (ltrain + lhorizon)) / iterate) # number of cross validation sets
if(nCross < 1.):
    print('Number of cross validation sets is invalid')
    print('nYears   = ', len(years))
    print('ltrain   = ', ltrain)
    print('lhorizon = ', lhorizon)
    print('iterate  = ', iterate)
    print('nCross = floor[ (nYears - (ltrain + lhorizon) + 1) / iterate ] = ', nCross)
    print('exiting...')
    sys.exit()
print('Number of validation sets: floor[ (nYears - (ltrain + lhorizon) + 1) / iterate ] = ', nCross + 1)

# Initial errors data structure
errors   = np.zeros(shape=(nCross+1, nlat, nlong))

# Initialize cross-validation indices
iB_train = 0
iB_test  = ltrain
iE_train = ltrain
iE_test  = ltrain+lhorizon

# plotting parameters
fs_label  = 15
fs_tick   = 14
fs_legend = 13

# Perform cross-validation test
i_cross = 0
while iE_test <= len(years):
    years_new = years[iB_train:iE_test]

    # Form train/test splits
    xtrain, xtest, lat_grid, long_grid, gridcount = \
        data_clean_split_single_feature(min_data_pts=min_data_pts,
                                        years=years_new,
                                        nTrain=ltrain,
                                        nHorizon=lhorizon,
                                        nlat=nlat,
                                        nlong=nlong,
                                        feature=user_feature)
    
    # Fit model and forecast
    xforecast, xcoef, xintercept  = forecasting(years=years_new, nlat=nlat, nlong=nlong,
                                                nTrain=ltrain, nHorizon=lhorizon, traindata=xtrain,
                                                reg_model=reg_model)
    
    # Compute errors and save to file (if selected)
    errors[i_cross, :, :] = RMSe(test=xtest, forecast=xforecast, nH=lhorizon, nLat=nlat, nLong=nlong)
    if(save_errors):
        error_file = 'error_ncross_'+str(i_cross+1)+'_minYear_'+str(years[0])+'_maxYear_'+str(years[len(years)-1])+'_var_'+user_feature
        columns = {'longitude':lat_grid.reshape(-1,1)[:,0], 
                   'latitude':long_grid.reshape(-1,1)[:,0], 
                   'intercept':xintercept.reshape(-1,1)[:,0], 
                   'slope':xcoef.reshape(-1,1)[:,0], 
                   'RMS errors':errors[i_cross, :, :].reshape(-1,1)[:,0]}
        df_errors = pd.DataFrame(columns)
        df_filter = df_errors[(df_errors != 0).any(axis=1)] # Remove zero'd out rows
        df_filter.to_csv(error_file+'.csv', index=False)
    else:
        print('Not saving errors to file.')

    # Plot results (if selected)
    if(plot_each_validation):
        plt.figure(i_cross)
        plt.plot(years[0:iE_train], xtrain[0:iE_train, ilat, ilong] * 9./5. + 32., linewidth=3.0, color='black', label='train data')
        plt.plot(years[iB_test:iE_test], xforecast[:, ilat, ilong] * 9./5. + 32., linewidth=1.5, color='red', linestyle='dashed', label=r'forecast')
        plt.plot(years[iB_test:iE_test], xtest[:, ilat, ilong]     * 9./5. + 32,  linewidth=3.0, color='blue', label='test data')
        plt.title('lat x long = ' + '{:.3f}'.format(lat_grid[ilat,ilong]) + ' x ' + '{:.3f}'.format(long_grid[ilat,ilong]) + ', RMS error: ' + '{:.3e}'.format(errors[i_cross, ilat, ilong]) + ', ncross = ' + str(i_cross+1))
        plt.tick_params(axis='x', labelsize=fs_tick)
        plt.xticks(years[0:iE_test], [int(label) for label in years[0:iE_test]])
        plt.xlabel(r'Year', fontsize = fs_label)
        plt.ylabel(r'$T$ [degrees Fahrenheit]', fontsize = fs_label)
        plt.legend(fontsize = fs_legend)

    # Iteration cross validation indices
    ltrain  += iterate
    iB_test  = ltrain
    iE_train = ltrain
    iE_test  = ltrain+lhorizon
    i_cross += 1

# Plot RMS error vs nCross for user specified coordinate point
crossValidation_ARR = [i for i in range(0,nCross+1)]
plt.figure(0)
plt.clf()
plt.plot(crossValidation_ARR, errors[:, ilat, ilong], linewidth=3.0, color='black')
plt.title('lat x long = ' + '{:.3f}'.format(lat_grid[ilat,ilong]) + ' x ' + '{:.3f}'.format(long_grid[ilat,ilong]))
plt.tick_params(axis='x', labelsize=fs_tick)
plt.ylabel(r'RMS error', fontsize = fs_label)
plt.xlabel(r'$n_{\mathrm{cross}}$', fontsize = fs_label)
plt.savefig('RMSe_vs_nCross_lat_'+
            '{:.3f}'.format(lat_grid[ilat,ilong])+
            '_long_'+
            '{:.3f}'.format(long_grid[ilat,ilong])+
            '_minYear_'+str(years[0])+
            '_maxYear_'+str(years[len(years)-1])+
            '_val_'+str(user_feature)+'.png')
