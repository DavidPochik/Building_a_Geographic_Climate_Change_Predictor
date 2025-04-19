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
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
sns.set_style("whitegrid")

# Specify directory string in which you have the data file stored
user_specified_directory = '../data/combined_us_GSOY_data.csv'

# Read in data
data = pd.read_csv(user_specified_directory)

# Convert temperature to K and compute TAVG
data['TMIN'] = data['TMIN'] + 273.15
data['TMAX'] = data['TMAX'] + 273.15
data['TAVG'] = (data['TMIN'] + data['TMAX']) / 2.0

######################## INPUTS SECTION #######################################
# Specify desired grid lengths
nlat         = 10
nlong        = 10

# Specify desired feature: TMIN, TMAX, TAVG, SNOW, PRCP, HTDD
user_feature = 'TAVG'

# lat/long indices for test plot, should be between 1 and nlat or nlong
ilat  = 5
ilong = 8

# Specify minimum number of data points per grid cell
# Ideally choose anything greater than or equal to one
min_data_pts = 10

# Specify timeline parameters
year_start = 2000                                                    # initial year
year_end   = 2024                                                     # final year
delta_year = 2                                                        # spacing between years
test_size  = 0.06 #0.2                                                # percentage size of test set

# Specify cross-validation parameters
ltrain   = 2  # initial length of train set (in element indices, not necessarily years)
lhorizon = 2  # length of validation set    (in element indices, not necessarily years)
iterate  = 1  # step size of validation set (in element indices, not necessarily years)

# Specify regression model
reg_model = Pipeline([('fit',LinearRegression())])

# Specify final year for projected deltas, and number of years in projection
year_end_forecast = 2030
nYfc = 30

# Specify options for (1) creating RMS error csv files for each validation set, (2) creating plots at each validation step, 
# (3) plotting the last validation step, (4) saving the test data, and (5) plotting the geographical feature data
save_errors          = False
plot_each_validation = False
plot_last_validation = False
save_test_data       = False
plotdata             = False

######################## END INPUTS SECTION ###################################

####################### FUNCTIONS ##############################################
def data_clean_split_single_feature(data, min_data_pts, years, nTrain, nHorizon, nlat, nlong, feature, plotdata):
    # Initializing train-test split
    x_train = np.zeros(shape=(nTrain, nlat, nlong))
    x_test  = np.zeros(shape=(nHorizon, nlat, nlong))

    # mean (or 'representative') coordinates
    lat_grid  = np.zeros(shape=(nlat, nlong))
    long_grid = np.zeros(shape=(nlat, nlong))

    # cell-centered coordinates
    lat_bin_center  = np.zeros(shape=(nlat, nlong))
    long_bin_center = np.zeros(shape=(nlat, nlong))

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
        lat_array  = []
        long_array = []

        # Keeps track of number of grid points per year that meet the min_data_pts threshold
        yearly_grid_number = 0
        for lat in latlabel:
            for long in longlabel:
                # Creating temporary dataframe that includes data only in the grid cell specified by lat and long
                df_filtered = data[(data['LATITUDE_BIN'] == latlabel[lat-1]) & 
                                   (data['LONGITUDE_BIN'] == longlabel[long-1]) & 
                                   (data['DATE'] == years[year])]

                # Removing any rows that have NaNs in any of the model features. Do this for filtered datasets to reduce computational overhead
                df_new = df_filtered.dropna(subset=[feature])
                num_stations_per_cell = df_new.shape[0]

                # Conditional statement checks to see that there are at least an 'min_data_pts' amount of non-NaN rows in this grid cell
                if(num_stations_per_cell+1>min_data_pts):
                    yearly_grid_number += 1

                    # Compute mean model features within this grid cell
                    x_array.append(float(np.mean(df_new[feature])))
                    size_array.append(float(df_new.shape[0]))
                    lat_array.append(float(np.mean(df_new['LATITUDE']))) #append(float((latitude[lat] + latitude[lat-1]) / 2.0))
                    long_array.append(float(np.mean(df_new['LONGITUDE']))) #.append(float((longitude[long] + longitude[long-1]) / 2.0))

                    # Compute average lat/long coordinates of weather stations in a grid cell
                    lat_grid[lat-1, long-1]  = float(np.mean(df_new['LATITUDE']))
                    long_grid[lat-1, long-1] = float(np.mean(df_new['LONGITUDE']))

                    # Compute cell centers
                    lat_bin_center[lat-1, long-1]  = (latitude[lat] + latitude[lat-1]) / 2.0
                    long_bin_center[lat-1, long-1] = (longitude[long] + longitude[long-1]) / 2.0

                    # form train-test split
                    if(year<nTrain):
                        x_train[year, lat-1, long-1] = float(np.mean(df_new[feature]))
                    else:
                        x_test[year-nTrain, lat-1, long-1] = float(np.mean(df_new[feature]))

        # Create geographical plot of feature data at final year in the current cross-validation step
        if(plotdata and year==len(years)-1):    
            coarse_df = pd.DataFrame.from_dict({'FEATURE' : x_array, 
                                            'LATITUDE' : lat_array, 
                                            'LONGITUDE' : long_array,
                                            'SIZE' : size_array})
            soln_type='FEATURE'
            plotzoom  = 3.0
            fig = px.scatter_map(coarse_df, lat='LATITUDE', lon='LONGITUDE', color=str(soln_type), 
                                 zoom=plotzoom, width=1000, height=700, size='SIZE', 
                                 title='Year = ' + str(years[year]) + ', min_data_pts = ' + str(min_data_pts) + ', nlat x nlong = ' + str(nlat) + 'x' + str(nlong))
            # fig.show()
            pio.write_image(fig, 'TAVG' + '_plot_' + str(years[year]) + '_minpts_' + 
                            str(min_data_pts) + '_nlat_x_nlong_' + str(nlat) + 'x' + str(nlong) + '.png')

            # clear plot
            fig.data   = []
            fig.layout = {}

        # Keep track of number of grid cells per year
        grid_count.append(int(yearly_grid_number))

        # Compute average number of stations per grid cell
        avg_cell_size = np.mean(size_array)
        print('Average number of stations per grid cell (rounded down) = ', np.floor(avg_cell_size))
        print('Number of grid cells for current year = ', yearly_grid_number)

    return x_train, x_test, lat_grid, long_grid, lat_bin_center, long_bin_center, grid_count

def data_test_set(data, min_data_pts, years, nlat, nlong, feature):
    # Initialize test set
    x_test        = np.zeros(shape=(len(years), nlat, nlong))

    # mean (or 'representative') coordinates
    lat_grid      = np.zeros(shape=(nlat, nlong))
    long_grid     = np.zeros(shape=(nlat, nlong))

    # cell-centered coordinates
    lat_bin_center  = np.zeros(shape=(nlat, nlong))
    long_bin_center = np.zeros(shape=(nlat, nlong))

    # Yearly grid count
    grid_count = []

    # Begin cleaning data + creating time series train-test split
    for year in range(len(years)):

        # Arrays in which model features will be stored
        x_array    = []
        size_array = []

        # Keeps track of number of grid points per year that meet the min_data_pts threshold
        yearly_grid_number = 0
        for lat in latlabel:
            for long in longlabel:
                # Creating temporary dataframe that includes data only in the grid cell specified by lat and long
                df_filtered = data[(data['LATITUDE_BIN'] == latlabel[lat-1]) & 
                                   (data['LONGITUDE_BIN'] == longlabel[long-1]) & 
                                   (data['DATE'] == years[year])]

                # Removing any rows that have NaNs in any of the model features. Do this for filtered datasets to reduce computational overhead
                df_new = df_filtered.dropna(subset=[feature])
                num_stations_per_cell = df_new.shape[0]

                # Conditional statement checks to see that there are at least an 'min_data_pts' amount of non-NaN rows in this grid cell
                if(num_stations_per_cell+1>min_data_pts): # pretty sure this can just be greater than or equal to instead of the +1 thingy
                    yearly_grid_number += 1

                    # Compute mean model features within this grid cell
                    x_array.append(float(np.mean(df_new[feature])))
                    size_array.append(float(df_new.shape[0]))

                    # mean or representative coordinates
                    lat_grid[lat-1, long-1]  = float(np.mean(df_new['LATITUDE']))
                    long_grid[lat-1, long-1] = float(np.mean(df_new['LONGITUDE']))

                    # cell-centered coordinates
                    lat_bin_center[lat-1, long-1]  = (latitude[lat] + latitude[lat-1]) / 2.0
                    long_bin_center[lat-1, long-1] = (longitude[long] + longitude[long-1]) / 2.0

                    # populate test set
                    x_test[year, lat-1, long-1] = float(np.mean(df_new[feature]))

        # Keep track of number of grid cells per year
        grid_count.append(int(yearly_grid_number))

        # Compute average number of stations per grid cell
        avg_cell_size = np.mean(size_array)
        print('Average number of stations per grid cell (rounded down) = ', np.floor(avg_cell_size))
        print('Number of grid cells for current year = ', yearly_grid_number)

    return x_test, lat_grid, long_grid, lat_bin_center, long_bin_center

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
    return data_forecast, model['fit'].coef_[0], model['fit'].intercept_

def RMSe(test, forecast, nH, nLat, nLong):
    errors = np.zeros(shape=(nLat, nLong))
    for i in range(0,nLat):
        for j in range(0,nLong):
            errors[i,j] = root_mean_squared_error(test[0:nH,i,j], forecast[0:nH,i,j])
    return errors 

def form_test_set(data, min_data_pts, years, nlat, nlong, user_feature, test_file, save_OPT):
    # Set aside test set
    xtest, lat_grid_test, long_grid_test, lat_center_test, long_center_test = \
        data_test_set(data=data,
                      min_data_pts=min_data_pts,
                      years=years,
                      nlat=nlat,
                      nlong=nlong,
                      feature=user_feature)
    
    if(save_OPT):
        print('Saving test data to file.')
        for i in range(0,len(years)):
            xtemp = xtest[i,:,:]
            columns_test = {'longitude (mean)':long_grid_test.reshape(-1,1)[:,0],
                            'longitude (cell-centered)':long_center_test.reshape(-1,1)[:,0], 
                            'latitude (mean)':lat_grid_test.reshape(-1,1)[:,0], 
                            'latitude (cell-centered)':lat_center_test.reshape(-1,1)[:,0], 
                            'feature var':xtemp.reshape(-1,1)[:,0]}
            df_test = pd.DataFrame(columns_test)
            df_test_f = df_test[(df_test != 0).all(axis=1)] # Remove zero'd out rows
            df_test_f.to_csv(test_file+'_year_'+str(years[i])+'.csv', index=False)
    else:
        print('Not saving test data to file.')

    return xtest

################### END FUNCTIONS ##############################################

# Build timeline array
nYear      = int(np.ceil((year_end - year_start) / (delta_year)) + 1) # number of years (rounded up)
nTest      = int(np.floor(test_size * nYear))                         # number of years in test set
years      = np.zeros(nYear)                                          # initialize years array
for i in range(0,nYear):
    years[i] = int(year_start + delta_year * i)
print('years = ', years)

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

# Number of cross validation sets
nCross = int(np.floor(len(years[0:nYear-nTest]) - (ltrain + lhorizon)) / iterate) 
if(nCross < 1.):
    print('Number of cross validation sets is invalid')
    print('nYears   = ', len(years[0:nYear-nTest]))
    print('ltrain   = ', ltrain)
    print('lhorizon = ', lhorizon)
    print('iterate  = ', iterate)
    print('nCross = floor[ (nYears - (ltrain + lhorizon) + 1) / iterate ] = ', nCross)
    print('exiting...')
    sys.exit()
print('Number of validation sets: floor[ (nYears - (ltrain + lhorizon) + 1) / iterate ] = ', nCross + 1)

# Initial errors data structure
errors = np.zeros(shape=(nCross+1, nlat, nlong))

# Initialize cross-validation indices
iB_train = 0
iB_test  = ltrain
iE_train = ltrain
iE_test  = ltrain+lhorizon

# plotting parameters
fs_label  = 15
fs_tick   = 14
fs_legend = 13

# Forming the test set
# Note that the test set and training sets are 
# created with different functions, even though 
# the functions appear to be very similar
xtest = form_test_set(data=data, 
                      min_data_pts=min_data_pts, 
                      years=years[nYear-nTest:nYear], 
                      nlat=nlat, 
                      nlong=nlong, 
                      user_feature=user_feature, 
                      test_file=user_feature+'_test_data', 
                      save_OPT=save_test_data)

# Perform cross-validation test
i_cross = 0
while iE_test <= len(years)-nTest:
    years_new = years[iB_train:iE_test]

    # Form cross validation splits with train data + clean data (remove NaNs)
    xtrain, xvalid, lat_grid, long_grid, lat_center, long_center, gridcount = \
        data_clean_split_single_feature(data=data,
                                        min_data_pts=min_data_pts,
                                        years=years_new,
                                        nTrain=ltrain,
                                        nHorizon=lhorizon,
                                        nlat=nlat,
                                        nlong=nlong,
                                        feature=user_feature,
                                        plotdata=plotdata)
        
    # Fit model and forecast
    xforecast, xcoef, xintercept  = forecasting(years=years_new, nlat=nlat, nlong=nlong,
                                                nTrain=ltrain, nHorizon=lhorizon, traindata=xtrain[iB_train:iE_train,:,:],
                                                reg_model=reg_model)
    
    # Compute errors and save to file (if selected)
    errors[i_cross, :, :] = RMSe(test=xvalid, forecast=xforecast, nH=lhorizon, nLat=nlat, nLong=nlong)

    if(save_errors and iE_test == len(years)-nTest):
        
        # Compute projection to nth year in the future (for demonstration purposes)
        years_forecast_Project = np.linspace(years_new[len(years_new)-1],year_end_forecast,nYfc)
        forecast_Project       = np.zeros(shape=(len(years_forecast_Project),nlat,nlong))
        coef_Project           = np.zeros(shape=(nlat,nlong))
        intercept_Project      = np.zeros(shape=(nlat,nlong))
        feature_delta          = np.zeros(shape=(nlat,nlong))
        frac_diff              = np.zeros(shape=(nlat,nlong))
        pltpoint_size          = np.ones(shape=(nlat,nlong)) * 5.
        xtrain_mean            = np.abs(np.mean(xtrain[iB_train:iE_train,:,:], axis=0))
        lat_2D                 = np.zeros(shape=(nlat, nlong))
        long_2D                = np.zeros(shape=(nlat, nlong))
        for i in range(0,nlat):
            for j in range(0,nlong):
                lat_2D[i,j] = latlabel[i]-1
                long_2D[i,j] = longlabel[j]-1
                forecast_Project[:,i,j], coef_Project[i,j], intercept_Project[i,j] = \
                    model_formation(time_train=years_new[0:ltrain].reshape(-1,1), data_train=xtrain[0:ltrain,i,j], model=reg_model, time_forecast=years_forecast_Project[:].reshape(-1,1))
                feature_delta[i,j] = forecast_Project[nYfc-1,i,j] - forecast_Project[0,i,j]

                # Compute performance measure (RMS error wrt train data mean)
                if(xtrain_mean[i,j]!=0.):
                    frac_diff[i,j] = 100. * np.abs(1. - np.abs(xtrain_mean[i,j] - errors[i_cross,i,j]) / xtrain_mean[i,j])

        # Build data structure for errors + model properties
        error_file = 'error_ncross_'+str(i_cross+1)+'_minYear_'+str(years[0])+'_maxYear_'+str(years[len(years)-1])+'_var_'+user_feature
        columns = {'longitude (mean)':long_grid.reshape(-1,1)[:,0],
                   'longitude (cell-centered)':long_center.reshape(-1,1)[:,0], 
                   'latitude (mean)':lat_grid.reshape(-1,1)[:,0], 
                   'latitude (cell-centered)':lat_center.reshape(-1,1)[:,0], 
                   'intercept':xintercept.reshape(-1,1)[:,0], 
                   'slope':xcoef.reshape(-1,1)[:,0], 
                   'RMS errors (relative to final validation set, NOT test set)':errors[i_cross, :, :].reshape(-1,1)[:,0],
                   'feature delta':feature_delta.reshape(-1,1)[:,0],
                   'Performance measure (%)':frac_diff.reshape(-1,1)[:,0],
                   'plot point size':pltpoint_size.reshape(-1,1)[:,0],
                   'lat label': lat_2D.reshape(-1,1)[:,0],
                   'long label': long_2D.reshape(-1,1)[:,0]}
        df_errors = pd.DataFrame(columns)
        df_filter = df_errors[(df_errors != 0).all(axis=1)] # Remove any zero entries
        df_filter.to_csv(error_file+'.csv', index=False)

        # Plot only the final validation step for a specified coordinate
        if(plot_last_validation):
            plt.figure(i_cross)
            plt.plot(years[0:iE_train], xtrain[0:iE_train, ilat, ilong], linewidth=3.0, color='black', label='train data')
            plt.plot(years[iB_test:iE_test], xforecast[:, ilat, ilong], linewidth=1.5, color='red', linestyle='dashed', label=r'forecast')
            plt.plot(years[iB_test:iE_test], xvalid[:, ilat, ilong],  linewidth=3.0, color='blue', label='validation data')
            plt.plot(years_forecast_Project, forecast_Project[:, ilat, ilong], linewidth=3.0, color='magenta', linestyle='dotted', label=r'data projection, $\Delta \mathrm{f}=$' + '{:.3e}'.format(feature_delta[ilat, ilong]))
            plt.title('lat x long = ' + '{:.3f}'.format(lat_grid[ilat,ilong]) + ' x ' + '{:.3f}'.format(long_grid[ilat,ilong]) + ', RMS error: ' + '{:.3e}'.format(errors[i_cross, ilat, ilong]) + ', ncross = ' + str(i_cross+1))
            plt.tick_params(axis='x', labelsize=fs_tick)
            plt.xticks(years[0:iE_test], [int(label) for label in years[0:iE_test]], rotation=45, size=7)
            plt.xlabel(r'Year', fontsize = fs_label)
            plt.tight_layout()
            if(user_feature=='TMAX'):
                plt.ylabel(r'$T_{\mathrm{max}}$ [K]', fontsize = fs_label)
            elif(user_feature=='SNOW'):
                plt.ylabel(r'Snowfall [mm]')
            elif(user_feature=='PRCP'):
                plt.ylabel(r'Rainfall [mm]')
            elif(user_feature=='HTDD'):
                plt.ylabel(r'Heating degree days')
            elif(user_feature=='TAVG'):
                plt.ylabel(r'$T_{\mathrm{avg}}$ [K]')
            elif(user_feature=='TMIN'):
                plt.ylabel(r'$T_{\mathrm{min}}$ [K]')
            plt.legend(fontsize = fs_legend)

    # Plot results at all steps (if selected)
    if(plot_each_validation):
        plt.figure(i_cross)
        plt.plot(years[0:iE_train], xtrain[0:iE_train, ilat, ilong], linewidth=3.0, color='black', label='train data')
        plt.plot(years[iB_test:iE_test], xforecast[:, ilat, ilong], linewidth=1.5, color='red', linestyle='dashed', label=r'forecast')
        plt.plot(years[iB_test:iE_test], xvalid[:, ilat, ilong],  linewidth=3.0, color='blue', label='validation data')
        plt.title('lat x long = ' + '{:.3f}'.format(lat_grid[ilat,ilong]) + ' x ' + '{:.3f}'.format(long_grid[ilat,ilong]) + ', RMS error: ' + '{:.3e}'.format(errors[i_cross, ilat, ilong]) + ', ncross = ' + str(i_cross+1))
        plt.tick_params(axis='x', labelsize=fs_tick)
        plt.xticks(years[0:iE_test], [int(label) for label in years[0:iE_test]], rotation=45, size=7)
        plt.xlabel(r'Year', fontsize = fs_label)
        plt.tight_layout()
        if(user_feature=='TMAX'):
            plt.ylabel(r'$T_{\mathrm{max}}$ [K]', fontsize = fs_label)
        elif(user_feature=='SNOW'):
            plt.ylabel(r'Snowfall [mm]')
        elif(user_feature=='PRCP'):
            plt.ylabel(r'Rainfall [mm]')
        elif(user_feature=='HTDD'):
            plt.ylabel(r'Heating degree days')
        elif(user_feature=='TAVG'):
            plt.ylabel(r'$T_{\mathrm{avg}}$ [K]')
        elif(user_feature=='TMIN'):
            plt.ylabel(r'$T_{\mathrm{min}}$ [K]')
        plt.legend(fontsize = fs_legend)

    # Iteration cross validation indices
    ltrain  += iterate
    iB_test  = ltrain
    iE_train = ltrain
    iE_test  = ltrain+lhorizon
    i_cross += 1

# Plot RMS error vs nCross for user specified coordinate point
crossValidation_ARR = [i for i in range(0,nCross+1)]
plt.figure(i_cross+1)
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
