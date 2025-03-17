# Load in some modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import plotly.express as px
import plotly.io as pio
sns.set_style("whitegrid")

# Specify directory string in which you have the data file stored
user_specified_directory = '/mnt/d/Erdos_DIR/project_data_TAWNYFILE_DIR/combined_us_GSOY_data.csv'

# Read in data
data = pd.read_csv(user_specified_directory)

######################## INPUTS ###################################
# Specify desired grid lengths
nlat         = 10
nlong        = 10

# Specify minimum number of data points per grid cell
# Ideally choose anything greater than or equal to one
min_data_pts = 1

# Specify years you want to look at
years = np.array([1960, 1970, 1980, 1990, 2000, 2010, 2020])

# plot parameters
plotzoom  = 3.0
plotlat   = 37
plotlong  = -95
plotrad   = 20.0
soln_type = np.array(['TEMP', 'SNOW', 'PRCP'])
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

avg_cell_size = []
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

    for lat in latlabel:
        for long in longlabel:
            # Creating temporary dataframe that includes data only in the grid cell specified by lat and long
            df_filtered = data[(data['LATITUDE_BIN'] == latlabel[lat-1]) & 
                               (data['LONGITUDE_BIN'] == longlabel[long-1]) & 
                               (data['DATE'] == years[year])]

            # Removing any rows that have NaNs in any of the model features.
            df_new = df_filtered.dropna(subset=['TMAX', 'PRCP', 'SNOW'])
            num_stations_per_cell = df_new.shape[0]

            # Conditional statement checks to see that there are at least an 'min_data_pts' amount of non-NaN rows in this grid cell
            if(num_stations_per_cell+1>min_data_pts):
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

    # Compute average number of stations per grid cell
    avg_cell_size = np.mean(size_array)
    print('Average number of stations per grid cell (rounded down) = ', np.floor(avg_cell_size))

    # Build new dataframe with grid-cell-averaged data
    coarse_df = pd.DataFrame.from_dict({'TEMP' : temp_array, 'PRCP' : prec_array, 'SNOW' : snow_array, 
                                        'LATITUDE' : bin_lat_avg, 'LONGITUDE' : bin_long_avg, 
                                        'LATITUDE_B' : bin_lat_B, 'LONGITUDE_B' : bin_long_B, 
                                        'SIZE' : size_array})

    # Plot all solution types for the current year in the loop
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