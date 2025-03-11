# Load in some modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import plotly.express as px
sns.set_style("whitegrid")

# Specify directory string in which you have the data file stored
user_specified_directory = '/mnt/d/Erdos_DIR/project_data_TAWNYFILE_DIR/combined_us_GSOY_data.csv'

# Specify desired grid lengths
nlat         = 50
nlong        = 100

# Read in data -> This step might take a while!
data = pd.read_csv(user_specified_directory)

# Finding latitude and longitude minima/maxima
latmin  = np.min(data['LATITUDE'].astype(float))
latmax  = np.max(data['LATITUDE'].astype(float))
longmin = np.min(data['LONGITUDE'].astype(float))
longmax = np.max(data['LONGITUDE'].astype(float))

# Creating latitude and longitude grids
latitude  = np.linspace(latmin, latmax, nlat)
longitude = np.linspace(longmin, longmax, nlong)

# Creating bin labels for latitude and longitude.
# Note: pd.cut requires these to be 1 less in length than latitude and longitude, hence nlat-1 and nlong-1
latlabel  = [i+1 for i in range(0,nlat-1)]
longlabel = [i+1 for i in range(0,nlong-1)]

# Labelling latitude and longitude coordinates from the data with a grid label.
data['LATITUDE_BIN']  = pd.cut(x=data['LATITUDE'],  bins=latitude,  labels=latlabel)
data['LONGITUDE_BIN'] = pd.cut(x=data['LONGITUDE'], bins=longitude, labels=longlabel)

# Arrays in which model features will be stored
temp_array = []
prec_array = []
snow_array = []

# Arrays in which mean latitude/longitude coordinates for all data belonging to a grid cell will be stored
bin_lat    = []
bin_long   = []
for lat in latlabel:
    for long in longlabel:
        # Creating temporary dataframe that includes data only in the grid cell specified by lat and long
        df_filtered = data[(data['LATITUDE_BIN'] == latlabel[lat-1]) & (data['LONGITUDE_BIN'] == longlabel[long-1])]

        # Removing any rows that have NaNs in any of the model features.
        df_filtered.dropna(subset=['TMAX', 'PRCP', 'SNOW'])

        # Conditional statement checks to see that there is at least one non-NaN row in this grid cell
        if(df_filtered.shape[0]>0):
            # Compute mean model features within this grid cell
            temp_array.append(float(np.mean(df_filtered['TMAX'])))
            prec_array.append(float(np.mean(df_filtered['PRCP'])))
            snow_array.append(float(np.mean(df_filtered['SNOW'])))

            # Compute mean lat/long coordinate to obtain a 'representative' coordinate per grid cell
            bin_lat.append(float(np.mean(df_filtered['LATITUDE'])))
            bin_long.append(float(np.mean(df_filtered['LONGITUDE'])))

# Build new dataframe with grid-cell-averaged data
coarse_df = pd.DataFrame.from_dict({'TEMP' : temp_array, 'PRCP' : prec_array, 'SNOW' : snow_array, 'LATITUDE' : bin_lat, 'LONGITUDE' : bin_long})

# Plot!
fig = px.density_map(coarse_df, lat='LATITUDE', lon='LONGITUDE', z='TEMP', radius=10,
                        center=dict(lat=37, lon=-95), zoom=1.4,
                        map_style="open-street-map",width=1000,height=700)
fig.show()
