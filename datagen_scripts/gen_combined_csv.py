import numpy as np
import pandas as pd
import os

# change to your location of the big list of all stations
stations_fpath = 'data/ghcnd-stations.txt'

# change to where you extracted the station-by-station tar.gz file
station_by_station_data_dir = 'C:/Users/tawny/Documents/station_by_station_data/'

# change to where you want to save the final combined file
save_dir = 'data'

# read in list of all stations
stations = pd.read_fwf(stations_fpath, colspecs=[(0,11), (12,21), (21,31), (31,38), (38,41), (41,72), (72,76), (76,80), (80,86)], 
                       names=['ID', 'Lat', 'Lon', 'Elevation', 'State', 'Name', 'GSN', 'HCN/RCN', 'WMO'])

# select only US stations, then get rid of AK and HI
us_stations = stations[np.array(['US' in stations['ID'][i] for i in range(len(stations))])]
us_stations = us_stations[us_stations['State']!='AK']
us_stations = us_stations[us_stations['State']!='HI']

# get stations that actually have GSOY data
us_stations_with_data = np.full(len(us_stations), False)
for i, station_id in enumerate(us_stations['ID'].values):
    us_stations_with_data[i] = os.path.exists(f'{station_by_station_data_dir}/{station_id}.csv')
us_stations = us_stations[us_stations_with_data]

# read and concatenate all the station-by-station data
station_dfs = [pd.read_csv(f'{station_by_station_data_dir}/{station_id}.csv') for station_id in us_stations['ID'].values]
gsoy_df = pd.concat(station_dfs, ignore_index=True)

# do a bit of cleaning
gsoy_df.drop(list(gsoy_df.filter(regex='_ATTRIBUTES')), axis=1, inplace=True)
gsoy_df['STATE'] = np.array([gsoy_df['NAME'].values[i][-5:-3] for i in range(len(gsoy_df))])
gsoy_df['T_DIFF'] = gsoy_df['TMAX'].values - gsoy_df['TMIN'].values

# save!
gsoy_df.to_csv(f'{save_dir}/combined_us48_GSOY_data.csv', index=False)