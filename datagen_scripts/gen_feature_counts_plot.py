import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import seaborn as sns

sns.set_palette("colorblind")

# load full combined dataset
gsoy_df = pd.read_csv('data/combined_us48_GSOY_data.csv')

# select only the years we'll use in our analysis
gsoy_df_cropped_years = gsoy_df[gsoy_df['DATE'].between(1949,2024)]

# list of features we will use
features = ['TMIN', 'TMAX', 'TAVG', 'SNOW', 'PRCP', 'HTDD']

# add the STATION column for reference
features_plus_station = ['STATION'] + features

# generate pandas groupby plot - all years on left, 1949-2024 on right
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,5),sharey=True, constrained_layout=True)
gsoy_df.groupby('DATE')[features_plus_station].count().plot(ax=ax1)
gsoy_df_cropped_years.groupby('DATE')[features_plus_station].count().plot(ax=ax2, legend=False)

# a bunch of formatting stuff
for ax in [ax1, ax2]:
    ax.set_ylabel('Count', fontsize=14)
    ax.set_xlabel('Year', fontsize=14)
    ax.tick_params(axis='y', which='both', direction='in', right=True, labelsize=12)
    ax.tick_params(axis='x', which='both', direction='in', top=True, labelsize=12)
ax1.xaxis.set_minor_locator(MultipleLocator(5))
ax2.xaxis.set_minor_locator(MultipleLocator(2))
ax1.yaxis.set_minor_locator(AutoMinorLocator(4))
ax1.legend(fontsize=10)

# color the years we actually look at on the left plot
ax1.axvspan(1949, 2024, color='gray', alpha=0.1, ec='none')

# indicate the year (1949) we will train the KMeans clusterer on for binning
ax2.axvline(1949, color='tab:red', alpha=0.1)

# indicate training/cross-validation and test sets
ax2.axvspan(1950, 2010, color='tab:blue', alpha=0.1, ec='none')
ax2.axvspan(2011, 2024, color='tab:orange', alpha=0.1, ec='none')

# save figure!
plt.savefig('num_features_and_traintest_years.png', bbox_inches='tight')