<h1> data_science_boot_camp_2025_project </h1>
<h2> Group members: Alison Duck, Jack Neustadt, David Pochik, Tawny Sit </h2>

<p> <strong>Today's Texas Might be Tomorrow's Ohio: Building a Geographic Climate Change Predictor</strong> </p>

This project explores and analyzes geographical climate change data in the contiguous United States from 1950 to the current year. The objective is to <strong>predict</strong> regional features, e.g., temperature, precipication, or snowfall, for a given year based on historical data.

This project uses raw data from the National Climatic Data Center.

## Project Overview
The major steps in this project include the following:
<ol>
<li>Obtain and clean climate data. </li>
<ol>
<li>Acquire climate data for the entire United States from 1950 until now ($\sim10^6$ weather station datapoints) </li>
<li>Remove 'dirty' data, i.e., NaNs or empty entries. </li>
<li>Limit our scope to the contiguous United States for simplicity.</li>
<li>Compute mean features in each discretized region (see next list entry) to simplify data overhead and allow for broader comparisons.</li>
</ol>
<li>Spatially organizing data </li>
<ol>
<li> Static latitude/longitude grids. </li>
<li> Static K-means clustered regions. </li>
</ol>
<li>Perform statistics/regression routines and make predictions</li>
<li>Perform error analysis to determine model performance</li>
<ol>
<li>Evaluate KPIs </li>
<li>Use best performing models to make future predictions on climate data</li>
</ol>
<li>Compare climate features between different regions and address the statement of the proposal</li>
</ol>

## Directories and Scripts
<code>deliverables</code> contains the project proposal, mission plan, KPIs, and list of stakeholders.

<code>datagen_scripts</code> contains the script(s) used to generate the <code>.csv</code> files used for this project
<ol>
<li><code>gen_combined_csv.py</code>: Takes raw NOAA climate base data and compiles a single <code>.csv</code> file with all the data.</li>
<li><code>gen_feature_counts_plot.py</code>: Plots number of data points per year for each weather station and feature variable.</li>
</ol>

<code>plots</code> contains select figures from the exploratory data analysis process under the <code>EDA</code> subdirectory and final presentation plots under the <code>Presentable</code> subdirectory.

<code>data</code> contains the zipped data used for this project.
<ol>
<li>Execute <code>cd data/</code> and <code>tar -xzvf compressedFile.tar.gz</code> to extract the data used in this project.</li>
</ol>

<code>analysis_scripts</code> contains the scripts used for performing exploratory data analysis, creating structured grids, building predictive models, and performing error analysis.
<ol>
<li><code>latitude_longitude_grid.py</code>: Organizes climate data into user-specified latitude/longitude grid lines with refinement <code>nlat</code> and <code>nlong</code>. Defaults are set to <code>10</code>. Computes mean temperature (min, max, and avg), precipitation, snowfall, and heating degree days within each grid cell for user-specified <code>years</code>. Creates a train/test time-series split with n-fold cross validation and performs simple linear regression on mean climate quantities.</li>
<ol>
<li>User must specify the directory in which the data are stored when running the script</li>
</ol>
<li><code>KMeans_Binning.py</code>: </li>
<li><code>alison_temp_map.py</code>: </li>
</ol>

## Software Requirements
The analysis scripts run on <code>Python</code> (version 3.12.x or later). The Python packages required for running all scripts are:
<ol>
<li><code>pandas</code></li>
<li><code>numpy</code></li>
<li><code>matplotlib</code></li>
<li><code>seaborn</code></li>
<li><code>sklearn</code></li>
<li><code>sys</code></li>
<li><code>plotly.express</code></li>
<li><code>plotly.io</code></li>
<li><code>os</code></li>
<li><code>cartopy</code></li>
</ol>
