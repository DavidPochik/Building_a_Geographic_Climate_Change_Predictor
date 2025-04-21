<h1> data_science_boot_camp_2025_project </h1>
<h2> Group members: Alison Duck, Jack Neustadt, David Pochik, Tawny Sit </h2>

<p> <strong>Today's Texas Might be Tomorrow's Ohio: Building a Geographic Climate Change Predictor</strong> </p>

## Project Overview
From the dawn of industrialization to today, the average global temperature has shifted upward by ~2.7 degrees Fahrenheit (~1.5 degrees Celsius) due to increased greenhouse gas emissions. The effects of this temperature increase have led to, among many other complications, more extreme weather patterns, increased sea levels, and hotter days. If emissions are left unchecked and temperatures continue to rise at their current (or projected) rate, then this will lead to drastic shifts in regional climate. For example, today's annual average temperature in Ohio will increase to that of today's annual temperature in Texas in Y years.

This project explores and analyzes geographical climate change data in the contiguous United States from 1950 to the current year. The objective is to <strong>predict</strong> regional features, e.g., temperature, precipitation, or snowfall, for a given year based on historical data, i.e., if I want to live in an area Y years from now that has roughly the same temperature or climate as region X today, where would I go?

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

## Project data
This project uses raw data from the NOAA National Climatic Data Center. <code>data</code> contains the zipped data used for this project.
<ol>
<li>Execute <code>cd data/</code> and <code>tar -xzvf compressedFile.tar.gz</code> to extract the data used in this project.</li>
</ol>

The scripts under <code>datagen_scripts</code> may be used to produce different data sets if the user provides the data files.

## Directories and Scripts
<code>deliverables</code> contains the project proposal, mission plan, KPIs, and list of stakeholders.

<code>datagen_scripts</code> contains the script(s) used to generate the <code>.csv</code> files used for this project
<ol>
<li><code>gen_combined_csv.py</code>: Takes raw NOAA climate base data and compiles a single <code>.csv</code> file with all the data.</li>
<li><code>gen_feature_counts_plot.py</code>: Plots number of data points per year for each weather station and feature variable.</li>
</ol>

<code>executive_summary</code> contains the document that summarizes the approach + findings of this project

<code>presentation_slides</code> contains the document used to present this project.

<code>plots</code> contains select figures from the exploratory data analysis process under the <code>EDA</code> subdirectory and final presentation plots under the <code>Presentable</code> subdirectory.

<code>analysis_scripts</code> contains the scripts used for performing exploratory data analysis, creating structured grids, building predictive models, and performing error analysis.
<ol>
<li><code>latitude_longitude_grid.py</code>: Organizes climate data into user-specified latitude/longitude grid lines with refinement <code>nlat</code> and <code>nlong</code>. Defaults are set to <code>10</code>. Computes mean temperature (min, max, and avg), precipitation, snowfall, and heating degree days within each grid cell for user-specified <code>years</code>. Creates a train/test time-series split with n-fold cross validation and performs simple linear regression on mean climate quantities.</li>
<li><code>KMeans_Binning.py</code>: Performs the K-means clustering and KNN classification binning method. Generates a CSV file of the binned data, (<code>binned_k100_1950to2024.csv</code>), a CSV file of the individual stations from 1950 to 2024 tagged by cluster membership (<code>combined_GSOY_us48_1950to2024_clustertagged.csv</code>), a decision boundary plot for every year from 1950 to 2024, and the GIF combining the plots (<code>k100_binning_animated.gif</code>).</li>
<li><code>preliminary_temperature_map.py.py</code>: This script is a paired down version of the full gridding method and linear regression for the average daily maximum temperature for each year. The script requires the data <code>combined_us_GSOY_data.csv</code> to be in the same directory as the script itself. The script uses the information from US weather stations from 1950 to 2010 and places each station into one of 100 equally sized bins. It then computes the average daily maximum temperature for each year across all the stations in each bins. Next it preforms a linear regression and computes the room mean square error to assess the suitability of the regression to describe our data set. These values are reported in a csv file saved to the same directory as the script.</li>
<li><code>KMeans_Cluster_Regression.ipynb</code>: Jupyter notebook for performing regression on the K-Means+KNN binned data. Both linear and quadratic regression models are explored for the six features considered in the project using 5-fold cross validation (horizon length = 10 years) and explanation of final model selection. Performs preliminary analysis on final RMSE values of the test set. Generates external figures for all 100 k-means clusters showing both models for the final validation set (validated on 2011-2020) and the final selected model for the test set (2021-2024). Generates prerequisite CSV files for running <code>classification_demo.ipynb</code> (<code>quadratic_preferred.csv</code>, <code>2050_quadratic_diffs.csv</code>, and <code>2050_linear_diffs.csv</code>).</li>
<li><code>classification_demo.ipynb</code>: This is a notebook that runs the climate change classification from building the classifier to re-classifying weather stations based on climate-change trends.</li>
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
<li><code>glob</code></li>
<li><code>PIL (pillow)</code></li>
</ol>
