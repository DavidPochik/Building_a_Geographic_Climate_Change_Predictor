<h1> data_science_boot_camp_2025_project </h1>
<h2> Group members: Alison Duck, Jack Neustadt, David Pochik, Tawny Sit </h2>

<p> <strong>Today's Texas Might be Tomorrow's Ohio: Building a Geographic Climate Change Predictor</strong> </p>

This project explores and analyzes geographical climate change data in the United States of America from 1950 to the current year. The objective is to <strong>predict</strong> regional features, e.g., temperature, precipication, or snowfall, for a given year based on historical data.

This project uses raw data from the National Climatic Data Center and the U.S. Energy Information Administration.

The major steps in this project include:
<ol>
<li>Obtain and clean climate data. </li>
<ol>
<li>Acquire climate data for the entire United States from 1950 until now ($\sim10^6$ weather station datapoints) </li>
<li>Remove 'dirty' data, i.e., NaNs or empty entries. </li>
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
</ol>
<ol>
<li>Use best performing models to make future predictions on climate data</li>
<ol>
<li>Compare climate features between different regions and address the statement of the proposal</li>
</ol>

<code>datagen_scripts</code> contains the script(s) used to generate the <code>.csv</code> files used for this project

<code>deliverables</code> contains the project proposal, mission plan, KPIs, and list of stakeholders.

<code>analysis_scripts</code> contains the scripts used for performing exploratory data analysis, creating structured grids, building predictive models, and performing error analysis..



