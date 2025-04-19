<h1> data_science_boot_camp_2025_project </h1>
<h2> Group members: Alison Duck, Jack Neustadt, David Pochik, Tawny Sit </h2>

<p> <strong> Today's Texas Might be Tomorrow's Ohio: Building a Geographic Climate Change Predictor </strong> </p>

<ol>
<li> <strong>Background:</strong> From the dawn of industrialization to today, the average global temperature has shifted upward by ~2.7 degrees Fahrenheit (~1.5 degrees Celsius) due to increased greenhouse gas emissions. The effects of this temperature increase have led to, among many other complications, more extreme weather patterns, increased sea levels, and hotter days. If emissions are left unchecked and temperatures continue to rise at their current (or projected) rate, then this will lead to drastic shifts in regional climate. For example, today's annual average temperature in Ohio will increase to that of today's annual temperature in Texas in Y years. </li>

<li> <strong>Goal:</strong> Build a model that predicts the temperature in region(s) X after Y years and use this model to determine where current climates will be sustained in the future, i.e., If I want to live in an area Y years from now that has roughly the same temperature or climate as region X today, where would I go? </li>

<li> <strong>Data Source(s):</strong> Global yearly climate data and the stations used to collect these data are provided by the National Climatic Data Center (NCDC). If needed, other data sources may include the NASA climate database or the National Oceanic and Atmospheric Administration. </li>

<li> <strong>Model features:</strong> spatial location, elevation, temperature (min/max/mean), snowfall, precipitation, heating degree days </li>

<li> <strong>Data refinement / observation:</strong> Perform routine data cleaning, i.e., eliminate NAN entries, convert data to appropriate formatting, and identify categorical variables. Begin by creating plots of target variables (temperature, or possible something else) as functions of the various model features to ascertain data relationships before performing statistical analysis. Some "location-averaging" may be required to allow for comparisons of disparate datasets.</li>

<li> <strong>Modelling methods:</strong> time-series analysis, linear regression models (SLR), quadratic regression </li>

<li> <strong>Key Performance Indicators:</strong> Benchmarking model by showing that our model predicts the current average temperatures (monthly/annual) based on prior data (Z years ago) within some reasonable RMSe. </li>

<li> <strong>Stakeholders:</strong> Homeowners Y years from now, energy suppliers, climate change education programs, realty agencies, home insurance providers. </li>

<li> <strong>Purpose:</strong> Extrapolate historic geographical climate trends to modern and future locations. Provide details for how much regions will change if their temperatures change by a rate Z over a time period Y. These models will be useful for putting climate change in understandable terms for a general audience and provide concrete examples of its impact. </li>
