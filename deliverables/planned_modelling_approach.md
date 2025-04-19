<h1> Today's Texas Might be Tomorrow's Ohio: Building a Geographic Climate Change Predictor </h1>
<h2> Alison Duck, Jack Neustadt, David Pochik, and Tawny Sit </h2>

<p> <strong> Planned modelling approach (with minor progress updates) </strong> </p>

<ol>
<li> To answer the question “where should I move Y years from now to experience the same climate (e.g., temperature, amount of precipitation, and amount of snow) as region X today,” we will compare climate properties between discrete regions over some timeline. </li>

<li> Given that the climate features of our data change as functions of time, we will model our data using time series analysis. </li>

<ol>
</li> We aim to forecast these climate features (temperature, precipitation, and snow) based on geographic location. </li>
</ol>

<li> We will investigate potential co-dependencies between these features, e.g., for a specified year, does one climate feature trend with another? </li>

<li> Due to historical data scarcity, we will only focus on data from 1950 and after. </li>

</li> We will use multiple methods to handle regional discretization. </li>

<ol>
<li> Binning regional data into longitudinal/latitudinal grid cells with some specified resolution and minimum number of station data per grid cell. </li>

<li> Using a proximity classification routine (K-means clustering) to form N number of regional clusters at some initial year (e.g., 1950), then using these regions to bin data for successive years. </li>

<ol>
<li> We may weigh the pros and cons of using either of our regional discretization schemes. </li>
</ol>
</ol>

<li> After regional selection, we will perform a time series train-test data split. </li>

<ol>
<li> We will begin by separating temporal observations by decades and will refine our approach later if necessary. </li>
<ol>
<li>We will first use a horizon length of 10 years and perform a k=7 cross validation split. </li>
</ol>
</ol>

<li> We will investigate the efficacy of SLR for forecasting regional climate features. </li>

<ol>
<li> We will perform error analysis (RMS error or F-test) on our fitted models. </li>

<li> We may investigate power law or polynomial fitting techniques in addition to SLR. </li>
</ol>

<li> Once we are confident in the accuracy of our model, we can begin predicting regional climate features and making direct comparisons between regions specified by one of our discretization techniques. </li>

<ol>
<li> We will finally be in the position to answer the big question of this project! </li>
</ol>
</ol>
