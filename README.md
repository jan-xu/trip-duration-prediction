# Trip Duration Prediction

## The Problem

Given training data (```train.csv```) on Uber/Lyft trip OD locations, starting time and durations, we would like to estimate travel times between two specified locations at a given departure time using machine learning models such as SVR, random forest, neural networks. Quality of the classifiers is assessed using root mean squared error (RMSE) of the predicted versus the actual durations on the test set (```test.csv```). A Kaggle competition was setup to evaluate the results on a public test data set (20% of total test set). The link to the Kaggle competition can be found here: https://www.kaggle.com/c/ce263n-hw4

### Training Set
The training dataset (```train.csv```) contains a csv file with ride start and end locations (specified as WGS84 coordinates), trip start time (local time), and trip duration in seconds.

Each line is a trip and has the following format:

```row_id, start_lng, start_lat, end_lng, end_lat, datetime, duration```

### Test Set
The test dataset (```test.csv```) contains a csv file with ride start and end locations (specified as WGS84 coordinates), and trip start time (local time). You need to build model based on the training set and estimate the travel times for all trips in the test set.

Each line in the test set has the following format:

```row_id, start_lng, start_lat, end_lng, end_lat, datetime```

## Solution Strategy

With only coordinate data of origin and destination of each trip, as well as time of the trip, it is essential to break down the components of this data. We want information about the distance travelled, any references of the timeframe, and potentially additional information about the conditions of each trip.

### *Step 1: Exploratory Data Analysis*
Looking at the training dataset, it is evident that the trips are split into two geographical regions: San Francisco (data from 2012) and New York (data from 2015). In the coordinate data, there are some missing values (removed), misrepresentations such as positive longitude values (corrected to negative) and false data (end_lng ≈ -50 -> destination is in the Atlantic Ocean; these data points were dropped).

Looking further into the training data, some of the trips have zero seconds of duration. Unless the start and end coordinates are the same, this must be a data error. These lines were hence dropped. Also, the duration of some trips is capped at 40000 seconds (≈ 11 hours). These trips have also been dropped, as well as all trips above 20000 seconds (≈ 6 hours).

### *Step 2: Feature Engineering*
Firstly, to estimate distance travelled, three methods have been considered: a) Manhattan distance (dlat + dlng), b) Euclidean distance (using the Haversine formula) and c) Google Maps distance. The Google Maps distance was retrieved by using the Distance Matrix API from Google Cloud Platform (from which an estimate of the trip duration was also retrieved). All of the Google Maps distances of trips traveling to or from Treasure Island outside SF were misrepresented as zero; these were substituted with the Manhattan distance and the Google Maps durations were estimated with the median speed (11 m/s).

Secondly, the time of the trip can be split into useful components on which travel time might depend upon: weekday (1-7), hour (0-23) and public holidays (1 for holiday, 0 otherwise). It can be seen that during public holidays such as Thanksgiving, Christmas, Labor Day or New Years, traffic volume is reduced.

Thirdly, geolocations have been considered by clustering the coordinate data using DBSCAN. All neighborhoods with a 500m radius with at least 2500 samples are considered a geolocation. The clustering process yielded seven such neighbourhoods, three of which were airports (JFK, LaGuardia and SFO) and the other four city centers of SF and NY. All the points that were not clustered were considered “standalone”. The labels were thus “airport”, “citycenter” and “standalone” (one-hot encoded).

Fourthly, weather conditions may influence travel time. Hence, historical precipitation data in NY and SF, taken from NOAA, was also included.

Lastly, all trips with a small Manhattan distance were marked either as “short_trip” (if gmaps distance also was small) or as “routing_error” (if the gmaps distance was long) using one-hot encoding.

### *Step 3: Prediction Model*
Many classic ML algorithms were employed to find the optimal prediction score; Linear Regression, Ridge, Lasso, SVM Regressor, Random Forest, XGBoost and FFNN. A stacked model was also considered but underperformed in terms of prediction score. The model that worked best was XGBoost, with hyper parameters ```max_depth=9```, ```learning_rate=0.045```, ```n_estimators=500```, ```reg_lambda=0.5```.

Kaggle RMSE score on public dataset with this model: **287.02604**

## Real-time implementation

This model could be very valuable for travel planning applications, which needs to know very accurately the travel time of trips in order to estimate trip fares, travel demand and fleet size amongst others. A real-time implementation of this model is very simple, since all it takes is inputting the start and end coordinates, the time of the trip and the weather data to estimate the trip duration. Knowing the correlation between the features of interest with trip duration is also useful for strategic purposes such as surge pricing, service coverage depending on the date or weather etc.
