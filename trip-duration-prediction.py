"""
# Trip Duration Prediction
# Jan Xu
# 05/05/2019
"""

##################
# Import modules #
##################

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

import datetime
import random
import time
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, BatchNormalization
from xgboost import XGBRegressor

import gmplot
import googlemaps

#######################
# Step 0: Import data #
#######################

train_df = pd.read_csv("dataset/train.csv", index_col="row_id")
test_df = pd.read_csv("dataset/test.csv", index_col="row_id")
combine = [train_df, test_df]

# Visualize training data points
gmap = gmplot.GoogleMapPlotter(train_df.start_lat.mean(), train_df.start_lng.mean(), 4)
gmap.heatmap(train_df.start_lat, train_df.start_lng)

# Store html with geospatial Google Maps heatmap
gmap.draw("gmaps/heatmap.html")

#####################################
# Step 1: Exploratory data analysis #
#####################################

# Convert to datetime and drop NA's
for df in combine:
    df.datetime = pd.to_datetime(df.datetime)
    df.dropna(axis=0, inplace=True)

# Drop invalid routes
train_df.drop([90810, 81553, 83473, 87892], axis=0, inplace=True)

# Drop trips above 20000 seconds
train_df.drop(train_df[train_df.duration > 20000.0].index, axis=0, inplace=True)

# Drop zero durations
train_df.drop(train_df.loc[(train_df.duration == 0) &
                           ((train_df.start_lng != train_df.end_lng) |
                            (train_df.start_lat != train_df.end_lat))].index, axis=0, inplace=True)

# Check datetime values
train_df.datetime.hist(figsize=(6,6))
plt.show()

# Check all other features
train_df.hist(figsize=(6,6))
plt.show()

# Check coordinate anomalies in start_lng
mismatched = train_df[train_df.start_lng > 50]
gmap = gmplot.GoogleMapPlotter(mismatched.start_lat.iloc[0], mismatched.start_lng.iloc[0], 7)
gmap.heatmap(mismatched.start_lat, mismatched.start_lng)
gmap.draw("gmaps/mismatched.html")

# Fix anomalies in start_lng
train_df.loc[mismatched.index, "start_lng"] = -train_df.loc[mismatched.index].start_lng

# Check coordinate anomalies in end_lng
mismatched2 = train_df[train_df.end_lng > -60]
gmap = gmplot.GoogleMapPlotter(mismatched2.end_lat.iloc[0], mismatched2.end_lng.iloc[0], 5)
gmap.heatmap(mismatched2.end_lat, mismatched2.end_lng)
gmap.draw("gmaps/mismatched2.html")

# Drop anomalies in end_lng
train_df.drop(mismatched2.index, axis=0, inplace=True)

###############################
# Step 2: Feature engineering #
###############################

## ESTIMATE DISTANCES

# Manhattan and Euclidean (Havesine) distances
def distance(df, method="manhattan"):
    earthR = 6378137.0
    pi180 = np.pi/180

    dlat = (df.end_lat - df.start_lat) * pi180
    dlng = (df.end_lng - df.start_lng) * pi180

    if method == "manhattan":
        ay = np.sin(np.abs(dlat)/2)**2
        cy = 2*np.arctan2(np.sqrt(ay), np.sqrt(1-ay))
        dy = earthR * cy

        ax = np.sin(np.abs(dlng)/2)**2
        cx = 2*np.arctan2(np.sqrt(ax), np.sqrt(1-ax))
        dx = earthR * cx

        distance = np.abs(dx) + np.abs(dy)

    elif method == "euclidean":
        a = (np.sin(dlat/2)**2 + np.cos(df.start_lat*pi180) * np.cos(df.end_lat*pi180) * np.sin(dlng/2)**2)
        c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))

        distance = earthR * c
    else:
        distance = 0

    return distance

for df in combine:
    df["manhattan"] = distance(df, method="manhattan")
    df["euclidean"] = distance(df, method="euclidean")

# Extract Google Maps data:
# The gmaps data was pre-downloaded by running the function extract_gmaps_data
# on both the training and test data. This was then imported later for usage in
# the model.

# Not used here
def extract_gmaps_data(df, test=False):
    gmaps = googlemaps.Client("<API-KEY>")

    distances = []
    durations = []

    df['pickup'] = df.start_lat.astype(str)+","+df.start_lng.astype(str)
    df['dropoff'] = df.end_lat.astype(str)+","+df.end_lng.astype(str)

    firstindex = df.index[0]
    finalindex = df.index[-1]
    interval = finalindex - firstindex

    for i in tqdm(range(interval // 9 + 1)):
        lastindex = firstindex + 8
        df_9 = df.loc[firstindex:lastindex]

        result = gmaps.distance_matrix(df_9.pickup.values,
                                       df_9.dropoff.values,
                                       units="metric")

        for i in range(len(df_9)):
            try:
                distances.append(result["rows"][i]["elements"][i]["distance"]["value"])
                durations.append(result["rows"][i]["elements"][i]["duration"]["value"])
            except:
                distances.append(0)
                durations.append(0)

        firstindex = lastindex + 1

    df['gmaps_dist'] = np.array(distances)
    df['gmaps_duration'] = np.array(durations)

    if test:
        save_gmaps_test_data(df)
    else:
        save_gmaps_train_data(df)

    return df

# Not used here
def save_gmaps_train_data(df):
    df[["gmaps_dist", "gmaps_duration"]].to_csv("gmapsdata/{0}-{1}.csv".format(df.index[0], df.index[-1]))

# Not used here
def save_gmaps_test_data(df):
    df[["gmaps_dist", "gmaps_duration"]].to_csv("gmapsdata/{0}-{1}.csv".format(df.index[0], df.index[-1]))

# Not used, data already extracted
"""
start = 0
interval = 10000

while start < len(train_df):
    print("Now starting with batch of the {0}'s...".format(start))
    train = train_df.loc[start:start+interval-1]
    train = extract_gmaps_data(train, test=False)
    start += interval

start = 0

while start < len(test_df):
    print("Now starting with batch of the {0}'s...".format(start))
    test = test_df.loc[start:start+interval-1]
    test = extract_gmaps_data(test, test=True)
    start += interval
"""

# Import gmaps data
gmaps_train_data = pd.read_csv("gmapsdata/gmaps_train_data.csv", index_col="row_id")
gmaps_test_data = pd.read_csv("gmapsdata/gmaps_test_data.csv", index_col="row_id")

train_df["gmaps_dist"] = gmaps_train_data.gmaps_dist
train_df["gmaps_duration"] = gmaps_train_data.gmaps_duration
train_df.dropna(axis=0, inplace=True)

test_df["gmaps_dist"] = gmaps_test_data.gmaps_dist
test_df["gmaps_duration"] = gmaps_test_data.gmaps_duration
test_df.dropna(axis=0, inplace=True)

# Treasure Island fix
for df in combine:
    TI_df = df[df.gmaps_dist == 0].loc[df.manhattan > 2000]
    df.loc[TI_df.index, "gmaps_dist"] = TI_df.manhattan
    df.loc[TI_df.index, "gmaps_duration"] = TI_df.manhattan / 11.0

## TIME FEATURES

# Add weekdays, hour and date columns and drop datetime
for df in combine:
    df["weekday"] = df.datetime.dt.weekday + 1
    df["hour"] = df.datetime.dt.hour
    df["date"] = df.datetime.dt.date
    df.drop("datetime", axis=1, inplace=True)

# Add marker for holidays (only 2015 since 2012 data is only one month)
holidays2015 = {
    "New Years Day": datetime.datetime(2015,1,1).date(),
    "Martin Luther King Day": datetime.datetime(2015,1,19).date(),
    "Easter Saturday": datetime.datetime(2015,4,4).date(),
    "Easter Sunday": datetime.datetime(2015,4,5).date(),
    "Memorial Sunday": datetime.datetime(2015,5,24).date(),
    "Memorial Day": datetime.datetime(2015,5,25).date(),
    "Independence Pre-day": datetime.datetime(2015,7,3).date(),
    "Independence Day": datetime.datetime(2015,7,4).date(),
    "Independence Post-day": datetime.datetime(2015,7,5).date(),
    "Labor Day": datetime.datetime(2015,9,7).date(),
    "Thanksgiving Day": datetime.datetime(2015,11,26).date(),
    "Thanksgiving Post-day": datetime.datetime(2015,11,27).date(),
    "Thanksgiving Post-post-day": datetime.datetime(2015,11,28).date(),
    "Christmas Eve": datetime.datetime(2015,12,24).date(),
    "Christmas Day": datetime.datetime(2015,12,25).date(),
    "Christmas Post-day": datetime.datetime(2015,12,26).date(),
    "New Years Eve": datetime.datetime(2015,12,31).date(),
}

for df in combine:
    df["holiday"] = np.zeros(df.index.shape)

    for d in holidays2015.values():
        df.loc[df.date == d, ['holiday']] = 1

## GEOLOCATIONS

# Clustering hotspots
def geolocations(combine):
    train_df = combine[0]
    test_df = combine[1]

    # Collate coordinates
    train_df['train_or_test'] = "train"
    test_df['train_or_test'] = "test"
    train_test = pd.concat([train_df,test_df]).reset_index()

    start = pd.DataFrame(train_test[["row_id", "start_lat", "start_lng", "train_or_test"]].values, columns=["row_id", "lat", "lng", "train_or_test"])
    start['start_or_end'] = "start"
    end = pd.DataFrame(train_test[["row_id", "end_lat", "end_lng", "train_or_test"]].values, columns=["row_id", "lat", "lng", "train_or_test"])
    end['start_or_end'] = "end"
    coords = pd.concat([start,end]).reset_index(drop=True)

    # Find clusters
    DB = DBSCAN(eps=0.005, min_samples=2500).fit(coords[["lng", "lat"]])
    print("Number of clusters found:", max(DB.labels_) + 1)

    # Visualize clusters
    for i in range(max(DB.labels_) + 1):
        cluster = coords.loc[np.argwhere(DB.labels_ == i).flatten()]
        gmap = gmplot.GoogleMapPlotter(cluster.lat.sample(1).values[0], cluster.lng.sample(1).values[0], 12)
        gmap.heatmap(cluster.lat, cluster.lng)
        gmap.draw("gmaps/cluster{0}.html".format(i))

    # As seen, the following clusters are:
    # -1 -> not clustered (standalone)
    # 0  -> SF Market Street (city)
    # 1  -> NY Manhattan (city)
    # 2  -> SFO Airport (airport)
    # 3  -> North SF (city)
    # 4  -> SF Mission District (city)
    # 5  -> JFK Airport (airport)
    # 6  -> LaGuardia Airport (airport)

    # Label clusters
    clusters = []
    for i in DB.labels_:
        if i == -1:
            clusters.append("standalone")
        elif i in [0, 1, 3, 4]:
            clusters.append("city")
        elif i in [2, 5, 6]:
            clusters.append("airport")

    # Visualize distribution of trips per cluster
    location = np.array(clusters)
    pd.value_counts(location).plot.bar()
    plt.show()

    # Add flags to columns
    coords['location'] = location

    train_locs = coords.loc[coords.train_or_test == "train"]
    test_locs = coords.loc[coords.train_or_test == "test"]

    train_start_locs = train_locs.loc[train_locs.start_or_end == "start"].loc[:,["row_id","location"]].set_index("row_id")
    train_end_locs = train_locs.loc[train_locs.start_or_end == "end"].loc[:,["row_id","location"]].set_index("row_id")
    test_start_locs = test_locs.loc[test_locs.start_or_end == "start"].loc[:,["row_id","location"]].set_index("row_id")
    test_end_locs = test_locs.loc[test_locs.start_or_end == "end"].loc[:,["row_id","location"]].set_index("row_id")

    train_df["start_loc"] = train_start_locs
    train_df["end_loc"] = train_end_locs
    test_df["start_loc"] = test_start_locs
    test_df["end_loc"] = test_end_locs

    train_df.drop("train_or_test", axis=1, inplace=True)
    test_df.drop("train_or_test", axis=1, inplace=True)

    combine = [train_df, test_df]

    for df in combine:
        df["airport"] = np.zeros(df.index.shape)
        df["citycenter"] = np.zeros(df.index.shape)
        df["standalone"] = np.zeros(df.index.shape)

        df.loc[df.start_loc == "airport", "airport"] = 1
        df.loc[df.end_loc == "airport", "airport"] = 1

        df.loc[df.start_loc == "city", "citycenter"] = 1
        df.loc[df.end_loc == "city", "citycenter"] = 1

        df.loc[df.start_loc == "standalone", "standalone"] = 1
        df.loc[df.end_loc == "standalone", "standalone"] = 1

        df.drop("start_loc", axis=1, inplace=True)
        df.drop("end_loc", axis=1, inplace=True)

    return combine

combine = geolocations(combine)

## PRECIPITATION DATA

# Import precipitation data
prec = pd.read_csv("raindata/precipitation.csv")
prec.date = pd.to_datetime(prec.date, dayfirst=True)

# Add precipitation data to dataframes
for df in combine:
    df["precipitation"] = np.zeros(df.start_lng.shape)

    for d in prec.date.dt.date:
        df.loc[df.date == d, 'precipitation'] = prec.loc[prec.date.dt.date == d, 'precipitation'].values

    df.drop("date", axis=1, inplace=True)

## MARK OUTLIERS

# Mark routing errors and short trips
for df in combine:
    df["routing_error"] = np.zeros(df.index.shape)
    df["short_trip"] = np.zeros(df.index.shape)

    df.loc[(df.gmaps_dist > 500) & (df.manhattan < 50), "routing_error"] = 1
    df.loc[(df.gmaps_dist < 500) & (df.manhattan < 50), "short_trip"] = 1

## Final dataframes

# Show correlation table
def correlation_table(train_df):
    colormap = plt.cm.viridis
    plt.figure(figsize=(10,10))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sb.heatmap(train_df.corr().round(2)\
                ,linewidths=0.1,vmax=1.0, square=True, cmap=colormap, \
                linecolor='white', annot=True);

correlation_table(train_df)

############################
# Step 3: Prediction Model #
############################

## DEFINE FUNCTIONS

# Plot feature importances
def plot_model_var_imp(model, X):
    imp = pd.DataFrame(model.feature_importances_,
                       columns = ['Importance'],
                       index = X.columns).sort_values(['Importance'], ascending=True)
    imp.plot(kind = 'barh')
    plt.show()

# Normalize data - not used for XGBoost and Random Forest
def normalizer(X):

    features = []

    coords = X[["start_lng", "start_lat", "end_lng", "end_lat"]]
    coordsnorm = pd.DataFrame(MinMaxScaler(feature_range=(-1,1)).fit_transform(coords), index=coords.index, columns=coords.columns)
    features.append(coordsnorm)

    dist = X[["manhattan", "euclidean", "gmaps_dist", "gmaps_duration"]]
    distnorm = pd.DataFrame(MinMaxScaler(feature_range=(0,10)).fit_transform(dist), index=dist.index, columns=dist.columns)
    features.append(distnorm)

    precipitation = X[["precipitation"]]
    precnorm = pd.DataFrame(MinMaxScaler(feature_range=(0,1)).fit_transform(precipitation), index=precipitation.index, columns=precipitation.columns)
    features.append(precnorm)

    times = X[["weekday", "hour"]]
    timesnorm = pd.DataFrame(MinMaxScaler(feature_range=(0,5)).fit_transform(times), index=times.index, columns=times.columns)
    features.append(timesnorm)

    flags = X[["holiday", "airport", "citycenter", "standalone", "routing_error", "short_trip"]]
    features.append(flags)

    Xnorm = pd.concat(features, axis=1)

    return Xnorm

# Fit model (only XGBoost here)
def regression(train_df, LINREG=False, RIDGE=False, LASSO=False, SVMR=False, XGB=True, RF=False, NN=False):
    X = train_df.drop("duration", axis=1)
    Y = train_df.duration
    Xn = normalizer(X)

    np.random.seed(1337) # set random seed for reproducibility
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)
    Xn_train, Xn_val, Yn_train, Yn_val = train_test_split(Xn, Y, test_size=0.2)

    # Linear Regression
    if LINREG:
        st = time.time()
        lin_reg = LinearRegression(normalize=True)
        lin_reg.fit(Xn_train, Yn_train)
        pred_lr = lin_reg.predict(Xn_val)

        print('Linear Regression RMSE with {0} training data points:'.format(len(Xn_train)),
              str(np.sqrt(mean_squared_error(pred_lr, Yn_val))))
        print('Linear Regression CPU Time:', time.time() - st, 'seconds\n')
    else:
        lin_reg = None

    # Ridge Regression
    if RIDGE:
        st = time.time()
        ridge = Ridge(alpha=0.5)
        ridge.fit(Xn_train, Yn_train)
        pred_ridge = ridge.predict(Xn_val)

        print('Ridge Regression RMSE with {0} training data points:'.format(len(Xn_train)),
              str(np.sqrt(mean_squared_error(pred_ridge, Yn_val))))
        print('Ridge Regression CPU Time:', time.time() - st, 'seconds\n')
    else:
        ridge = None

    # Lasso Regression
    if LASSO:
        st = time.time()
        lasso = Lasso(alpha=0.1, max_iter=5000)
        lasso.fit(Xn_train, Yn_train)
        pred_lasso = lasso.predict(Xn_val)

        print('Lasso Regression RMSE with {0} training data points:'.format(len(Xn_train)),
              str(np.sqrt(mean_squared_error(pred_lasso, Yn_val))))
        print('Lasso Regression CPU Time:', time.time() - st, 'seconds\n')
    else:
        lasso = None

    # Support Vector Machines Regressor (non-linear kernel)
    if SVMR:
        st = time.time()
        svr = SVR()
        svr.fit(X_train, Y_train)
        pred_svr = svr.predict(X_val)

        print('Support Vector Regressor RMSE with {0} training data points:'.format(len(X_train)),
              str(np.sqrt(mean_squared_error(pred_svr, Y_val))))
        print('Support Vector Regressor CPU Time:', time.time() - st, 'seconds\n')
    else:
        svr = None

    # XGBoost
    if XGB:
        st = time.time()
        xgb = XGBRegressor(max_depth=9, learning_rate=0.045, n_estimators=500, reg_lambda=0.5)
        xgb.fit(X_train, Y_train)
        pred_xgb = xgb.predict(X_val)

        print('XGBoost RMSE with {0} training data points:'.format(len(X_train)),
              str(np.sqrt(mean_squared_error(pred_xgb, Y_val))))
        print('XGBoost CPU Time:', time.time() - st, 'seconds\n')
        plot_model_var_imp(xgb, X_train)
    else:
        xgb = None

    # Random Forest
    if RF:
        st = time.time()
        random_forest = RandomForestRegressor(n_estimators=500)
        random_forest.fit(X_train, Y_train)
        pred_rf = random_forest.predict(X_val)

        print('Random Forest RMSE with {0} training data points:'.format(len(X_train)),
              str(np.sqrt(mean_squared_error(pred_rf, Y_val))))
        print('Random Forest CPU Time:', time.time() - st, 'seconds\n')
        plot_model_var_imp(random_forest, X_train)
    else:
        random_forest = None

    # Neural Network
    if NN:
        st = time.time()
        NN_model = Sequential()
        NN_model.add(Dense(20, kernel_initializer='normal', input_dim=Xn_train.shape[1], activation='relu'))
        NN_model.add(Dense(100, activation='relu', activity_regularizer=l2(0.2)))
        NN_model.add(Dense(250, activation='relu', activity_regularizer=l2(0.2)))
        NN_model.add(Dense(50, activation='relu', activity_regularizer=l2(0.2)))
        NN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))
        NN_model.compile(loss='mse', optimizer='adam')
        history = NN_model.fit(Xn_train, Yn_train, epochs=500, batch_size=20, verbose=2, validation_split=0.2)

        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

        pred_nn = NN_model.predict(Xn_val)
        print('Neural Network RMSE with {0} training data points:'.format(len(Xn_train)),
              str(np.sqrt(mean_squared_error(pred_nn, Yn_val))))
        print('Neural Network CPU Time:', time.time() - st, 'seconds\n')
    else:
        NN_model = None

    models = {"Linear Regression": lin_reg,
              "Ridge Regression": ridge,
              "Lasso Regression": lasso,
              "SVR": svr,
              "Random Forest": random_forest,
              "XGBoost": xgb,
              "Neural Network": NN_model}

    return models

## FIT MODEL

models = regression(train_df)

# Predict durations
xgb = models["XGBoost"]
test_pred = xgb.predict(test_df)

## COMPARE PREDICTION WITH TRAINING DATA

# Comparing histograms of predictions
def compare_predictions(pred_1, pred_2):
    bins = np.histogram(np.hstack((pred_1, pred_2)), bins=100)[1] #get the bin edges
    plt.hist(pred_1, bins=bins, alpha=1)
    plt.hist(pred_2, bins=bins, alpha=0.7)
    plt.title("Prediction 1 vs Prediction 2")
    plt.xlabel("Duration [s]")
    plt.ylabel("Number of instances")
    plt.legend(["Prediction 1", "Prediction 2"])
    plt.show()

compare_predictions(train_df.sample(len(test_df)).duration.values, test_pred)

###############################
# Step 4: Export as .csv-file #
###############################

def to_submission(prediction):
    date_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_string = "output/test_prediction_" + date_string + ".csv"

    try:
        df = pd.DataFrame(prediction, columns=["duration"])
    except:
        df = pd.DataFrame(prediction.flatten(), columns=["duration"])

    df.index.name = "row_id"
    df.to_csv(file_string)

to_submission(test_pred)

######################################
# APPENDIX: XGBoost parameter tuning #
######################################

# This section shows how hyperparameter tuning would work for the XGBoost
# algorithm, searching for the optimal value of max_depth and learning_rate.

# Prepare training data
X = train_df.drop("duration", axis=1)
Y = train_df.duration
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)

# XGBoost hyperparameter tuning
max_depths = [7, 8, 9, 10, 11]
learning_rates = [0.04, 0.042, 0.044, 0.046, 0.048, 0.05]
optimum = np.ones((3,3)) * float('inf')

for max_depth in tqdm(max_depths):
    for learning_rate in tqdm(learning_rates):
        xgb = XGBRegressor(max_depth=max_depth, learning_rate=learning_rate, n_estimators=500, reg_lambda=0.5)
        print("Now training:\tmax_depth = {0}\tlearning_rate = {1}".format(max_depth, learning_rate))
        xgb.fit(X_train, Y_train)
        pred_xgb = xgb.predict(X_val)
        error = np.sqrt(mean_squared_error(pred_xgb, Y_val))
        print("RMSE: {0}".format(error), "\n")
        if error < optimum[0,0]:
            optimum[2,:], optimum[1,:] = optimum[1,:], optimum[0,:]
            optimum[0,:] = np.array([error, max_depth, learning_rate])
        elif error < optimum[1,0]:
            optimum[2,:] = optimum[1,:]
            optimum[1,:] = np.array([error, max_depth, learning_rate])
        elif error < optimum[2,0]:
            optimum[2,:] = np.array([error, max_depth, learning_rate])

# Display optimal hyperparameters and fit into model
print('Optimal hyperparameters:', optimum)

xgb = XGBRegressor(max_depth=int(optimum[0][1]), learning_rate=optimum[0][2], n_estimators=500, reg_lambda=0.5)
xgb.fit(X_train, Y_train)
pred_xgb = xgb.predict(X_val)

print('Optimal XGBoost RMSE with {0} training data points:'.format(len(X_train)),
      str(np.sqrt(mean_squared_error(pred_xgb, Y_val))))

plot_model_var_imp(xgb, X_train)
