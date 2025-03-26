# Required libraries and settings
import os
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import itertools

import warnings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

# Load dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(script_dir, 'demand-forecasting-kernels-only', 'train.csv')
test_path = os.path.join(script_dir, 'demand-forecasting-kernels-only', 'test.csv')

train = pd.read_csv(train_path, parse_dates=['date'])
test = pd.read_csv(test_path, parse_dates=['date'])

# Combine train and test for data preprocessing
df = pd.concat([train, test], sort=False) 

# Extracting various time-related features from the 'date' column to help in analyzing patterns over time.
df['month'] = df.date.dt.month
df['day_of_month'] = df.date.dt.day
df['day_of_year'] = df.date.dt.dayofyear 
df['week_of_year'] = df.date.dt.isocalendar().week
df['day_of_week'] = df.date.dt.dayofweek
df['year'] = df.date.dt.year
df["is_wknd"] = df.date.dt.weekday // 4
df['is_month_start'] = df.date.dt.is_month_start.astype(int)
df['is_month_end'] = df.date.dt.is_month_end.astype(int)

# Generates random noise with a normal distribution (mean=0, std=1.6)
# The noise is the same length as the input dataframe and can be used to add variability to data.
# Noise refers to random variations or disturbances in data that are not part of the true signal.
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))

# Sorts the DataFrame by 'store', 'item', and 'date' in ascending order to ensure proper order for time-series analysis or grouped operations.
df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)

# This function creates lag features for the sales data by shifting the sales values and adding random noise.
def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

# 91 represents the 3 months point
# 98 is roughly 3 months and a week
# 105 is 3 months and 2 weeks
# 112 is 4 months
# 119 is 4 months and 1 week
# 126 is 4 months and 2 weeks
# 182 is 6 months
# 364 is 1 year
# 546 is 1.5 years
# 728 is 2 years
lags = [91, 98, 105, 112, 119, 126, 182, 364, 546, 728]
df = lag_features(df, lags)

# This function creates rolling mean features for the sales data by calculating the rolling mean over specified 
# window sizes and adding random noise for each store-item combination.
def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe

# 365 represents a 1-year rolling mean
# 546 represents a 1.5-year rolling mean
# 730 represents a 2-year rolling mean
df = roll_mean_features(df, [365, 546, 730])

# This function creates exponentially weighted moving average (EWM) features for the sales data by calculating 
# the EWM over specified alpha and lag values for each store-item combination.
def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

# 0.99 represents a very high weight on recent data, making it more sensitive to recent observations.
# 0.95 represents a slightly lower weight on recent data, allowing for a more balanced smoothing.
# 0.9 gives even less weight to recent observations, with a more smoothed effect over time.
# 0.8 further reduces the influence of recent data, emphasizing longer-term trends.
# 0.7 continues to decrease the weight on recent observations, giving more focus to the past data.
# 0.5 gives an equal weight to past and recent observations, providing a simple moving average-like behavior.
alphas = [0.99, 0.95, 0.9, 0.8, 0.7, 0.5]
df = ewm_features(df, alphas, lags)

# One-Hot Encoding - used to convert categorical variables (day_of_week, month) into binary columns for machine learning models
df = pd.get_dummies(df, columns=['day_of_week', 'month'])

# Log Transformation - used to compress skewed sales data and improve model performance by normalizing the distribution
df['sales'] = np.log1p(df["sales"].values)

# Train set until the beginning of 2017 (until the end of 2016).
train = df.loc[(df["date"] < "2017-01-01"), :]

# First 3 months of 2017 as the validation set.
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]

# Independent variables
cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

# Selecting the dependent variable for the train set
Y_train = train['sales']

# Selecting the independent variables for the train set
X_train = train[cols]

# Selecting the dependent variable for the validation set
Y_val = val['sales']

# Selecting the independent variables for the validation set
X_val = val[cols]

# SMAPE (Symmetric Mean Absolute Percentage Error) function to calculate the forecast accuracy
# It computes the symmetric mean absolute percentage error between predicted values (preds) and actual values (target)
# It handles zero values in the dataset to avoid division by zero errors.
def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

# Custom SMAPE evaluation function for LightGBM model.
# It calculates SMAPE between predicted values (preds) and actual values (labels) in LightGBM.
# The predictions and labels are transformed using np.expm1() to inverse log transformation before computing SMAPE.
# Returns the name of the metric ('SMAPE'), the calculated SMAPE value, and a flag (False) indicating no special behavior.
def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

# LightGBM parameters
lgb_params = {'metric': {'mae'}, # Mean Absolute Error
              'num_leaves': 10, # Number of leaves in one tree, controls complexity of the tree
              'learning_rate': 0.02, # The rate at which the model updates
              'feature_fraction': 0.8, # Fraction of features used for each tree
              'max_depth': 5,
              'verbose': 0, 
              'num_boost_round': 10000, 
              'early_stopping_rounds': 200, # Stops training if validation score doesn't improve after 200 rounds
              'nthread': -1}

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape, # observing the error
                  verbose_eval=100)

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

# percentage of validation error
smape(np.expm1(y_pred_val), np.expm1(Y_val))

