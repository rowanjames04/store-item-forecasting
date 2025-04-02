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

# Feature engineering
df['month'] = df.date.dt.month
df['day_of_month'] = df.date.dt.day
df['day_of_year'] = df.date.dt.dayofyear 
df['week_of_year'] = df.date.dt.isocalendar().week
df['day_of_week'] = df.date.dt.dayofweek
df['year'] = df.date.dt.year
df["is_wknd"] = df.date.dt.weekday // 4
df['is_month_start'] = df.date.dt.is_month_start.astype(int)
df['is_month_end'] = df.date.dt.is_month_end.astype(int)

def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))

df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)

def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

lags = [91, 98, 105, 112, 119, 126, 182, 364, 546, 728]
df = lag_features(df, lags)

def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe

df = roll_mean_features(df, [365, 546, 730])

def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.99, 0.95, 0.9, 0.8, 0.7, 0.5]
df = ewm_features(df, alphas, lags)

df = pd.get_dummies(df, columns=['day_of_week', 'month'])
df['sales'] = np.log1p(df["sales"].values)

# Train/val split
train = df.loc[(df["date"] < "2017-01-01"), :]
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]

cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]
Y_train = train['sales']
X_train = train[cols]
Y_val = val['sales']
X_val = val[cols]

def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

# Unified LightGBM parameters
lgb_params = {
    'metric': 'mae',
    'num_leaves': 10,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'max_depth': 5,
    'verbose': -1,
    'nthread': -1,
    'early_stopping_rounds': 200
}

# Model training
lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)


model = lgb.train(
    lgb_params,
    lgbtrain,
    valid_sets=[lgbtrain, lgbval],
    num_boost_round=10000,
    feval=lgbm_smape,
    callbacks=[lgb.log_evaluation(100)]
)

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
smape(np.expm1(y_pred_val), np.expm1(Y_val))

# Final model
train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]

test = df.loc[df.sales.isna()]
X_test = test[cols]

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

final_params = lgb_params.copy()
final_params.pop('early_stopping_rounds', None)

final_model = lgb.train(
    final_params,
    lgbtrain_all,
    num_boost_round=model.best_iteration,
    callbacks=[lgb.log_evaluation(100)]
)

test_preds = final_model.predict(X_test, num_iteration=final_model.best_iteration)

forecast = pd.DataFrame({
    "date": test["date"],
    "store": test["store"],
    "item": test["item"],
    "sales": test_preds
})
# Plotting - Save to files instead of showing

# 1. Store 1 Item 1 Forecast (saved to file)
plt.figure(figsize=(20, 6))
forecast[(forecast.store == 1) & (forecast.item == 1)] \
    .set_index("date")["sales"] \
    .plot(color="green", legend=True, label="Store 1 Item 1 Forecast")
plt.title("Store 1 Item 1 - Sales Forecast")
plt.ylabel("Log-Scaled Sales")
plt.grid(True)
plt.savefig('store1_item1_forecast.png', dpi=300, bbox_inches='tight')
plt.close()  # Important: closes the figure to free memory

# 2. Store 1 Item 17 Comparison (saved to file)
plt.figure(figsize=(20, 6))
train[(train.store == 1) & (train.item == 17)] \
    .set_index("date")["sales"] \
    .plot(legend=True, label="Actual Sales")
forecast[(forecast.store == 1) & (forecast.item == 17)] \
    .set_index("date")["sales"] \
    .plot(legend=True, label="Forecasted Sales")
plt.title("Store 1 Item 17 - Actual vs Forecast")
plt.ylabel("Log-Scaled Sales")
plt.xlabel("Date")
plt.grid(True)
plt.legend()
plt.savefig('store1_item17_comparison.png', dpi=300, bbox_inches='tight')
plt.close()  # Important: closes the figure

# Assuming your code has already run and produced the 'forecast' DataFrame
# which contains 'date', 'store', 'item', and 'sales' (log-scaled predictions)

# 1. Convert predictions back from log scale
test_preds_exp = np.expm1(test_preds)

# 2. Create output DataFrame with 'id' as int right away
output = test[['id']].copy().astype(int)  # Cleaner one-step conversion
output['sales'] = test_preds_exp.clip(lower=0)  # Combine clip with assignment

# 3. Sort & save
output = output.sort_values('id')
output_path = os.path.join(script_dir, 'output.csv')  # Changed to output.csv
output.to_csv(output_path, index=False)

print(f"Output file saved to: {output_path}")
print(output.head())

print("Plots saved as:")
print("- store1_item1_forecast.png")
print("- store1_item17_comparison.png")
print(df.shape)