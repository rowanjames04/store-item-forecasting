# Demand Forecasting with LightGBM

A machine learning project that predicts future product sales using historical data and advanced feature engineering techniques. Built with Python and LightGBM, it processes time series data to make accurate forecasts and evaluates them using the SMAPE metric (a percentage-based error measure that handles scale and zero values better than traditional metrics).

## Features

- Predicts future sales for individual items across multiple stores
- Advanced time-based feature engineering:
  - Day of week, month, year, weekend flags
  - Lag features (e.g., sales from 7, 14, 28 days ago)
  - Rolling and exponentially weighted averages
- One-hot encoding for date-related categorical features
- Log-transform on sales data to stabilize variance
- Custom SMAPE metric for better error interpretation on relative scales
- Visualizations of:
  - Forecasted sales for a selected item
  - Actual vs predicted sales comparison
- Generates a CSV file with predicitive results

## Technologies Used

- Python 3
- Pandas & NumPy for data processing
- LightGBM (a fast gradient boosting library)
- Matplotlib for plotting
- scikit-learn for splitting datasets and metrics

## Installation

1. Clone the repository  
   `git clone https://github.com/rowanjames04/store-item-forecasting.git`

2. Navigate to the project directory  
   `cd store-item-forecasting`

3. Install dependencies  
   `pip install pandas numpy matplotlib lightgbm scikit-learn`

## Usage

Run the script:  
`store-item-forecasting.py`

Outputs:
- `output.csv` — Sales predictions formatted for Kaggle submission
- `store1_item1_forecast.png` — Forecasted future sales for item 1 in store 1
- `store1_item17_comparison.png` — Comparison of actual vs predicted sales for item 17 in store 1

## File Structure

- `forecast.py` — Main script with data loading, feature engineering, training, and forecasting
- `demand-forecasting-kernels-only/train.csv` — Historical sales data
- `demand-forecasting-kernels-only/test.csv` — Future data for prediction
- `output.csv` — Final predictions in submission format
- `store1_item1_forecast.png` — Plot of forecasted sales
- `store1_item17_comparison.png` — Plot comparing real and predicted values

## Notes

- LightGBM is used for its speed and accuracy in handling large tabular data.
- SMAPE (Symmetric Mean Absolute Percentage Error) is used to evaluate performance fairly across varying sales scales.
- Forecasts are log-transformed during modeling and converted back to normal scale for output.

## License

This project is open-source under the MIT License.

## Contact

For issues or questions, please open a GitHub issue or contact me at your.email@example.com
