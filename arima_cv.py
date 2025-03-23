import numpy as np
import pandas as pd
import os
from pmdarima import auto_arima
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# Define the file path (adjust if needed)
path = os.path.join(os.getcwd(), "data/processed_data")
file_path = os.path.join(path, "train.csv")

# Load the CSV data
df = pd.read_csv(file_path)

# Inspect the first few entries of "year_week" to see its format
print("Original 'year_week' values:")
print(df['year_week'].head())

# Function to clean the "year_week" string.
def clean_year_week(s):
    parts = s.split('/')
    date_part = parts[-1]
    if date_part.endswith('-1'):
        date_part = date_part[:-2]
    return date_part

# Create a new column with the cleaned date string and convert to datetime.
df['week_start'] = pd.to_datetime(df['year_week'].apply(clean_year_week), format='%Y-%m-%d')

# Verify the conversion
print("\nAfter cleaning and converting to datetime:")
print(df[['year_week', 'week_start']].head())

# Aggregate sales per week and category using "total_price" as the sales measure.
weekly_sales = df.groupby(['week_start', 'category'])['total_price'].sum().reset_index()
print("\nWeekly aggregated sales data:")
print(weekly_sales.head())

# Dictionary to store forecast results for each category.
forecast_results = {}
# Dictionary to store cross-validation error metrics for each category.
cv_metrics = {}

# Number of splits for time series cross-validation
n_splits = 5

# Forecast for each category using auto_arima and evaluate using time series cross-validation.
for cat in weekly_sales['category'].unique():
    # Filter data for the current category and set the datetime index.
    cat_data = weekly_sales[weekly_sales['category'] == cat].set_index('week_start').sort_index()
    
    # Ensure the time series has a regular weekly frequency.
    cat_data = cat_data.asfreq('W')
    
    print(f"\nTime series for category '{cat}':")
    print(cat_data.head())
    
    # If there is not enough data, skip this category.
    if cat_data['total_price'].dropna().empty or len(cat_data) < n_splits + 1:
        print(f"Not enough valid data for category '{cat}'. Skipping forecast.")
        forecast_results[cat] = None
        cv_metrics[cat] = None
        continue

    # Time Series Cross-Validation using TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_errors = []
    for train_index, test_index in tscv.split(cat_data):
        train_data = cat_data.iloc[train_index]['total_price'].dropna()
        test_data = cat_data.iloc[test_index]['total_price'].dropna()
        if len(train_data) == 0 or len(test_data) == 0:
            continue
        try:
            # Fit auto_arima on the training fold.
            model_cv = auto_arima(
                train_data,
                seasonal=False,
                trace=False,
                error_action='ignore',
                suppress_warnings=True
            )
            # Forecast for the length of the validation set.
            forecast_cv = model_cv.predict(n_periods=len(test_data))
            error = mean_absolute_error(test_data, forecast_cv)
            fold_errors.append(error)
        except Exception as ex:
            print(f"Cross-validation failed for category '{cat}': {ex}")
    
    if fold_errors:
        avg_error = np.mean(fold_errors)
        cv_metrics[cat] = {'fold_errors': fold_errors, 'avg_error': avg_error}
        print(f"Average CV MAE for category '{cat}': {avg_error:.2f}")
    else:
        cv_metrics[cat] = None
        print(f"No valid CV errors computed for category '{cat}'.")
    
    # Fit auto_arima on the full available data for the category.
    try:
        model_auto = auto_arima(
            cat_data['total_price'].dropna(),
            seasonal=False,
            trace=True,
            error_action='ignore',
            suppress_warnings=True
        )
        print(model_auto.summary())
        
        # Forecast 20 weeks ahead.
        forecast_values = model_auto.predict(n_periods=20)
        
        # Create a datetime index for the forecast:
        last_date = cat_data.index[-1]
        forecast_index = pd.date_range(
            start=last_date + pd.offsets.Week(1),
            periods=20,
            freq='W'
        )
        
        forecast_series = pd.Series(forecast_values, index=forecast_index)
        forecast_results[cat] = forecast_series
        
    except Exception as e:
        forecast_results[cat] = None
        print(f"auto_arima model failed for category '{cat}': {e}")

# Construct a DataFrame to store the predictions.
rows = []
for cat, forecast_series in forecast_results.items():
    if forecast_series is not None:
        for forecast_date, forecast_value in forecast_series.items():
            week_str = forecast_date.strftime('%Y-%m-%d')
            row = {
                'year_week': None,  # Not needed, will be removed
                'category': cat,
                'num_items_sold': np.nan,
                'total_price': forecast_value,
                'total_freight_value': np.nan,
                'most_sold_state': np.nan,
                'most_bought_state': np.nan,
                'avg_review_score': np.nan,
                'week': week_str
            }
            rows.append(row)

forecast_df = pd.DataFrame(rows)

# Remove unwanted columns and reorder so that 'week' is the first column.
final_df = forecast_df[['week', 'category', 'total_price']]

# Define the output CSV file path.
output_path = os.path.join(path, "arima_prediction_cv.csv")

# Save the final DataFrame to a CSV file.
final_df.to_csv(output_path, index=False)

print("\nPrediction CSV file has been created at:", output_path)
print("\nCross-validation metrics by category:")
print(cv_metrics)  # MAE per fold and the average
