import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools  # For cycling through colors
import os

warnings.filterwarnings("ignore")

# 1. Load dataset and prepare data
path = os.getcwd() + "/data/processed_data"
file_path = os.path.join(path, "train_data.csv")
df = pd.read_csv(file_path)

print("Dataset preview:")
print(df.head())
print("Columns:", df.columns.tolist())

# Convert the 'week' column: extract the start date from the week range and convert it to datetime
df['week'] = pd.to_datetime(df['week'].str.split('/').str[0])

# 2. Identify only the price columns (columns that end with '_price')
price_cols = [col for col in df.columns if col.endswith('_price')]
print("Identified price columns:", price_cols)

results = {}      # Will store results for each price column
error_metrics = {}  # Store error metrics

# Process each price column with additional exogenous regressors
for col in price_cols:
    # Derive prefix to identify matching exogenous columns
    prefix = col.replace('_price', '')
    exog_suffixes = ['_items', '_freight', '_review', '_shipping']
    exog_cols = [prefix + suffix for suffix in exog_suffixes if prefix + suffix in df.columns]
    print(f"For {col}, using exogenous columns: {exog_cols}")
    
    # Sort data by week
    col_df = df.sort_values('week').reset_index(drop=True)
    ts = col_df[col]
    
    # If exogenous columns exist, extract them
    if exog_cols:
        exog_all = col_df[exog_cols]
    else:
        exog_all = None
    
    # Split the data: training = first 80%, test = last 20%
    n = len(ts)
    split_point = int(n * 0.8)
    train = ts.iloc[:split_point]
    test  = ts.iloc[split_point:]
    weeks_test = col_df['week'].iloc[split_point:]
    
    if exog_all is not None:
        exog_train = exog_all.iloc[:split_point]
        exog_test  = exog_all.iloc[split_point:]
    else:
        exog_train = None
        exog_test  = None
    
    # Use auto_arima with exogenous regressors if available; otherwise fallback.
    try:
        from pmdarima import auto_arima
        auto_model = auto_arima(
            train,
            exogenous=exog_train,
            seasonal=True,
            m=12,  # Adjust seasonal period if needed
            error_action='ignore',
            suppress_warnings=True
        )
        order = auto_model.order
        seasonal_order = auto_model.seasonal_order
    except Exception as e:
        print(f"auto_arima not available or failed for column '{col}': {e}")
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 12)
    
    # Fit SARIMAX model with exogenous regressors
    model = SARIMAX(
        train,
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_fit = model.fit(disp=False)
    
    # Forecast on the test period (using corresponding exogenous values)
    forecast = model_fit.forecast(steps=len(test), exog=exog_test)
    
    # Calculate error metrics
    mse = mean_squared_error(test, forecast)
    mae = mean_absolute_error(test, forecast)
    error_metrics[col] = {'mse': mse, 'mae': mae}
    
    # Save results for later use (including test weeks, actual and forecast values)
    results[col] = {
        'test': test,
        'forecast': forecast,
        'weeks_test': weeks_test,
        'order': order,
        'seasonal_order': seasonal_order,
        'exog_train': exog_train,
        'exog_test': exog_test
    }

# 3. Write all predictions to a CSV file
all_predictions = []
for col in price_cols:
    pred_df = pd.DataFrame({
        'week': results[col]['weeks_test'].values,
        'actual': results[col]['test'].values,
        'forecast': results[col]['forecast'].values,
    })
    pred_df['price_category'] = col
    all_predictions.append(pred_df)
    
predictions_df = pd.concat(all_predictions, ignore_index=True)

# Save to a new folder "prediction_data" in your current working directory
path2 = os.getcwd() + "/data/prediction_data"
file_path2 = os.path.join(path2, "sarima_prediction.csv")
predictions_df.to_csv(file_path2, index=False)
print(f"Predictions have been saved to {file_path2}")

# 4. Choose plot option: "combined" or "individual"
plot_option = "individual"  # individual takes a long time

if plot_option == "combined":
    # Combined Plot: all categories on one graph with each category having a unique color
    # Create a color cycle, e.g., tab20 for up to 20 distinct colors
    color_cycle = itertools.cycle(plt.cm.tab20(np.linspace(0, 1, 20)))
    # Assign a unique color to each price column
    color_map = {}
    for col in price_cols:
        color_map[col] = next(color_cycle)
    
    plt.figure(figsize=(14, 8))
    for col in price_cols:
        weeks = results[col]['weeks_test']
        actual = results[col]['test']
        forecast = results[col]['forecast']
        
        # Use the same color for actual & forecast
        c = color_map[col]
        plt.plot(weeks, actual, color=c, label=f'Actual {col}')
        plt.plot(weeks, forecast, color=c, linestyle='--', label=f'Forecast {col}')
    
    plt.xlabel('Week')
    plt.ylabel('Price')
    plt.title('Actual vs. Forecast Prices by Category (Test Period)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

elif plot_option == "individual":
    # Individual Plots: one graph per category
    for col in price_cols:
        plt.figure(figsize=(10, 5))
        weeks = results[col]['weeks_test']
        actual = results[col]['test']
        forecast = results[col]['forecast']
        
        plt.plot(weeks, actual, label=f'Actual {col}', color='darkorange')
        plt.plot(weeks, forecast, label=f'Forecast {col}', color='orange', linestyle='--')
        plt.xlabel('Week')
        plt.ylabel('Price')
        plt.title(f'Actual vs. Forecast for {col}\nSARIMA order: {results[col]["order"]}, seasonal_order: {results[col]["seasonal_order"]}')
        plt.legend()
        plt.tight_layout()
        plt.show()
else:
    print("Invalid plot option. Choose either 'combined' or 'individual'.")
