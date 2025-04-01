import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import itertools  # For cycling through colors
import os

warnings.filterwarnings("ignore")

# 1. Load datasets: training and testing
train_path = os.path.join(os.getcwd() + "/data/processed_data", "train_data.csv")
test_path  = os.path.join(os.getcwd() + "/data/processed_data", "test_data.csv")

df_train = pd.read_csv(train_path)
df_test  = pd.read_csv(test_path)

print("Train data preview:")
print(df_train.head())
print("Test data preview:")
print(df_test.head())

# Convert the 'week' column in both datasets:
df_train['week'] = pd.to_datetime(df_train['week'].str.split('/').str[0])
df_test['week']  = pd.to_datetime(df_test['week'].str.split('/').str[0])

# 2. Identify only the price columns (columns that end with '_price') from the training dataset
price_cols = [col for col in df_train.columns if col.endswith('_price')]
print("Identified price columns:", price_cols)

results = {}       # Will store results for each price column
error_metrics = {}  # Store error metrics for each price column

# Process each price column with additional exogenous regressors
for col in price_cols:
    # Derive prefix to identify matching exogenous columns
    prefix = col.replace('_price', '')
    exog_suffixes = ['_items', '_freight', '_review', '_shipping']
    # Ensure the exogenous columns exist in both train and test datasets
    exog_cols = [prefix + suffix for suffix in exog_suffixes 
                 if (prefix + suffix in df_train.columns) and (prefix + suffix in df_test.columns)]
    print(f"For {col}, using exogenous columns: {exog_cols}")
    
    # Sort training and test data by week
    train_df = df_train.sort_values('week').reset_index(drop=True)
    test_df  = df_test.sort_values('week').reset_index(drop=True)
    
    train_ts = train_df[col]
    test_ts  = test_df[col]
    weeks_test = test_df['week']
    
    # Extract exogenous variables if available
    if exog_cols:
        exog_train = train_df[exog_cols]
        exog_test  = test_df[exog_cols]
    else:
        exog_train = None
        exog_test  = None
    
    # Use auto_arima with exogenous regressors on the training data to select orders
    try:
        from pmdarima import auto_arima
        auto_model = auto_arima(
            train_ts,
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
    
    # Fit SARIMAX model with exogenous regressors using training data
    model = SARIMAX(
        train_ts,
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_fit = model.fit(disp=False)
    
    # Forecast on the test period (using corresponding exogenous values)
    forecast = model_fit.forecast(steps=len(test_ts), exog=exog_test)
    
    # Calculate error metrics using the actual test data
    mse = mean_squared_error(test_ts, forecast)
    mae = mean_absolute_error(test_ts, forecast)
    r2 = r2_score(test_ts, forecast)
    error_metrics[col] = {'mse': mse, 'mae': mae, 'r2': r2}
    
    # Save results for later use
    results[col] = {
        'test': test_ts,
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

# Save predictions to a new folder "prediction_data" in your current working directory
path2 = os.getcwd() + "/data/prediction_data"
file_path2 = os.path.join(path2, "sarima_prediction.csv")
predictions_df.to_csv(file_path2, index=False)
print(f"Predictions have been saved to {file_path2}")

# Save error metrics to a CSV file
error_metrics_df = pd.DataFrame([
    {'price_category': col, 'mse': error_metrics[col]['mse'], 'mae': error_metrics[col]['mae'], 'r2': error_metrics[col]['r2']}
    for col in error_metrics
])
file_path_metrics = os.path.join(path2, "sarima_error_metrics.csv")
error_metrics_df.to_csv(file_path_metrics, index=False)
print(f"Error metrics have been saved to {file_path_metrics}")

# 4. Choose plot option: "combined" or "individual"
plot_option = "combined"  # Change to "individual" for separate plots per category

if plot_option == "combined":
    # Combined Plot: all categories on one graph with each category having a unique color
    color_cycle = itertools.cycle(plt.cm.tab20(np.linspace(0, 1, 20)))
    color_map = {}
    for col in price_cols:
        color_map[col] = next(color_cycle)
    
    plt.figure(figsize=(14, 8))
    for col in price_cols:
        weeks = results[col]['weeks_test']
        actual = results[col]['test']
        forecast = results[col]['forecast']
        
        c = color_map[col]
        plt.plot(weeks, actual, color=c, label=f'Actual {col}')
        plt.plot(weeks, forecast, color=c, linestyle='--', label=f'Forecast {col}')
    
    plt.xlabel('Week')
    plt.ylabel('Price')
    plt.title('Actual vs. Forecast Prices by Category (Test Data)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

elif plot_option == "individual":
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
