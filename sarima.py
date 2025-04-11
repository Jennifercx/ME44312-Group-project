import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from functions_data_processing import process_data


warnings.filterwarnings("ignore")

# Parameters
time_steps = 1
output_name = 'price'
# categories = ["automotive", "baby", "beauty_health", "electronics", "entertainment", "fashion", "food", "furniture", "home", "miscellaneous", "office_supplies", "pets", "sports", "tools", "toys"]
categories = ["bed_bath_table", "health_beauty", "sports_leisure", "furniture_decor", "computers_accessories"]
model_name = 'ARIMA'

# Paths
data_path = os.path.join(os.getcwd(), "data/processed_data/processed_dataset.csv")
data_set = pd.read_csv(data_path)
os.makedirs("results", exist_ok=True)
result_path = os.path.join(os.getcwd(), "results")

# For storage
y_pred_all = {}
y_true_all = {}
error_metrics = {}

for category in categories:

    # Load and process data
    X_train_scaled, X_val_scaled, _, y_train_scaled, y_val_scaled, scaler_y = process_data(data_set, category, time_span = time_steps)

    # Use auto_arima with exogenous regressors on the training data to select orders
    auto_model = auto_arima(
        y_train_scaled,
        exogenous=X_train_scaled,
        seasonal=True,
        m=12,  # Adjust seasonal period if needed
        error_action='ignore',
        suppress_warnings=True
    )
    order = auto_model.order
    seasonal_order = auto_model.seasonal_order
    
    # Fit SARIMAX model with exogenous regressors using training data
    model = SARIMAX(
        y_train_scaled,
        exog=X_train_scaled,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_fit = model.fit(disp=False)
    
    # Forecast on the test period (using corresponding exogenous values)
    y_pred_scaled = model_fit.forecast(steps=len(y_val_scaled), exog=X_val_scaled)
    
    # Calculate error metrics using the actual test data
    y_pred = scaler_y.inverse_transform(np.array(y_pred_scaled).reshape(-1,1))
    y_true = scaler_y.inverse_transform(np.array(y_val_scaled).reshape(-1,1))

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    error_metrics[category] = {'mse': mse, 'mae': mae, 'r2': r2}
    
    y_pred_all[category] = y_pred
    y_true_all[category] = y_true

# Store error metrics
temp_df = pd.DataFrame(error_metrics)
file_path_metrics = os.path.join(result_path, model_name + "_error_metrics.csv")
temp_df.to_csv(file_path_metrics, index=False)

# Store y_pred metrics
temp_df = pd.DataFrame({
    category: preds.ravel()
    for category, preds in y_pred_all.items()
})
file_path_metrics = os.path.join(result_path, model_name + "_y_pred.csv")
temp_df.to_csv(file_path_metrics, index=False)

# Store y_true metrics
temp_df = pd.DataFrame({
    category: preds.ravel()
    for category, preds in y_true_all.items()
})
file_path_metrics = os.path.join(result_path, model_name + "_y_true.csv")
temp_df.to_csv(file_path_metrics, index=False)
