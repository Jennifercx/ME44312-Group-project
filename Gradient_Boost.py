from xgboost import XGBRegressor
from functions_data_processing import process_data_scaled
from functions_model_evaluation import validate_model
import numpy as np
import os
import pandas as pd

# Parameters
time_steps = 2
output_name = 'price'
# categories = ["automotive", "baby", "beauty_health", "electronics", "entertainment", "fashion", "food", "furniture", "home", "miscellaneous", "office_supplies", "pets", "sports", "tools", "toys"]
product_categories = ["bed_bath_table", "health_beauty", "sports_leisure", "furniture_decor", "computers_accessories"]
model_name = 'GB'

# Paths
data_path = os.path.join(os.getcwd(), "data/processed_data/processed_dataset.csv")
data_set = pd.read_csv(data_path)
os.makedirs("results", exist_ok=True)
result_path = os.path.join(os.getcwd(), "results")

# For storage
y_pred_all = {}
y_true_all = {}
histories = {}
error_metrics = {}

# Extract only the columns for the current category (assume naming like 'category_1_feature1', etc.)
for category in product_categories:

    # Scale and reshape data
    X_train_scaled, X_val_scaled, scaler_X, y_train_scaled, y_val_scaled, scaler_y = process_data_scaled(data_set, category, time_span = time_steps)
    
    # Reshape data to format the model wants
    y_train_scaled = np.ravel(y_train_scaled)
    y_val_scaled = np.ravel(y_val_scaled)

    # Train model
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train_scaled, y_train_scaled)

    # Validate model
    y_pred, y_true, mae, mse, r2 = validate_model(model, X_val_scaled, y_val_scaled.reshape(-1, 1), scaler_y)

    # Store validation data
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