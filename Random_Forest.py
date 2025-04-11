from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from functions_data_processing import process_data, process_data_2
from functions_model_evaluation import validate_model, evaluate_model
import os
import pandas as pd
import numpy as np

# 1. process data
#product_categories = ['category_1', 'category_2', 'category_3', ...]  # Fill this with your actual categories
product_categories = ["bed_bath_table", "health_beauty", "sports_leisure", "furniture_decor", "computers_accessories"]
path = "data/processed_data/processed_dataset.csv"
data_set = pd.read_csv(os.path.join(os.getcwd(), path))



time_steps = 2
models = {}  # To store the models for each category
error_metrics = {}


for category in product_categories:
    # Extract only the columns for the current category (assume naming like 'category_1_feature1', etc.)


    # Scale and reshape data (assuming your process_data handles this)
    X_train, X_val, scaler_X, y_train, y_val, scaler_y = process_data_2(data_set, category, time_span = time_steps)
    y_train = np.array(y_train).ravel()
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    models[category] = model

    mae, mse, r2 = validate_model(model, X_val, y_val, scaler_y)
    error_metrics[category] = {'mse': mse, 'mae': mae, 'r2': r2}
    
print(error_metrics)
# time_steps = 2
# X_train_scaled, X_val_scaled, scaler_X, y_train_scaled, y_val_scaled, scaler_y = process_data(time_steps)

# # 2. Build the Random Forest & XGBoost
# model = RandomForestRegressor(n_estimators=100, random_state=42)

# # 3. Train the models
# model.fit(X_train_scaled, y_train_scaled)

# # Grid Search for Random Forest
# param_grid = {
#     'n_estimators': [50, 100],
#     'max_depth': [3, 5],
#     'min_samples_split': [2, 5]
# }

# grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3,
#                            scoring='neg_mean_absolute_error', n_jobs=-1)
# grid_search.fit(X_val_scaled, y_val_scaled)

# print("Best Parameters for Random Forest:", grid_search.best_params_)

# # 4. validate model
# validate_model(model, X_val_scaled, y_val_scaled, scaler_y)