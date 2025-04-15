from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from functions_data_processing import generate_X_y, split_data, scale_data
from functions_model_evaluation import validate_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import os
import pandas as pd
import numpy as np

# Parameters
time_steps = 1 # Number of weeks used to predict next weeks data
training_percentage = 0.9
output_name = 'price' #Name of the output feature

# The categories list for which product type a model should be created
# categories = ["automotive", "baby", "beauty_health", "electronics", "entertainment", "fashion", "food", "furniture", "home", "miscellaneous", "office_supplies", "pets", "sports", "tools", "toys"]
categories = ["bed_bath_table", "health_beauty", "sports_leisure", "furniture_decor", "computers_accessories"]

# Paths
model_name = 'RF' # To store the output name under
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
for category in categories:

    # Load and process data
    X, y = generate_X_y(data_set, category, time_span = time_steps, output = output_name)
    print(X.shape)
    print(np.sum(X[:, 0]))
    # Create train, validate, and test data sets
    training_weeks = int(training_percentage * len(X))
    X_train, X_test = split_data(X, training_weeks)
    y_train, y_test  = split_data(y, training_weeks)

    # Reshape data to format the model wants
    y_train = np.ravel(y_train)
    y_test  = np.ravel(y_test )

    # Train model
    base_model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5]
    }

    # Grid search
    grid_search = GridSearchCV(
        base_model, param_grid,
        cv=3, scoring='neg_mean_absolute_error',
        n_jobs=-1, return_train_score=True
    )
    grid_search.fit(X_train, y_train)

    # Store "history" from grid search results
    history_df = pd.DataFrame(grid_search.cv_results_)

    # Train final model using best params
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Test model
    y_pred = best_model.predict(X_test)

    # 2. Evaluate model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Store test data
    error_metrics[category] = {'mse': mse, 'mae': mae, 'r2': r2}
    y_pred_all[category] = y_pred
    y_true_all[category] = y_test
    histories[category] = history_df
    
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

# Store histories metrics
for category, history_df in histories.items():
    path = os.path.join(result_path, model_name + f"_history_{category}.csv")
    history_df.to_csv(path, index=False)
