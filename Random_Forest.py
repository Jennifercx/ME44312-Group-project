from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from functions_data_processing import process_data
from functions_model_evaluation import validate_model

# 1. process data
time_steps = 2
X_train_scaled, X_val_scaled, scaler_X, y_train_scaled, y_val_scaled, scaler_y = process_data(time_steps)

# 2. Build the Random Forest & XGBoost
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 3. Train the models
model.fit(X_train_scaled, y_train_scaled)

# Grid Search for Random Forest
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3,
                           scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_val_scaled, y_val_scaled)

print("Best Parameters for Random Forest:", grid_search.best_params_)

# 4. validate model
validate_model(model, X_val_scaled, y_val_scaled, scaler_y)