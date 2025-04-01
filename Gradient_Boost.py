from xgboost import XGBRegressor
from functions_data_processing import process_data
from functions_model_evaluation import validate_model

# 1. process data
time_steps = 2
X_train_scaled, X_val_scaled, scaler_X, y_train_scaled, y_val_scaled, scaler_y = process_data(time_steps)

# 2. Build the Random Forest & XGBoost
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# 3. Train the models
model.fit(X_train_scaled, y_train_scaled)

# 4. validate model
validate_model(model, X_val_scaled, y_val_scaled, scaler_y)