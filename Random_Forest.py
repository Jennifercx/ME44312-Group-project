import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from functions import create_input_target_vectors, scale_data, scale_per_column_data, evaluate_model, predict_data, ModelEvaluation
import os
import matplotlib.pyplot as plt

# 1. Load the data
data_dir = os.path.join(os.getcwd(), "data")
processed_data_dir = os.path.join(data_dir, "processed_data")
train_set = pd.read_csv(os.path.join(processed_data_dir, "train_data.csv"))

# 2. Create input and target vectors
time_steps = 2  # Number of time steps (weeks) to look back
X, y = create_input_target_vectors(train_set, time_steps)  # Adjust time_steps as needed

# 3. Create test and train data, normalize features and target
n_features = 102
# X_train_scaled, X_val_scaled, scaler_X, y_train_scaled, y_val_scaled, scaler_y = scale_data(X, y, time_steps, n_features)

X = X.reshape(X.shape[0], -1)
y = y.reshape(y.shape[0], -1)
print(X.shape)
X = pd.DataFrame(X)
y = pd.DataFrame(y)

X_train_scaled, X_val_scaled, scaler_X, y_train_scaled, y_val_scaled, scaler_y = scale_per_column_data(X, y)
print(X_train_scaled)
# 3.1 make X_data 2D for the model to work
# X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], -1)
# X_val_scaled = X_val_scaled.reshape(X_val_scaled.shape[0], -1)

# 4. Build the Random Forest & XGBoost
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 5. Train the models
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

# 6. Predict and evaluate model
y_true, y_pred = predict_data(model, X_val_scaled, y_val_scaled, scaler_y)

# 7. Evaluate model
evaluate_model(y_true, y_pred)

# Widget
LSTMev = ModelEvaluation(y_true, y_pred, name="LSTM")
categories = ["automotive", "baby", "beauty_health", "construction_tools", "electronics", "entertainment", "fashion", "food", "furniture", "garden_tools", "gifts", "home_appliances", "housewares", "luggage", "office_supplies", "other", "pets", "sports", "telephony", "toys"]
LSTMev.set_categories(categories)
LSTMev.plot_categories()
print(LSTMev.EvaluateResults())