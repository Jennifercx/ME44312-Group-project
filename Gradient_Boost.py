import pandas as pd
from xgboost import XGBRegressor
from functions import create_input_target_vectors, scale_data, evaluate_model, predict_data, ModelEvaluation
import os

# 1. Load the data
data_dir = os.path.join(os.getcwd(), "data")
processed_data_dir = os.path.join(data_dir, "processed_data")
train_set = pd.read_csv(os.path.join(processed_data_dir, "train_data.csv"))

# 2. Create input and target vectors
time_steps = 2  # Number of time steps (weeks) to look back
X, y = create_input_target_vectors(train_set, time_steps)  # Adjust time_steps as needed

# 3. Create test and train data, normalize features and target
n_features = 102
X_train_scaled, X_val_scaled, scaler_X, y_train_scaled, y_val_scaled, scaler_y = scale_data(X, y, time_steps, n_features)
X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], -1)
X_val_scaled = X_val_scaled.reshape(X_val_scaled.shape[0], -1)

# 4. Build the Random Forest & XGBoost
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# 5. Train the models
model.fit(X_train_scaled, y_train_scaled)

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