import pandas as pd
from tensorflow import keras
import os
from functions import create_input_target_vectors, scale_data, evaluate_model, predict_data, ModelEvaluation

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

# 4. Build LSTM Model
model = keras.Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),  # First LSTM layer
    Dropout(0.2),  # Regularization
    LSTM(32),  # Second LSTM layer
    Dropout(0.2),
    Dense(16, activation="relu"),  # Hidden layer
    Dense(20)  # Output layer
])

# Compile the model
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

# 5. Train the model
history = model.fit(X_train_scaled, y_train_scaled, epochs=200, validation_data=(X_val_scaled, y_val_scaled),
                    callbacks=[early_stopping], batch_size=8, verbose=1)

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