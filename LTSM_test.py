import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from data_processing import create_input_target_vectors
from Testing import ModelEvaluation

# 1. Load the data

data_dir = os.path.join(os.getcwd(), "data")
processed_data_dir = os.path.join(data_dir, "processed_data")

train_set = pd.read_csv(os.path.join(processed_data_dir, "train_data.csv"))

# 2. Create input and target vectors
time_steps = 2  # Number of time steps (weeks) to look back
X, y = create_input_target_vectors(train_set, time_steps)  # Adjust time_steps as needed

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# 3. Normalize features and target
S, T, F = len(X_train), time_steps, 102  # 1000 samples, 10 weeks, 102 features

# Step 1: Reshape to 2D (flatten time steps)
X_train_reshaped = X_train.reshape(-1, F)  # Shape (S*T, 102)
X_val_reshaped = X_val.reshape(-1, F)      # Shape (S_val*T, 102)

# Step 2: Fit the scaler on training data only
scaler_X = MinMaxScaler()
scaler_X.fit(X_train_reshaped)
scaler_y = MinMaxScaler()
scaler_y.fit(y_train) 

# Step 3: Transform both train and validation sets
X_train_scaled = scaler_X.transform(X_train_reshaped)
X_val_scaled = scaler_X.transform(X_val_reshaped)

# Step 4: Reshape back to original 3D shape
X_train_scaled = X_train_scaled.reshape(S, T, F)
X_val_scaled = X_val_scaled.reshape(X_val.shape[0], T, F)

y_train_scaled = scaler_y.transform(y_train)  
y_val_scaled = scaler_y.transform(y_val) 

# Reshape X for LSTM (3D array)
#X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
#X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 4. Build LSTM Model
model = keras.Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),  # First LSTM layer
    Dropout(0.2),  # Regularization
    LSTM(32),  # Second LSTM layer
    Dropout(0.2),  # Regularization
    Dense(16, activation="relu"),  # Hidden layer
    Dense(20)  # Output layer
])

# Compile the model
model.compile(optimizer="adam", loss="huber", metrics=["mae"])

# Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)

# 5. Train the model
history = model.fit(X_train_scaled, y_train_scaled, epochs=200, validation_data=(X_val_scaled, y_val_scaled),
                    callbacks=[early_stopping], batch_size=8, verbose=1)

# Get the loss values from the history object
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plotting
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')

# 6. Evaluate the model
loss, mae = model.evaluate(X_val_scaled, y_val_scaled)
print(f"Test Loss: {loss:.2f}, Test MAE: {mae:.2f}")

# 7. Make predictions and revert scaling
y_pred_scaled = model.predict(X_val_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)  # Reverse transformation
y_pred_shift = y_pred[1:]
y_test_original = scaler_y.inverse_transform(y_val_scaled)  # Reverse transformation for actual values 
y_test_trimmed = y_test_original[:-1]
print()

# 8. Visualize model performance
plt.figure(figsize=(12, 5))
plt.plot(y_test_trimmed, label="Actual sales", marker="o")
plt.plot(y_pred_shift, label="Predicted sales", marker="x")
plt.legend()
plt.title("Predicted vs. Actual Sales")
plt.xlabel("Time (weeks)")
plt.ylabel("Sales Price")
plt.show()

LSTMev = ModelEvaluation(y_test_trimmed, y_pred_shift, name="LSTM")
categories = ["automotive", "baby", "beauty_health", "construction_tools", "electronics", "entertainment", "fashion", "food", "furniture", "garden_tools", "gifts", "home_appliances", "housewares", "luggage", "office_supplies", "other", "pets", "sports", "telephony", "toys"]
LSTMev.set_categories(categories)
LSTMev.plot_categories()
print(LSTMev.EvaluateResults())
