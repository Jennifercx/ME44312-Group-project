import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

data_dir = os.getcwd()+"\\data\\"
# 1. Load datasets
df_orders = pd.read_csv(data_dir+"olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp"])
df_order_items = pd.read_csv(data_dir+"olist_order_items_dataset.csv")

# Merge order items with orders
df = df_orders.merge(df_order_items, on="order_id")

# Aggregate sales per week
df["week"] = df["order_purchase_timestamp"].dt.to_period("W")
sales_per_week = df.groupby("week")["price"].sum().reset_index()

# Convert week to datetime
sales_per_week["week"] = sales_per_week["week"].dt.start_time

# Sort by date
sales_per_week = sales_per_week.sort_values("week")

# 2. Prepare features with lagged variables
def create_lagged_features(data, lags=1):
    """Creates lag features for time series forecasting."""
    df = data.copy()
    for i in range(1, lags+1):
        df[f"lag_{i}"] = df["price"].shift(i)
    df.dropna(inplace=True)  # Remove NaN values
    return df

data = create_lagged_features(sales_per_week)

# Split features and target
X = data.drop(columns=["week", "price"])
y = data["price"].values.reshape(-1, 1)  # Ensure y is a 2D array

# 3. Normalize features and target
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

# Reshape X for LSTM (3D array)
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 4. Build LSTM Model
model = keras.Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),  # First LSTM layer
    Dropout(0.2),  # Regularization
    LSTM(32),  # Second LSTM layer
    Dropout(0.2),
    Dense(16, activation="relu"),  # Hidden layer
    Dense(1)  # Output layer
])

# Compile the model
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

# 5. Train the model
history = model.fit(X_train_reshaped, y_train, epochs=200, validation_data=(X_test_reshaped, y_test),
                    callbacks=[early_stopping], batch_size=8, verbose=1)

# 6. Evaluate the model
loss, mae = model.evaluate(X_test_reshaped, y_test)
print(f"Test Loss: {loss:.2f}, Test MAE: {mae:.2f}")

# 7. Make predictions and revert scaling
y_pred_scaled = model.predict(X_test_reshaped)
y_pred = scaler_y.inverse_transform(y_pred_scaled)  # Reverse transformation
y_pred_shift = y_pred[1:]
y_test_original = scaler_y.inverse_transform(y_test)  # Reverse transformation for actual values
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

# 9. Predict sales for next week
latest_features = X_scaled[-1].reshape(1, X_train.shape[1], 1)  # Use the latest available data
next_week_prediction_scaled = model.predict(latest_features)
next_week_prediction = scaler_y.inverse_transform(next_week_prediction_scaled)  # Convert back to original scale
print(f"Expected sales for next week: {next_week_prediction[0][0]:.2f}")
