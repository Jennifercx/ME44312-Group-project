import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load the data
df_orders = pd.read_csv("archive/olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp"])
df_order_items = pd.read_csv("archive/olist_order_items_dataset.csv")

# Merge order items with orders
df = df_orders.merge(df_order_items, on="order_id")

# Aggregate sales per week
df["week"] = df["order_purchase_timestamp"].dt.to_period("W")
sales_per_week = df.groupby("week")["price"].sum().reset_index()

# Convert week to datetime
# sales_per_week["week"] = sales_per_week["week"].astype(str)
# sales_per_week["week"] = pd.to_datetime(sales_per_week["week"])

sales_per_week["week"] = sales_per_week["week"].dt.start_time


# Sort by date
sales_per_week = sales_per_week.sort_values("week")

# 2. Prepare data for neural network
def create_lagged_features(data, lags=4):
    """Creates lag features for time series forecasting"""
    df = data.copy()
    for i in range(1, lags+1):
        df[f"lag_{i}"] = df["price"].shift(i)
    df.dropna(inplace=True)  # Drop NaN values from shifting
    return df

data = create_lagged_features(sales_per_week)

# Split features and labels
X = data.drop(columns=["week", "price"])
y = data["price"]

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

from tensorflow.keras.layers import LSTM, Dense

X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model = keras.Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50),
    Dense(25, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = model.fit(X_train_reshaped, y_train, epochs=100, validation_data=(X_test_reshaped, y_test))




# # 3. Define a Simple Neural Network
# model = keras.Sequential([
#     keras.layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
#     keras.layers.Dropout(0.3),  # Voorkomt overfitting
#     keras.layers.Dense(64, activation="relu"),
#     keras.layers.Dense(32, activation="relu"),
#     keras.layers.Dense(1)
    
#     # keras.layers.Dense(16, activation="relu", input_shape=(X_train.shape[1],)),
#     # keras.layers.Dense(8, activation="relu"),
#     # keras.layers.Dense(8, activation="relu"),
#     # keras.layers.Dense(8, activation="relu"),
#     # keras.layers.Dense(8, activation="relu"),
#     # keras.layers.Dense(8, activation="relu"),
#     # keras.layers.Dense(8, activation="relu"),
#     # keras.layers.Dense(8, activation="relu"),
#     # keras.layers.Dense(8, activation="relu"),
#     # keras.layers.Dense(8, activation="relu"),
#     # keras.layers.Dense(1)  # Output layer for regression
# ])

# model.compile(optimizer="adam", loss="mse", metrics=["mae"])
# early_stopping = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

# # 4. Train the model
# history = model.fit(X_train, y_train, epochs=500, batch_size=8, validation_data=(X_test, y_test), callbacks=[early_stopping],  verbose=1)

# 5. Evaluate model
# loss, mae = model.evaluate(X_test, y_test)
loss, mae = model.evaluate(X_test_reshaped, y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# 6. Predict next week's sales
# latest_features = X_scaled[-1].reshape(1, -1)  # Get latest available data point
# next_week_prediction = model.predict(latest_features)
# print(f"Predicted sales for next week: {next_week_prediction[0][0]}")

latest_features = X_scaled[-1].reshape(1, X_train.shape[1], 1)  # Voeg extra dimensie toe
next_week_prediction = model.predict(latest_features)
print(f"Predicted sales for next week: {next_week_prediction[0][0]}")

# # Plot training history
# plt.plot(history.history["loss"], label="Train Loss")
# plt.plot(history.history["val_loss"], label="Val Loss")
# plt.legend()
# plt.title("Model Training Performance")
# plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Get model predictions on test data
# y_pred = model.predict(X_test)
y_pred = model.predict(X_test_reshaped)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Plot actual vs predicted sales
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual Sales", marker="o")
plt.plot(y_pred, label="Predicted Sales", marker="x")
plt.legend()
plt.title("Predicted vs. Actual Sales")
plt.show()



