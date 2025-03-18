# Importeren van benodigde libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Laad de datasets
df_orders = pd.read_csv("archive/olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp"])
df_order_items = pd.read_csv("archive/olist_order_items_dataset.csv")

# Merge order items met orders
df = df_orders.merge(df_order_items, on="order_id")

# Aggregate sales per week
df["week"] = df["order_purchase_timestamp"].dt.to_period("W")
sales_per_week = df.groupby("week")["price"].sum().reset_index()

# Convert week naar datetime
sales_per_week["week"] = sales_per_week["week"].dt.start_time

# Sorteer op datum
sales_per_week = sales_per_week.sort_values("week")

# 2. Voorbereiden van features met lagged variabelen
def create_lagged_features(data, lags=1):
    """Creëert lag features voor tijdreeksvoorspelling."""
    df = data.copy()
    for i in range(1, lags+1):
        df[f"lag_{i}"] = df["price"].shift(i)
    df.dropna(inplace=True)  # Verwijder NaN-waarden
    return df

data = create_lagged_features(sales_per_week)

# Splits features en target
X = data.drop(columns=["week", "price"])
y = data["price"].values.reshape(-1, 1)  # Zorg dat y een 2D-array is

# 3. Normalisatie van features en target
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)

# Splits dataset in train- en testsets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

# Reshape X voor LSTM (3D-array)
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 4. LSTM Model
model = keras.Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),  # Eerste LSTM laag
    Dropout(0.2),  # Regularisatie
    LSTM(32),  # Tweede LSTM laag
    Dropout(0.2),
    Dense(16, activation="relu"),  # Verborgen laag
    Dense(1)  # Output laag
])

# Compileren van het model
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Early Stopping om overfitting te voorkomen
early_stopping = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

# 5. Train het model
history = model.fit(X_train_reshaped, y_train, epochs=200, validation_data=(X_test_reshaped, y_test), #batch_size=8, verbose=1)
                    callbacks=[early_stopping], batch_size=8, verbose=1)

# 6. Evaluatie van het model
loss, mae = model.evaluate(X_test_reshaped, y_test)
print(f"Test Loss: {loss:.2f}, Test MAE: {mae:.2f}")

# 7. Voorspellingen maken en omzetten naar originele schaal
y_pred_scaled = model.predict(X_test_reshaped)
y_pred = scaler_y.inverse_transform(y_pred_scaled)  # Omgekeerde transformatie

y_test_original = scaler_y.inverse_transform(y_test)  # Omgekeerde transformatie voor echte waarden

# 8. Visualisatie van de prestaties
plt.figure(figsize=(12, 5))
plt.plot(y_test_original, label="Actual sales", marker="o")
plt.plot(y_pred, label="Predicted sales", marker="x")
plt.legend()
plt.title("Predicted vs. Actual sales")
plt.xlabel("Time (weeks)")
plt.ylabel("Salesprice")
plt.show()

# 9. Voorspelling maken voor volgende week
latest_features = X_scaled[-1].reshape(1, X_train.shape[1], 1)  # Gebruik laatste beschikbare data
next_week_prediction_scaled = model.predict(latest_features)
next_week_prediction = scaler_y.inverse_transform(next_week_prediction_scaled)  # Terugzetten naar originele schaal
print(f"Verwachte verkoop voor volgende week: {next_week_prediction[0][0]:.2f}")
