from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from functions_data_processing import process_data
from functions_model_evaluation import validate_model

# 1. process data
time_steps = 2
X_train_scaled, X_val_scaled, scaler_X, y_train_scaled, y_val_scaled, scaler_y = process_data(time_steps)

# 1.1 reshape data
X_train_scaled = X_train_scaled.to_numpy()
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], time_steps, X_train_scaled.shape[1] // time_steps))
X_val_scaled = X_val_scaled.to_numpy()
X_val_scaled = X_val_scaled.reshape((X_val_scaled.shape[0], time_steps, X_val_scaled.shape[1] // time_steps))

# 2. Build LSTM Model
model = keras.Sequential([
    LSTM(256, return_sequences=True, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),  # First LSTM layer
    Dropout(0.2),  # Regularization
    LSTM(32),  # Second LSTM layer
    Dropout(0.1),  # Regularization
    #Dense(16, activation="relu"),  # Hidden layer
    Dense(15)  # Output layer
])

# Compile the model
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

# 3. Train the model
history = model.fit(X_train_scaled, y_train_scaled, epochs=200, validation_data=(X_val_scaled, y_val_scaled),
                    callbacks=[early_stopping], batch_size=8, verbose=1)

# 4. validate model
validate_model(model, X_val_scaled, y_val_scaled, scaler_y)