from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from functions_data_processing import process_data_for_category
from matplotlib import pyplot as plt
import os
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output

# Parameters
time_steps = 2
output_name = 'price'
categories = ["automotive", "baby", "beauty_health", "electronics", "entertainment", "fashion", "food", "furniture", "home", "miscellaneous", "office_supplies", "pets", "sports", "tools", "toys"]

# Output directory
os.makedirs("results", exist_ok=True)

# For interactive visual
y_pred_all = {}
y_true_all = {}
histories = {}

for category in categories:
    print(f"\n▶️ Training model for category: {category}")

    # Step 1: Load and process data
    X_train_scaled, X_val_scaled, _, y_train_scaled, y_val_scaled, scaler_y = process_data_for_category(category, time_steps)

    # Step 1.1: Reshape input to 3D
    if hasattr(X_train_scaled, 'to_numpy'):
        X_train_scaled = X_train_scaled.to_numpy()
    if hasattr(X_val_scaled, 'to_numpy'):
        X_val_scaled = X_val_scaled.to_numpy()

    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], time_steps, X_train_scaled.shape[1] // time_steps))
    X_val_scaled = X_val_scaled.reshape((X_val_scaled.shape[0], time_steps, X_val_scaled.shape[1] // time_steps))

    # Step 2: Build LSTM model
    model = keras.Sequential([
        keras.Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
        LSTM(256, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0005), loss="huber", metrics=["mae"])

    early_stopping = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)

    # Step 3: Train the model
    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=200,
        batch_size=8,
        callbacks=[early_stopping],
        verbose=1
    )

    # Step 4: Predict and store
    y_pred_scaled = model.predict(X_val_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_val_scaled)

    y_pred_all[category] = y_pred
    y_true_all[category] = y_true
    histories[category] = history

    # Optional: save the model
    # model.save(f"results/{category}_lstm_model.keras")

# === INTERACTIVE WIDGET VISUALIZATION ===
category_dropdown = widgets.Dropdown(
    options=categories,
    description='Category:',
    layout=widgets.Layout(width='50%')
)

output = widgets.Output()

def plot_selected_category(change):
    with output:
        clear_output(wait=True)
        cat = change['new']

        plt.figure(figsize=(12, 4))

        # Plot 1: Training and validation loss
        plt.subplot(1, 2, 1)
        plt.plot(histories[cat].history['loss'], label='Train Loss')
        plt.plot(histories[cat].history['val_loss'], label='Val Loss')
        plt.title(f'{cat.title()} - Training vs Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot 2: Prediction vs Actual
        plt.subplot(1, 2, 2)
        plt.plot(y_true_all[cat], label='Actual', linewidth=2)
        plt.plot(y_pred_all[cat], label='Predicted', linestyle='--')
        plt.title(f'{cat.title()} - Predicted vs Actual Price')
        plt.xlabel('Week')
        plt.ylabel('Price')
        plt.legend()

        plt.tight_layout()
        plt.show()

category_dropdown.observe(plot_selected_category, names='value')
display(category_dropdown, output)
category_dropdown.value = categories[0]  # trigger initial plot
