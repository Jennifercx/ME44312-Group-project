# imports
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers

from functions_data_processing import generate_X_y, split_data, scale_data
from functions_model_evaluation import validate_model

import os
import pandas as pd

# Parameters
time_steps = 1 # Number of weeks used to predict next weeks data
training_percentage = 0.8
validation_percentage = 0.3 # what percentage of the validation/testing data should be use for validation
output_name = 'price' #Name of the output feature

# The categories list for which product type a model should be created
# categories = ["automotive", "baby", "beauty_health", "electronics", "entertainment", "fashion", "food", "furniture", "home", "miscellaneous", "office_supplies", "pets", "sports", "tools", "toys"]
categories = ["bed_bath_table", "health_beauty", "sports_leisure", "furniture_decor", "computers_accessories"]

# Paths
model_name = 'LSTM' # To store the output name under
data_path = os.path.join(os.getcwd(), "data/processed_data/processed_dataset.csv")
data_set = pd.read_csv(data_path)
os.makedirs("results", exist_ok=True)
result_path = os.path.join(os.getcwd(), "results")

# For storage
y_pred_all = {}
y_true_all = {}
histories = {}
error_metrics = {}

for category in categories:
    # print(f"\n▶️ Training model for category: {category}")

    # Load and process data
    X, y = generate_X_y(data_set, category, time_span = time_steps, output = output_name)

    # Create train, validate, and test data sets
    training_weeks = int(training_percentage * len(X))
    X_train, X_test = split_data(X, training_weeks)
    y_train, y_test = split_data(y, training_weeks)

    validation_weeks = int((1-validation_percentage) * training_weeks)
    X_validate = X_train[validation_weeks:]
    y_validate = y_train[validation_weeks:]

    # Scale train and validation data
    X_train_scaled, X_validate_scaled, X_test_scaled, X_scaler = scale_data(X_train, X_validate, X_test)
    y_train_scaled, y_validate_scaled, y_test_scaled, y_scaler = scale_data(y_train, y_validate, y_test)

    # Reshape data to format the model wants
    if hasattr(X_train_scaled, 'to_numpy'):
        X_train_scaled = X_train_scaled.to_numpy()
    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], time_steps, X_train_scaled.shape[1] // time_steps))
    if hasattr(X_validate_scaled, 'to_numpy'):
        X_validate_scaled = X_validate_scaled.to_numpy()
    X_validate_scaled = X_validate_scaled.reshape((X_validate_scaled.shape[0], time_steps, X_validate_scaled.shape[1] // time_steps))
    if hasattr(X_test_scaled, 'to_numpy'):
        X_test_scaled = X_test_scaled.to_numpy()
    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], time_steps, X_test_scaled.shape[1] // time_steps))

    # Build LSTM model
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

    # Train the model
    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_validate_scaled, y_validate_scaled),
        epochs=200,
        batch_size=8,
        callbacks=[early_stopping],
        verbose=1
    )

    # Validate model
    y_pred, y_true, mae, mse, r2 = validate_model(model, X_test_scaled, y_test, y_scaler)

    # Store validation data
    error_metrics[category] = {'mse': mse, 'mae': mae, 'r2': r2}
    y_pred_all[category] = y_pred
    y_true_all[category] = y_true
    histories[category] = history

# Store error metrics
temp_df = pd.DataFrame(error_metrics)
file_path_metrics = os.path.join(result_path, model_name + "_error_metrics.csv")
temp_df.to_csv(file_path_metrics, index=False)

# Store y_pred metrics
temp_df = pd.DataFrame({
    category: preds.ravel()
    for category, preds in y_pred_all.items()
})
file_path_metrics = os.path.join(result_path, model_name + "_y_pred.csv")
temp_df.to_csv(file_path_metrics, index=False)

# Store y_true metrics
temp_df = pd.DataFrame({
    category: preds.ravel()
    for category, preds in y_true_all.items()
})
file_path_metrics = os.path.join(result_path, model_name + "_y_true.csv")
temp_df.to_csv(file_path_metrics, index=False)

# Store histories metrics
for category, history in histories.items():
    temp_df = pd.DataFrame(history.history)
    file_path = os.path.join(result_path, model_name + f"_history_{category}.csv")
    temp_df.to_csv(file_path, index=False)