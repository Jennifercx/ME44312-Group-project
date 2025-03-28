# --- import packages -------------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np
import matplotlib.pyplot as plt

# --- create input and target vectors for time series forecasting -------------------------------------------------------------------------------------------------------

def create_input_target_vectors(df, time_span):
    """
    Creates input and target vectors for time series forecasting while keeping the week structure.

    Args:
        df (pd.DataFrame): The input DataFrame containing weekly data.
        time_span (int): The number of weeks to use as input for each prediction.

    Returns:
        tuple: A tuple containing the input vector (X) and the target vector (y).
    """
    X = []
    y = []
    price_cols = [col for col in df.columns if 'price' in col]
    for i in range(len(df) - time_span ):
        X.append(df.iloc[i:i + time_span].drop(columns=['week']).values)
        y.append(df.iloc[i + time_span][price_cols].values)
    return np.array(X), np.array(y)

def scale_data(X, y, time_steps, n_features):
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 3. Normalize features and target
    S, T, F = len(X_train), time_steps, n_features  # 1000 samples, 10 weeks, 102 features

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

    return X_train_scaled, X_val_scaled, scaler_X, y_train_scaled, y_val_scaled, scaler_y

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, R^2: {r2:.2f}")
    return 0

def predict_data(model, X_test, y_test, scaler_y):
    y_pred = scaler_y.inverse_transform(model.predict(X_test))  # Reverse transformation
    y_pred = y_pred[:-1]                 # Shift data since we predict the next week based on the current week
    y_true = scaler_y.inverse_transform(y_test)  # Reverse transformation for actual values
    y_true = y_true[1:]

    # evaluate model
    evaluate_model(y_true, y_pred)

    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="Actual sales", marker="o")
    plt.plot(y_pred, label="Predicted sales", marker="x")
    plt.legend()
    plt.title("Predicted vs. Actual Sales")
    plt.xlabel("Time (weeks)")
    plt.ylabel("Sales Price")
    plt.show()
    return 0