# --- import packages -------------------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os

# --- functions -------------------------------------------------------------------------------------------------------

# create input and output data
def create_input_output(df, time_span, output):
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
    price_cols = [col for col in df.columns if output in col] # 'price' 'items'
    for i in range(len(df) - time_span ):
        X.append(df.iloc[i:i + time_span].drop(columns=['week']).values)
        y.append(df.iloc[i + time_span][price_cols].values)
    
    # make X and y 2D
    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], -1)
    y = y.reshape(y.shape[0], -1)

    return X, y

# Normalizes data
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

# Normalizes data per column
def scale_per_column_data(train, test):

    # step 1: transform to pd.dataframe
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)

    # Step 1: Fit the scaler on training data only
    scaler = MinMaxScaler()

    # step 2: fit per column
    train_scaled = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
    test_scaled = pd.DataFrame(scaler.fit_transform(test), columns=test.columns)
  
    return train_scaled, test_scaled, scaler

# combines several functions
def process_data(time_steps, output = "item", data_path = "data/processed_data", train_data = "train_data.csv", test_data = "test_data.csv"):

    # 1. Load the data
    data_dir = os.path.join(os.getcwd(), data_path)
    train_set = pd.read_csv(os.path.join(data_dir, train_data))
    test_set = pd.read_csv(os.path.join(data_dir, test_data))

    # 2. Create input and target vectors
    time_steps = 2  # Number of time steps (weeks) to look back
    X_train, y_train = create_input_output(train_set, time_steps, output)  # Adjust time_steps as needed
    X_test, y_test = create_input_output(test_set, time_steps, output)  # Adjust time_steps as needed

    # 3. normalize features and target
    X_train_scaled, X_val_scaled, scaler_X = scale_per_column_data(X_train, X_test)
    y_train_scaled, y_val_scaled, scaler_y = scale_per_column_data(y_train, y_test)

    # return processed data
    return X_train_scaled, X_val_scaled, scaler_X, y_train_scaled, y_val_scaled, scaler_y
