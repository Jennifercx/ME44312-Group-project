# --- import packages -------------------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os

# --- functions -------------------------------------------------------------------------------------------------------

# create input and output data
def create_input_output_old(df, time_span, output):
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
    output_cols = [col for col in df.columns if output in col] # 'price' 'items'
    for i in range(len(df) - time_span - 1):
        X.append(df.iloc[i:i + time_span].drop(columns=['week']).values)
        y.append(df.iloc[1 + i + time_span][output_cols].values)
    
    # make X and y 2D
    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], -1)
    y = y.reshape(y.shape[0], -1)

    return X, y

def create_input_output(df, time_span, output_name):
    X = []
    y = []
    output_cols = [col for col in df.columns if output_name in col] # 'price' 'items'
    for i in range(len(df) - time_span - 1): # -1 is to shift the data between train and test
        X.append(df.iloc[i:i + time_span].drop(columns=['week']).values)
        y.append(df.iloc[i + time_span + 1][output_cols].values) # 1 is to select the test as next week
    
    # make X and y 2D
    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], -1)
    y = y.reshape(y.shape[0], -1)

    return X, y

# Normalizes data
def scale_data_old(X, y, time_steps, n_features):
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
def scale_data(train, test):

    # step 1: transform to pd.dataframe
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)

    # Step 1: Fit the scaler on training data only
    scaler = MinMaxScaler()
    scaler.fit(train)

    # step 2: fit per column
    train_scaled = pd.DataFrame(scaler.transform(train))
    test_scaled = pd.DataFrame(scaler.transform(test))
  
    return train_scaled, test_scaled, scaler

# combines several functions
def process_data_old(time_steps, output = "items", data_path = "data/processed_data", train_data = "train_data.csv", test_data = "test_data.csv"):

    # 1. Load the data
    data_dir = os.path.join(os.getcwd(), data_path)
    train_set = pd.read_csv(os.path.join(data_dir, train_data))
    test_set = pd.read_csv(os.path.join(data_dir, test_data))

    # 2. Create input and target vectors
    time_steps = 2  # Number of time steps (weeks) to look back
    X_train, y_train = create_input_output(train_set, time_steps, output)  # Adjust time_steps as needed
    X_test, y_test = create_input_output(test_set, time_steps, output)  # Adjust time_steps as needed

    # 3. normalize features and target
    X_train_scaled, X_val_scaled, scaler_X = scale_data(X_train, X_test)
    y_train_scaled, y_val_scaled, scaler_y = scale_data(y_train, y_test)

    # return processed data
    return X_train_scaled, X_val_scaled, scaler_X, y_train_scaled, y_val_scaled, scaler_y

def process_data(time_steps, output = "items", path = "data/processed_data/processed_dataset.csv"):

    # 1. Load the data
    data_set = pd.read_csv(os.path.join(os.getcwd(), path))

    # 2. Create input and target vectors
    time_steps = 2  # Number of time steps (weeks) to look back
    X, y = create_input_output(data_set, time_steps, output) 

    # 3. Split data into test and train
    # random splitting
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # This determines how week the validation data should be
    n_test = 17 #

    # Split data: Train data will be everything except the last n_test samples, Test data will be the last n_test samples
    X_train, X_test = X[:-n_test], X[-n_test:]
    y_train, y_test = y[:-n_test], y[-n_test:]

    # 3. normalize features and target
    X_train_scaled, X_val_scaled, scaler_X = scale_data(X_train, X_test)
    y_train_scaled, y_val_scaled, scaler_y = scale_data(y_train, y_test)

    # return processed data
    return X_train_scaled, X_val_scaled, scaler_X, y_train_scaled, y_val_scaled, scaler_y

def process_data_2(df, category, time_span = 1, output = "items"):
    '''
    input
    time_span is the number of week in the past the model should use to predict next week
    df is the dataframe containing the dataset
    category is the name of the product category that should be filtered from df
    output is the name of the output feature

    function
    this function creates test and validation data for a single product category

    output
    '''

    # Filter out only the selected product category with product
    input_cols = df[[col for col in df.columns if col.startswith(category)]]
    output_cols = df[[category + '_' + output]]

    # 2. Create input and target vectors
    X = []
    y = []
    for i in range(len(df) - time_span - 1): # -1 is to shift the data between train and test
        X.append(input_cols.iloc[i:i + time_span].values)
        y.append(output_cols.iloc[i + time_span + 1].values) # 1 is to select the test as next week
    
    # make X and y 2D
    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], -1)
    y = y.reshape(y.shape[0], -1)

    # X = X.reshape((X.shape[0], time_span, X.shape[1] // time_span))
    # y = y.reshape((y.shape[0], time_span, y.shape[1] // time_span))

    # 3. Split data into test and train
    # random splitting
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # This determines how week the validation data should be 
    n_test = 17 + time_span 

    # Split data: Train data will be everything except the last n_test samples, Test data will be the last n_test samples
    X_train, X_val = X[:-n_test], X[-n_test:]
    y_train, y_val  = y[:-n_test], y[-n_test:]

    # 3. normalize features and target
    X_train_scaled, X_val_scaled, scaler_X = scale_data(X_train, X_val)
    y_train_scaled, y_val_scaled, scaler_y = scale_data(y_train, y_val)

    # return processed data
    return X_train_scaled, X_val_scaled, scaler_X, y_train_scaled, y_val_scaled, scaler_y