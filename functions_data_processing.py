# --- import packages -------------------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

# --- functions -------------------------------------------------------------------------------------------------------

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

# Splits data into train and validation data
def process_data(df, category, time_span = 1, output = "items"):
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
