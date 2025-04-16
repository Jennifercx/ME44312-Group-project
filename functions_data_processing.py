# --- import packages -------------------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

# --- functions -------------------------------------------------------------------------------------------------------

# Splits data into train and validation data
def generate_X_y(df, category, time_span = 1, output = "items"):
    '''
    inputs
    time_span   : number of week in the past the model should use to predict next week
    df          : dataframe containing the dataset
    category    : name of the product category that should be filtered from df
    output      : name of the output feature

    returns
    X           : The input featurs
    y           : The output features
    '''

    # Filter out only the selected product category with product
    input_cols = df[[col for col in df.columns if col.startswith(category)]]
    output_cols = df[[category + '_' + output]]

    # 2. Create input and target vectors
    X = []
    y = []

    # Here y is a shifted version of data_set such that X[0] is week 1 and y[0] is week 2, if time_span = 1
    for i in range(time_span, len(df) - 1): # -1 is to shift the data between train and validation/test data
        X.append(input_cols.iloc[i - time_span:i].values)
        y.append(output_cols.iloc[i + 1].values) # 1 is to select the test as next week

    # make X and y 2D
    X = np.array(X)
    X = X.reshape(X.shape[0], -1)
    y = np.array(y)
    y = y.reshape(y.shape[0], -1)
    
    # Return X and y
    return X, y

# Split the data set in two parts around index
def split_data(df, index):
    return df[:index], df[index:]
  
# Normalizes data per column
def scale_data(train, validate, test):

    # fit scaler to train data
    train_df = pd.DataFrame(train)
    scaler = MinMaxScaler()
    scaler.fit(train_df)

    # transform validation data
    validate_df = scaler.transform(pd.DataFrame(validate))
    
    # transform test data
    test_df = scaler.transform(pd.DataFrame(test))

    return train_df, validate_df, test_df, scaler