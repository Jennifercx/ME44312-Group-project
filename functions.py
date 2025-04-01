# --- import packages -------------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import clear_output
import pandas as pd
# --- functions -------------------------------------------------------------------------------------------------

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
    price_cols = [col for col in df.columns if 'items' in col] # 'price' 'items'
    for i in range(len(df) - time_span ):
        X.append(df.iloc[i:i + time_span].drop(columns=['week']).values)
        y.append(df.iloc[i + time_span][price_cols].values)
    return np.array(X), np.array(y)

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

def scale_per_column_data(X, y):
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Step 2: Fit the scaler on training data only
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # X_train = pd.DataFrame(X_train_reshaped)
    # X_val_reshaped = pd.DataFrame(X_val_reshaped)
    # y_train_reshaped = pd.DataFrame(y_train_reshaped)
    # y_val_reshaped = pd.DataFrame(y_val_reshaped)

    # step 3: fit per column
    X_train_scaled = pd.DataFrame(scaler_X.fit_transform(X_train), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler_X.fit_transform(X_val), columns=X_val.columns)
    y_train_scaled = pd.DataFrame(scaler_y.fit_transform(y_train), columns=y_train.columns)
    y_val_scaled = pd.DataFrame(scaler_y.fit_transform(y_val), columns=y_val.columns)

    return X_train_scaled, X_val_scaled, scaler_X, y_train_scaled, y_val_scaled, scaler_y

# Evaluates the model for the selected parameters
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, R^2: {r2:.2f}")
    return 0

# Plots the weekly sales per product category
class ModelEvaluation:

    def __init__(self, y_true, y_pred, name="Model"):
        self.name = name
        self.y_true = y_true
        self.y_pred = y_pred
        self.ResultsShape = y_true.shape
        self.categories = [f"Category {i+1}" for i in range(self.ResultsShape[1])]
    
    def EvaluateResults(self):
    
        error = self.y_pred - self.y_true
        mse = (error ** 2).mean()
        rmse = mse ** 0.5
        mae = abs(error).mean()   
        return mse, rmse, mae
    
    def set_categories(self, categories):
        self.categories = categories
    
    def plot_category(self, cat):

        y_true = self.y_true
        y_pred = self.y_pred
        clear_output(wait=True)
        cat_index = self.categories.index(cat)
        weeks = np.arange(1, y_true.shape[0] + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(weeks, y_true[:, cat_index], marker='o', label='y_true')
        plt.plot(weeks, y_pred[:, cat_index], marker='x', label='y_pred')
        plt.title(f"Results for {cat}")
        plt.xlabel("Week")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_categories(self):
        # Create a dropdown widget for selecting categories
        dropdown = widgets.Dropdown(options=self.categories, description='Category:')
        widgets.interact(self.plot_category, cat=dropdown)

# Predicts and plots the sales per week per product category
def predict_data(model, X_test, y_test, scaler_y):
    y_pred = scaler_y.inverse_transform(model.predict(X_test))  # Reverse transformation
    y_pred = y_pred[:-1]                 # Shift data since we predict the next week based on the current week
    y_true = scaler_y.inverse_transform(y_test)  # Reverse transformation for actual values
    y_true = y_true[1:]

    # plt.figure(figsize=(12, 5))
    # plt.plot(y_true, label="Actual sales", marker="o")
    # plt.plot(y_pred, label="Predicted sales", marker="x")
    # plt.legend()
    # plt.title("Predicted vs. Actual Sales")
    # plt.xlabel("Time (weeks)")
    # plt.ylabel("Sales Price")
    # plt.show()
    return y_true, y_pred