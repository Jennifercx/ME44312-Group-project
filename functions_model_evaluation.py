# --- import packages -------------------------------------------------------------------------------------------------
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import clear_output

# --- functions -------------------------------------------------------------------------------------------------------

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

# visualise output
def plot_widget(y_true, y_pred, name = 'Widget'):
    Widget = ModelEvaluation(y_true, y_pred, name)
    categories = ["automotive", "baby", "beauty_health", "construction_tools", "electronics", "entertainment", "fashion", "food", "furniture", "garden_tools", "gifts", "home_appliances", "housewares", "luggage", "office_supplies", "other", "pets", "sports", "telephony", "toys"]
    Widget.set_categories(categories)
    Widget.plot_categories()
    print(Widget.EvaluateResults())

def validate_model(model, X_val_scaled, y_val_scaled, scaler_y):

    # 1. predict data
    y_true, y_pred = predict_data(model, X_val_scaled, y_val_scaled, scaler_y)

    # 2. Evaluate model
    evaluate_model(y_true, y_pred)

    # 3. visualise results
    plot_widget(y_true, y_pred)
