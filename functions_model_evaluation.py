# --- import packages -------------------------------------------------------------------------------------------------
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import clear_output

# --- functions -------------------------------------------------------------------------------------------------------

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
        print(cat, cat_index)
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

# Visualise output
def plot_widget(y_true, y_pred, name = 'Widget'):
    Widget = ModelEvaluation(y_true, y_pred, name)
    categories = ["bed_bath_table", "health_beauty", "sports_leisure", "furniture_decor", "computers_accessories"]
    # categories = ["automotive", "baby", "beauty_health","electronics", "entertainment", "fashion", "food", "furniture", "home", "miscellaneous", "office_supplies", "pets", "sports", "tools", "toys"]
    Widget.set_categories(categories)
    Widget.plot_categories()
    print(Widget.EvaluateResults())

def validate_model(model, X_validate_scaled, y_test, scaler_y):

    # 1. predict data
    y_pred = model.predict(X_validate_scaled)
    y_pred = scaler_y.inverse_transform(np.array(y_pred).reshape(-1,1)) # Reverse transformation

    # 2. Evaluate model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return y_pred, y_test, mae, mse, r2

    # === INTERACTIVE WIDGET VISUALIZATION ===

def plot_model(categories):
    category_dropdown = widgets.Dropdown(
        options=categories,
        description='Category:',
        layout=widgets.Layout(width='50%')
    )

    output = widgets.Output()