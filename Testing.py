import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import clear_output




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