import os
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output
from matplotlib import pyplot as plt

def interactive_model_plot(model_dfs, categories, error_indices):
    """
    model_dfs: a dict mapping model name (e.g., 'LTSM') to its error metrics dataframe.
               Each dataframe should have error metrics for each category arranged row-wise,
               e.g., row0 = MSE, row1 = MAE, row2 = R².
    categories: list of category names (order corresponds to columns in the model dfs).
    error_indices: list of error indices (e.g., ['mae', 'mse', 'r2']).
    """
    # Create interactive widgets
    error_index_dropdown = widgets.Dropdown(
        options=error_indices,
        description='Error Index:',
        layout=widgets.Layout(width='50%')
    )
    
    model_select = widgets.SelectMultiple(
        options=list(model_dfs.keys()),
        value=list(model_dfs.keys()),  # default: all models selected
        description='Models:',
        layout=widgets.Layout(width='50%', height='150px')
    )
    
    output = widgets.Output()

    # Mapping from error index to row index in the dfs
    error_index_map = {
        'mse': 0,  # first row
        'mae': 1,  # second row
        'r2': 2    # third row
    }
    
    def plot_func(change):
        with output:
            clear_output(wait=True)
            selected_error = error_index_dropdown.value
            selected_models = model_select.value
            if not selected_models:
                print("Please select at least one model.")
                return
            
            error_row_idx = error_index_map[selected_error]
            # Prepare error values for each selected model and each category
            model_errors = {}  # dictionary: model -> list of error values for each category
            for m in selected_models:
                df = model_dfs[m]
                # Assume each column corresponds to a category and we extract the correct row.
                vals = [df.iloc[error_row_idx, cat_idx] for cat_idx in range(len(categories))]
                model_errors[m] = vals
            
            # Plotting
            num_models = len(selected_models)
            barWidth = 0.20
            base = np.arange(len(categories))
            
            plt.figure(figsize=(14, 8))
            # Loop over selected models and plot their bars with adjusted positions
            for i, m in enumerate(selected_models):
                pos = base + i * barWidth
                # Optionally assign a color based on the model name
                color = None
                if m == 'LTSM': color = 'r'
                elif m == 'RF': color = 'g'
                elif m == 'ARIMA': color = 'b'
                elif m == 'GB': color = 'y'
                plt.bar(pos, model_errors[m], width=barWidth, edgecolor='grey', label=m, color=color)
            
            # Set x-axis ticks and labels at the center of the grouped bars
            plt.xlabel('Category', fontweight='bold', fontsize=15)
            plt.ylabel(f'{selected_error.upper()} Value', fontweight='bold', fontsize=15)
            plt.xticks(base + barWidth*(num_models-1)/2, categories)
            plt.title(f'{selected_error.upper()} for Selected Models Across Categories')
            plt.legend()
            plt.show()

    # Observe changes in both widgets
    error_index_dropdown.observe(plot_func, names='value')
    model_select.observe(plot_func, names='value')

    # Display the complete interactive UI
    display(widgets.VBox([error_index_dropdown, model_select, output]))
    
    # Trigger an initial plot
    error_index_dropdown.value = error_indices[0]


# Example usage:
# Define file paths and load your error metrics CSV files
result_path = os.path.join(os.getcwd(), "results")
df_LTSM = pd.read_csv(os.path.join(result_path, 'LTSM_error_metrics.csv'))
df_RF = pd.read_csv(os.path.join(result_path, 'RF_error_metrics.csv'))
df_ARIMA = pd.read_csv(os.path.join(result_path, 'ARIMA_error_metrics.csv'))
df_GB = pd.read_csv(os.path.join(result_path, 'GB_error_metrics.csv'))


# List of categories (assumes order corresponds to dataframe columns)
categories = ["bed_bath_table", "health_beauty", "sports_leisure", "furniture_decor", "computers_accessories"]

# List of error indices to choose from
error_indices = ['mae', 'mse', 'r2']

# Define the dictionary mapping model names to their dataframes
# Note: Uncomment the models you want to include in the plot
model_dfs = {
    'LTSM': df_LTSM,
    #'RF': df_RF,
    'ARIMA': df_ARIMA,
    #'GB': df_GB
}

# Call the function to display the interactive plot
interactive_model_plot(model_dfs, categories, error_indices)