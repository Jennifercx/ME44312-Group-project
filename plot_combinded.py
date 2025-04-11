import ipywidgets as widgets
from IPython.display import display, clear_output
import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# Pathing
result_path = os.path.join(os.getcwd(), "results")

# Input parameters
output_name = 'price'
categories = ["bed_bath_table", "health_beauty", "sports_leisure", "furniture_decor", "computers_accessories"]
error_indices = ['mae', 'mse', 'r2']
list_models = ['LTSM', 'RF', 'ARIMA', 'GB']

# Load error metrics
df_LTSM = pd.read_csv(os.path.join(result_path, 'LTSM_error_metrics.csv'))
df_RF = pd.read_csv(os.path.join(result_path, 'RF_error_metrics.csv'))
df_ARIMA = pd.read_csv(os.path.join(result_path, 'ARIMA_error_metrics.csv'))
df_GB = pd.read_csv(os.path.join(result_path, 'GB_error_metrics.csv'))

# === INTERACTIVE WIDGET VISUALIZATION ===
error_index_dropdown = widgets.Dropdown(
    options=error_indices,
    description='Error Index:',
    layout=widgets.Layout(width='50%')
)

output = widgets.Output()

def plot_selected_error_index(change):
    with output:
        clear_output(wait=True)
        
        # Get selected error index
        selected_error_index = error_index_dropdown.value
        
        # These will hold the metrics per model for each category
        error_values_LTSM = []
        error_values_RF = []
        error_values_ARIMA = []
        error_values_GB = []

        # Mapping error index to row index in the CSV
        error_index_map = {
            'mse': 0,  # MSE is in the first row
            'mae': 1,  # MAE is in the second row
            'r2': 2    # R² is in the third row
        }
        
        # Get the correct row index based on the selected error index
        error_row_idx = error_index_map[selected_error_index]

        # Extract error values for each model and category
        for cat_idx in range(len(categories)):
            # For LTSM model
            error_values_LTSM.append(df_LTSM.iloc[error_row_idx, cat_idx])
            # For RF model
            error_values_RF.append(df_RF.iloc[error_row_idx, cat_idx])
            # For ARIMA model
            error_values_ARIMA.append(df_ARIMA.iloc[error_row_idx, cat_idx])
            # For GB model
            error_values_GB.append(df_GB.iloc[error_row_idx, cat_idx])

        # Bar chart visualization
        barWidth = 0.20
        fig = plt.subplots(figsize=(14, 8))

        # Bar positions for the different bars
        br1 = np.arange(len(categories))  # for error values
        br2 = [x + barWidth for x in br1]  # next bar for RF
        br3 = [x + barWidth for x in br2]  # next bar for ARIMA
        br4 = [x + barWidth for x in br3]  # next bar for GB

        # Plot the bars for each model and each category
        plt.bar(br1, error_values_LTSM, color='r', width=barWidth, edgecolor='grey', label='LTSM')
        plt.bar(br2, error_values_RF, color='g', width=barWidth, edgecolor='grey', label='RF')
        plt.bar(br3, error_values_ARIMA, color='b', width=barWidth, edgecolor='grey', label='ARIMA')
        plt.bar(br4, error_values_GB, color='y', width=barWidth, edgecolor='grey', label='GB')

        # Set labels and title
        plt.xlabel('Category', fontweight='bold', fontsize=15)
        plt.ylabel(f'{selected_error_index.upper()} Value', fontweight='bold', fontsize=15)
        plt.xticks([r + barWidth for r in range(len(categories))], categories)  # x-axis labels are categories

        plt.title(f'{selected_error_index.upper()} for All Models Across Categories')
        plt.legend()
        plt.show()

# Link the dropdown to the plot function
error_index_dropdown.observe(plot_selected_error_index, names='value')

# Display the widgets and initial plot
display(error_index_dropdown, output)

# Set default value to trigger initial plot
error_index_dropdown.value = 'mse'  # Default error index is MSE
