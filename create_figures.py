import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colormaps

# Choose m
selected_models = [ 'LSTM', 'ARIMA']
selected_error = 'r2'
fig_name = 'r2.png'
path = os.path.join(os.getcwd(), "figures/" + fig_name)

cmap = plt.get_cmap('viridis')  # Get the Viridis colormap
colors = [cmap(i / (len(selected_models))) for i in range(len(selected_models))]  # Normalize between 0 and 1

# Example usage:
# Define file paths and load your error metrics CSV files
result_path = os.path.join(os.getcwd(), "results")
df_LSTM = pd.read_csv(os.path.join(result_path, 'LSTM_error_metrics.csv'))
df_RF = pd.read_csv(os.path.join(result_path, 'RF_error_metrics.csv'))
df_ARIMA = pd.read_csv(os.path.join(result_path, 'ARIMA_error_metrics.csv'))
df_GB = pd.read_csv(os.path.join(result_path, 'GB_error_metrics.csv'))


# List of categories (assumes order corresponds to dataframe columns)
categories = ["bed_bath_table", "health_beauty", "sports_leisure", "furniture_decor", "computers_accessories"]

# List of error indices to choose from
error_indices = ['mae', 'mse', 'r2']

error_index_map = {
'mse': 0,  # first row
'mae': 1,  # second row
'r2': 2    # third row
}

# Define the dictionary mapping model names to their dataframes
# Note: Uncomment the models you want to include in the plot
model_dfs = {
    'LSTM': df_LSTM,
    'RF': df_RF,
    'ARIMA': df_ARIMA,
    #'GB': df_GB
}

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
    plt.bar(pos, model_errors[m], width=barWidth, edgecolor='grey', label=m, color=colors[i])

# Set x-axis ticks and labels at the center of the grouped bars
plt.xlabel('Category', fontweight='bold', fontsize=15)
plt.ylabel(f'{selected_error.upper()} Value', fontweight='bold', fontsize=15)
plt.xticks(base + barWidth*(num_models-1)/2, categories)
plt.title(f'{selected_error.upper()} for Selected Models Across Categories')
plt.legend()
plt.savefig(path)
plt.show()

