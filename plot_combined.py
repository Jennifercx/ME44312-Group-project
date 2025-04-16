# - plot combined stuff ------------------------------------------------------------------------
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colormaps
model_name = 'SARIMAX'


cmap = plt.get_cmap('viridis')  # Get the Viridis colormap
colors = [cmap(i / (4)) for i in range(5)]  # Normalize between 0 and 1

# Example usage:
# Define file paths and load your error metrics CSV files
path = os.path.join(os.getcwd(), "figures/" + model_name + "_combined")
result_path = os.path.join(os.getcwd(), "results")
df_true = pd.read_csv(os.path.join(result_path, model_name + '_y_true.csv'))
df_pred = pd.read_csv(os.path.join(result_path, model_name + '_y_pred.csv'))

# List of categories (assumes order corresponds to dataframe columns)
categories = ["bed_bath_table", "health_beauty", "sports_leisure", "furniture_decor", "computers_accessories"]

plt.figure(figsize=(14, 8))
i = 0
for cat in categories:
    plt.plot(df_true[cat], color=colors[i], label=f'Actual {cat}')
    plt.plot(df_pred[cat], color=colors[i], linestyle='--', label=f'Predicted {cat}')
    i += 1

plt.xlabel('Week')
plt.ylabel('Price')
plt.title('Actual vs. predicted prices by category for ' + model_name + " model")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(path)
plt.show()