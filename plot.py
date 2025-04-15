# imports
import ipywidgets as widgets
from IPython.display import display, clear_output
import os
from matplotlib import pyplot as plt
import pandas as pd

# pathing
result_path = os.path.join(os.getcwd(), "results")

# input parameters
output_name = 'price'
categories = ["bed_bath_table", "health_beauty", "sports_leisure", "furniture_decor", "computers_accessories"]
model = 'LSTM'

# plot parameters
y_pred = pd.read_csv(os.path.join(result_path, model + '_y_pred.csv'))
y_true = pd.read_csv(os.path.join(result_path, model + '_y_true.csv'))
error_metrics = pd.read_csv(os.path.join(result_path, model + '_error_metrics.csv'))

# === INTERACTIVE WIDGET VISUALIZATION ===
category_dropdown = widgets.Dropdown(
    options=categories,
    description='Category:',
    layout=widgets.Layout(width='50%')
)

output = widgets.Output()

def plot_selected_category(change):
    with output:
        clear_output(wait=True)
        cat = change['new']

        plt.figure(figsize=(12, 4))

        # Plot 1: Training and validation loss
        if model == 'LTSM':
            plt.subplot(1, 2, 1)
            df_history = pd.read_csv(os.path.join(result_path, model + '_history_' + cat + '.csv'))
            plt.plot(df_history['loss'], label='Train Loss')
            plt.plot(df_history['val_loss'], label='Val Loss')
            plt.title(f'{cat.title()} - Training vs Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            # Plot 2: Prediction vs Actual
            plt.subplot(1, 2, 2)
            plt.plot(y_true[cat], label='Actual', linewidth=2)
            plt.plot(y_pred[cat], label='Predicted', linestyle='--')
            plt.title(f'{cat.title()} - Predicted vs Actual Price')
            plt.xlabel('Week')
            plt.ylabel('Price')
            plt.legend()

            plt.tight_layout()
            plt.show()
        else:
            plt.plot(y_true[cat], label='Actual', linewidth=2)
            plt.plot(y_pred[cat], label='Predicted', linestyle='--')
            plt.title(f'{cat.title()} - Predicted vs Actual Price for ' + model)
            plt.xlabel('Week')
            plt.ylabel('Price')
            plt.legend()
            
            plt.tight_layout()
            plt.show()

category_dropdown.observe(plot_selected_category, names='value')
display(category_dropdown, output)
category_dropdown.value = categories[0]  # trigger initial plot

