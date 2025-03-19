import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

import os
path = os.path.join(os.path.dirname(__file__), "processed_data")

df = pd.read_csv(os.path.join(path, "arima_prediction.csv"), parse_dates=['week'])
df = df.sort_values('week')

positions = ['first', 'second', 'third', 'fourth', 'fifth']

evaluation_metrics = {}

for pos in positions:
    pred_col = f"{pos}_pred_amount"
    true_col = f"{pos}_true_amount"
    
    valid = df[[pred_col, true_col]].dropna()
    
    if valid.empty:
        print(f"No valid data for ranking position: {pos}")
        continue
    
    mae = mean_absolute_error(valid[true_col], valid[pred_col])
    rmse = np.sqrt(mean_squared_error(valid[true_col], valid[pred_col]))
    
    evaluation_metrics[pos] = {'MAE': mae, 'RMSE': rmse}
    print(f"{pos.capitalize()} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

fig, axs = plt.subplots(len(positions), 1, figsize=(10, 2 * len(positions)), sharex=True)
for i, pos in enumerate(positions):
    pred_col = f"{pos}_pred_amount"
    true_col = f"{pos}_true_amount"
    
    axs[i].plot(df['week'], df[true_col], label='True', marker='o')
    axs[i].plot(df['week'], df[pred_col], label='Predicted', marker='x')
    axs[i].set_title(f"{pos.capitalize()} Ranking")
    axs[i].legend()
    axs[i].set_ylabel("Sales Amount")
    
plt.xlabel("Week")
plt.tight_layout()
plt.show()
