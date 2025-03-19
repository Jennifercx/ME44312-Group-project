import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import os

path = os.path.join(os.path.dirname(__file__), "processed_data")
data = pd.read_csv(os.path.join(path, "weekly_sales_per_category.csv"), parse_dates=['week'])
data = data.sort_values('week')

sales_columns = [col for col in data.columns if col.startswith('items_sold_')]

forecasts_dict = {}
true_values_dict = {}
training_sum = {}

for col in sales_columns:
    ts = data.set_index('week')[col]
    ts = ts.asfreq('W-MON')
    ts = ts.fillna(0)
    
    n = len(ts)
    train_size = int(n * 0.8)  # only use first 80% to predict
    train, test = ts[:train_size], ts[train_size:]
    
    training_sum[col] = train.sum()
    
    try:
        model = ARIMA(train, order=(1, 1, 2))  # can change this
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(test))  # forecast next 20%
        forecasts_dict[col] = forecast
        true_values_dict[col] = test
    except Exception as e:
        print(f"ARIMA model failed for {col}: {e}")
        forecasts_dict[col] = None
        true_values_dict[col] = None

top5_categories = sorted(training_sum, key=training_sum.get, reverse=True)[:5]
print("Top 5 categories (by training sales sum):")
for i, cat in enumerate(top5_categories):
    print(f"{i + 1}. {cat}")  # could improve by removing items_sold_ and _ from string

test_weeks = None
for cat in top5_categories:
    if forecasts_dict[cat] is not None:
        test_weeks = forecasts_dict[cat].index
        break

if test_weeks is None:
    raise ValueError("No forecasts available for top categories.")

result_rows = []

for week in test_weeks:
    week_data = []
    for cat in top5_categories:
        pred = forecasts_dict[cat].loc[week] if forecasts_dict[cat] is not None else np.nan
        true = true_values_dict[cat].loc[week] if true_values_dict[cat] is not None else np.nan
        week_data.append({'category': cat, 'pred': pred, 'true': true})
    
    week_data_sorted = sorted(week_data, key=lambda x: x['pred'], reverse=True)
    
    row = {'week': week}
    for i, item in enumerate(week_data_sorted, start=1):
        row[f"{['first','second','third','fourth','fifth'][i-1]}_sold_category"] = item['category']
        row[f"{['first','second','third','fourth','fifth'][i-1]}_pred_amount"] = item['pred']
        row[f"{['first','second','third','fourth','fifth'][i-1]}_true_amount"] = item['true']
    
    result_rows.append(row)

result_df = pd.DataFrame(result_rows)
result_df = result_df.sort_values('week')
result_df.to_csv(os.path.join(path, "arima_prediction.csv"), index=False)



# import pmdarima as pm

# # Suppose 'train' is your training time series
# model = pm.auto_arima(
#     train, 
#     start_p=0, max_p=5,
#     start_q=0, max_q=5,
#     d=None,  # let auto_arima find the differencing order
#     seasonal=False,  # set to True if you suspect seasonality
#     trace=True,
#     error_action='ignore',  
#     suppress_warnings=True,
#     stepwise=True
# )

# print(model.summary())

# # Once the best model is found, forecast
# forecast = model.predict(n_periods=len(test))
