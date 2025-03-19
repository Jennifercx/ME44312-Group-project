import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Custom function to calculate MAPE.
    Be cautious if y_true can be zero or very small.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# =====================
# 1. LOAD THE DATA
# =====================
df = pd.read_csv(
    'processed_data/items_sold_every_3days.csv', 
    parse_dates=['3_day_period_start']   # Adjust if your date column name is different
)

df.sort_values('3_day_period_start', inplace=True)
df.set_index('3_day_period_start', inplace=True)  # optional but helpful for time-based features

# =====================
# 2. SHIFT THE TARGET FOR NEXT-WEEK FORECAST
# =====================
# We create a new column that represents next week's sales (the forecast target).
df['items_next_3days'] = df['items_sold'].shift(-1)

# We'll drop the last row because it doesn't have a "next week" value
df.dropna(inplace=True)

# =====================
# 3. FEATURE ENGINEERING
# =====================

# (a) Lag Features
max_lag = 2  # 2-3 seems to work best but still not good
for lag in range(1, max_lag + 1):
    df[f'lag_{lag}'] = df['items_sold'].shift(lag)

# Because we created lag features, the first `max_lag` rows become NaN. We drop them.
df.dropna(inplace=True)

# (b) Rolling Window Features
df['rolling_mean_3'] = df['items_sold'].rolling(window=3).mean()
df['rolling_std_3'] = df['items_sold'].rolling(window=3).std()
df.dropna(inplace=True)  # Drop rows with NaNs due to rolling

# (c) Calendar Features (Seasonality)
df['month'] = df.index.month
df['week_of_year'] = df.index.isocalendar().week.astype(int)  # from Pandas 1.1 onwards

# =====================
# 4. TRAIN-TEST SPLIT
# =====================
# We'll use 80% of data for training, 20% for testing
split_point = int(len(df) * 0.8)
train = df.iloc[:split_point]
test = df.iloc[split_point:]

# Our target is now the "items_next_3days"
target = 'items_next_3days'

# Collect all feature columns
# We consider lag_*, rolling_*, and the calendar features
features = [
    col for col in df.columns 
    if col.startswith('lag_') 
       or col.startswith('rolling_') 
       or col in ['month', 'week_of_year']
]

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

# =====================
# 5. TIME SERIES CROSS-VALIDATION
# =====================
tscv = TimeSeriesSplit(n_splits=3)

# =====================
# 6. EXTENDED GRID SEARCH
# =====================
param_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 1.0],
    # Optionally, you can also tune min_samples_leaf or other params:
    # 'min_samples_leaf': [1, 2, 5],
}

gbr = GradientBoostingRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=gbr,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=tscv,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("\n=== Grid Search Results ===")
print("Best Params:", best_params)
print("Best CV Score (MSE):", -grid_search.best_score_)

# =====================
# 7. FINAL EVALUATION ON TEST SET
# =====================
test_preds = best_model.predict(X_test)

test_mse = mean_squared_error(y_test, test_preds)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, test_preds)
test_r2 = r2_score(y_test, test_preds)
test_mape = mean_absolute_percentage_error(y_test, test_preds)

print("\n=== Final Model Performance on Test Set ===")
print(f"MSE:  {test_mse:.2f}")
print(f"RMSE: {test_rmse:.2f}")
print(f"MAE:  {test_mae:.2f}")
print(f"R^2:  {test_r2:.3f}")
print(f"MAPE: {test_mape:.2f}%")

# =====================
# 8. PLOT ACTUAL VS. PREDICTED
# =====================
plt.figure(figsize=(10, 6))
plt.plot(test.index, y_test, label='Actual Next-Week Sales', marker='o')
plt.plot(test.index, test_preds, label='Predicted Next-Week Sales', marker='x')
plt.xlabel('Date (every 3 days)')
plt.ylabel('Items Sold Next Week')
plt.title('Actual vs Predicted Item-Sales (Next Week)')
plt.legend()
plt.show()
