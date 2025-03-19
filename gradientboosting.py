import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import matplotlib.pyplot as plt

# Load the dataset; make sure to parse dates properly
df = pd.read_csv('processed_data/items_sold_every_3days.csv', parse_dates=['3_day_period_start'])
df.sort_values('3_day_period_start', inplace=True)  # ensure data is in chronological order

# Optionally, set the date as index for easier plotting
df.set_index('3_day_period_start', inplace=True)

# Create lag features to incorporate past sales into predictions
# Here we create lags for the previous 3 weeks
for lag in [1, 2, 3]:
    df[f'lag_{lag}'] = df['items_sold'].shift(lag)

# Remove rows with missing values (due to lag creation)
df.dropna(inplace=True)

# Split the dataset into training and testing sets based on time
split_point = int(len(df) * 0.8)
train = df.iloc[:split_point]
test = df.iloc[split_point:]

# Define features and target variable
features = [col for col in df.columns if col.startswith('lag_')]
target = 'items_sold'

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

# Initialize and train the gradient boosting model
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}

tscv = TimeSeriesSplit(n_splits=3)

model = GradientBoostingRegressor(random_state=42)

grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = GradientBoostingRegressor(**best_params, random_state=42)
best_model.fit(X_train, y_train)

# Make predictions on the test set
preds = best_model.predict(X_test)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_test, preds)
print("Test MSE:", mse)

# Plot the actual vs predicted sales
plt.figure(figsize=(10, 6))
plt.plot(test.index, y_test, label='Actual Sales', marker='o')
plt.plot(test.index, preds, label='Predicted Sales', marker='x')
plt.xlabel('Date (per week)')
plt.ylabel('Sales')
plt.title('Actual vs Predicted Item-Sales')
plt.legend()
plt.show()

print("Best Params:", grid_search.best_params_)
print("Best Score (MSE):", -grid_search.best_score_)

best_model = grid_search.best_estimator_
