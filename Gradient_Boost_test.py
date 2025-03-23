import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure the dataset is sorted
sales_per_week = sales_per_week.sort_values("week")

# Create lag features
def create_lagged_features(data, lags=3):
    df = data.copy()
    for i in range(1, lags+1):
        df[f"lag_{i}"] = df["price"].shift(i)
    return df.dropna()  # Drop NaN rows after shifting

data = create_lagged_features(sales_per_week)

# Train-test split (keep time order!)
X = data.drop(columns=["week", "price"])  # Features
y = data["price"]  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train Random Forest & XGBoost
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# Evaluation function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    return mae, rmse, r2

rf_results = evaluate_model(rf_model, X_test, y_test)
xgb_results = evaluate_model(xgb_model, X_test, y_test)

print(f"Random Forest - MAE: {rf_results[0]:.2f}, RMSE: {rf_results[1]:.2f}, R²: {rf_results[2]:.2f}")
print(f"XGBoost - MAE: {xgb_results[0]:.2f}, RMSE: {xgb_results[1]:.2f}, R²: {xgb_results[2]:.2f}")

# Grid Search for Random Forest
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3,
                           scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters for Random Forest:", grid_search.best_params_)

# Predict Next Week's Sales
latest_features = X_train.iloc[-1:].values  # Use last known training row
next_week_sales_rf = rf_model.predict(latest_features)
next_week_sales_xgb = xgb_model.predict(latest_features)

print(f"Next week's sales prediction (Random Forest): {next_week_sales_rf[0]:.2f}")
print(f"Next week's sales prediction (XGBoost): {next_week_sales_xgb[0]:.2f}")
