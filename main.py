import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
import sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import os
path = os.path.join(os.path.dirname(__file__), "data")
path2 = os.path.join(os.path.dirname(__file__), "processed_data")

# this should be able to work without names= but somehow doesnt
customers = pd.read_csv(os.path.join(path, "olist_customers_dataset.csv"), header=1, names=["customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_city", "customer_state"])
geolocation = pd.read_csv(os.path.join(path, "olist_geolocation_dataset.csv"), header=1, names=["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng", "geolocation_city", "geolocation_state"])
order_items = pd.read_csv(os.path.join(path, "olist_order_items_dataset.csv"), header=1, names=["order_id", "order_item_id", "product_id", "seller_id", "shipping_limit_date", "price","freight_value"])
order_payments = pd.read_csv(os.path.join(path, "olist_order_payments_dataset.csv"), header=1, names=["order_id", "payment_sequential", "payment_type", "payment_installments", "payment_value"])
order_reviews = pd.read_csv(os.path.join(path, "olist_order_reviews_dataset.csv"), header=1, names=["review_id","order_id", "review_score", "review_comment_title", "review_comment_message", "review_creation_date", "review_answer_timestamp"])
orders = pd.read_csv(os.path.join(path, "olist_orders_dataset.csv"), header=1, names=["order_id", "customer_id", "order_status", "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date"])
products = pd.read_csv(os.path.join(path, "olist_products_dataset.csv"), header=1, names=["product_id", "product_category_name", "product_name_lenght", "product_description_lenght", "product_photos_qty", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"])
sellers = pd.read_csv(os.path.join(path, "olist_sellers_dataset.csv"), header=1, names=["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"])
translation = pd.read_csv(os.path.join(path, "product_category_name_translation.csv"), header=1, names=["product_category_name", "product_category_name_english"])

""" count amount of orders per week """
orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
orders["week"] = orders["order_purchase_timestamp"].dt.to_period("W").dt.start_time
orders_per_week = orders.groupby("week").size().reset_index(name="sales")
orders_per_week["week_index"] = (orders_per_week["week"] - orders_per_week["week"].min()).dt.days // 7
orders_per_week = orders_per_week[orders_per_week["sales"] >= 10]
orders_per_week.to_csv(os.path.join(path2, "orders_per_week.csv"), index=False)

""" amount of items per week """
merged = pd.merge(orders[["order_id", "order_purchase_timestamp"]], order_items, on="order_id", how="inner")
merged["week"] = merged["order_purchase_timestamp"].dt.to_period("W").dt.start_time
items_per_week = merged.groupby("week").size().reset_index(name="items_sold")
items_per_week["week_index"] = (items_per_week["week"] - items_per_week["week"].min()).dt.days // 7
items_per_week = items_per_week[items_per_week["items_sold"] >= 10]
items_per_week.to_csv(os.path.join(path2, "items_per_week.csv"), index=False)

""" items sold every 3 days """
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
merged = pd.merge(
    orders[['order_id', 'order_purchase_timestamp']],
    order_items[['order_id']],
    on='order_id'
)
merged.set_index('order_purchase_timestamp', inplace=True)
items_sold = merged.resample('3D').size().reset_index()
items_sold.columns = ['3_day_period_start', 'items_sold']
items_sold = items_sold[items_sold["items_sold"] >= 10]
items_sold.to_csv(os.path.join(path2, "items_sold_every_3days.csv"), index=False)

""" idk """
# orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
# orders["week"] = orders["order_purchase_timestamp"].dt.to_period("W").dt.start_time

# orders_items = pd.merge(orders[["order_id", "week"]], order_items, on="order_id", how="inner")
# weekly_revenue = orders_items.groupby("week").agg({
#     "price": "sum", 
#     "freight_value": "sum",
#     "order_id": "count"
# }).rename(columns={"order_id": "order_count"})

# weekly_revenue["total_revenue"] = weekly_revenue["price"] + weekly_revenue["freight_value"]
# weekly_revenue["average_order_value"] = weekly_revenue["total_revenue"] / weekly_revenue["order_count"]
# unique_customers = orders.groupby("week")["customer_id"].nunique().reset_index(name="unique_customers")

# reviews = pd.merge(orders[["order_id", "week"]], order_reviews[["order_id", "review_score"]], on="order_id", how="inner")
# weekly_reviews = reviews.groupby("week")["review_score"].mean().reset_index(name="avg_review_score")

# weekly_features = weekly_revenue.reset_index().merge(unique_customers, on="week", how="left")
# weekly_features = weekly_features.merge(weekly_reviews, on="week", how="left")

""" order per week with overlapping weeks? (rolling) """
# orders["date"] = orders["order_purchase_timestamp"].dt.floor("d")
# orders_per_day = orders.groupby("date").size().reset_index(name="sales")
# full_date_range = pd.date_range(start=orders_per_day["date"].min(),
#                                 end=orders_per_day["date"].max(),
#                                 freq="D")
# orders_per_day = orders_per_day.set_index("date").reindex(full_date_range, fill_value=0).rename_axis("date").reset_index()
# orders_per_day["rolling_sales"] = orders_per_day["sales"].rolling(window=7).sum()
# orders_per_day["day_index"] = orders_per_day.index
# orders_per_day = orders_per_day.dropna(subset=["rolling_sales"])  # drop data <7 days
# orders_per_day = orders_per_day[orders_per_day["rolling_sales"] >= 10]
# orders_per_day.to_csv(os.path.join(path2, "orders_per_day.csv"), index=False)

""" products prediction """
# df = orders.merge(order_items, on="order_id", how="inner")
# df = df.merge(products, on="product_id", how="inner")
# df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
# df["week"] = df["order_purchase_timestamp"].dt.to_period("W").dt.start_time

# weekly_products = (
#     df.groupby(["product_id", "week"])["order_id"]
#     .nunique()
#     .reset_index(name="sales")
# )

# weekly_products = weekly_products.sort_values(["week", "product_id"]).reset_index(drop=True)

# def create_lags(group, lags=[1, 2, 3]):  # sales from the previous 1–3 weeks
#     for lag in lags:
#         group[f'lag_{lag}'] = group['sales'].shift(lag)
#     group["target"] = group["sales"].shift(-1)
#     return group

# weekly_products = weekly_products.groupby("product_id").apply(create_lags).reset_index(drop=True)
# weekly_products = weekly_products.dropna(subset=["lag_1", "lag_2", "lag_3", "target"])
# weekly_products.to_csv(os.path.join(path2, "weekly_products.csv"), index=False)

# features = ["lag_1", "lag_2", "lag_3"]
# X = weekly_products[features]
# y = weekly_products["target"]

""" train and test data """
X = items_per_week[["week_index"]]
y = items_per_week["items_sold"]

# X = weekly_features[["total_revenue", "order_count", "average_order_value", "unique_customers", "avg_review_score"]]
# y = weekly_features["items_sold"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

def RandomForest(X_train, y_train, X_test):
    regr = RandomForestRegressor(n_estimators=100, random_state=42)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    return y_pred

def GradientBoosting(X_train, y_train, X_test):
    gbr = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=10, random_state=42)
    gbr.fit(X_train, y_train)
    y_pred = gbr.predict(X_test)
    return y_pred

def XGBoost(X_train, y_train, X_test):
    xgbc = xgb.XGBRegressor(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False)
    xgbc.fit(X_train, y_train)
    y_pred = xgbc.predict(X_test)
    return y_pred

def NeuralNetwork(X_train, y_train, X_test):
    mlp = MLPRegressor(
        hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    return y_pred

RF = RandomForest(X_train, y_train, X_test)
GB = GradientBoosting(X_train, y_train, X_test)
XGB = GradientBoosting(X_train, y_train, X_test)
NN = NeuralNetwork(X_train, y_train, X_test)

RF_mse = np.sqrt(mean_squared_error(y_test, RF))
GB_mse = np.sqrt(mean_squared_error(y_test, GB))
XGB_mse = np.sqrt(mean_squared_error(y_test, XGB))
NN_mse = np.sqrt(mean_squared_error(y_test, NN))

# print("Test values:", y_test.astype(int))

print("Random Forest regressor results:", RF.astype(int))
print("Mean squared error:", RF_mse)

print("Gradient Boosting regressor results:", GB.astype(int))
print("Mean squared error:", GB_mse)

print("XGBoosting classifier results:", XGB.astype(int))
print("Mean squared error:", XGB_mse)

print("Neural Network regressor results:", NN.astype(int))
print("Mean squared error:", NN_mse)

y_test_reset = y_test.reset_index(drop=True)
y_pred_reset = pd.Series(XGB).reset_index(drop=True)

plt.figure(figsize=(10, 6))
plt.plot(y_test_reset, label="Real Values", marker="o")
plt.plot(y_pred_reset, label="Predicted Values", marker="x")
plt.xlabel("Test Sample Index")
plt.ylabel("Target Value")
plt.title("Real vs. Predicted Values")
plt.legend()
plt.grid(True)
plt.show()
