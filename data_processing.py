# --- import packages -------------------------------------------------------------------------------------------------
import pandas as pd
import os

# data directory
data_dir = os.getcwd()+"\\data\\"

# import dataframes seperately
df_customer = pd.read_csv(data_dir+"olist_customers_dataset.csv")
df_geolocation = pd.read_csv(data_dir+"olist_geolocation_dataset.csv")
df_order_items = pd.read_csv(data_dir+"olist_order_items_dataset.csv")
df_payments = pd.read_csv(data_dir+"olist_order_payments_dataset.csv")
df_reviews = pd.read_csv(data_dir+"olist_order_reviews_dataset.csv")
df_orders = pd.read_csv(data_dir+"olist_orders_dataset.csv")
df_products = pd.read_csv(data_dir+"olist_products_dataset.csv")
df_sellers = pd.read_csv(data_dir+"olist_sellers_dataset.csv")
df_product_category_name = pd.read_csv(data_dir+"product_category_name_translation.csv")

# combine all datasets and save to .csv file
df_all = df_customer.merge(df_orders, on='customer_id', how='left').merge(df_reviews, on='order_id', how='left').merge(df_payments, on='order_id', how='left').merge(df_order_items, on='order_id', how='left').merge(df_sellers, on='seller_id', how='left').merge(df_products, on='product_id', how='left').merge(df_product_category_name, on='product_category_name', how='left')
df_shipping_time = pd.DataFrame((pd.to_datetime(df_orders["order_delivered_customer_date"]) -pd.to_datetime(df_orders["order_purchase_timestamp"])).dt.total_seconds()/(24*3600))
df_all["shipping_time"] = pd.DataFrame((pd.to_datetime(df_orders["order_delivered_customer_date"]) -pd.to_datetime(df_orders["order_purchase_timestamp"])).dt.total_seconds()/(24*3600))
df_all.to_csv('data/merged_data.csv', index=False)

# --- create weekly dataset -------------------------------------------------------------------------------------------------------

# select relevant columns only
df_data = df_all[["order_purchase_timestamp", "review_score", "price", "freight_value", "shipping_time", "product_category_name_english"]]
df_data.dropna(inplace=True)
df_data["week"] = pd.to_datetime(df_data['order_purchase_timestamp']).dt.to_period("W")

# per week and per item type dataframes
item = df_data.groupby(['week', 'product_category_name_english']).size().unstack(fill_value=0)
total_items = df_data.groupby('week').size()
item = item.div(total_items, axis=0)

price = df_data.groupby(['week', 'product_category_name_english'])['price'].sum().unstack(fill_value=0)
total_price = df_data.groupby('week')['price'].sum()
price = price.div(total_price, axis=0)

freight_value = df_data.groupby(['week', 'product_category_name_english'])['freight_value'].sum().unstack(fill_value=0)
total_freight_value = df_data.groupby('week')['price'].sum()
freight_value = freight_value.div(total_freight_value, axis=0)

review_score = df_data.groupby(['week', 'product_category_name_english'])['review_score'].mean().unstack(fill_value=0)

shipping_time = df_data.groupby(['week', 'product_category_name_english'])['shipping_time'].mean().unstack(fill_value=0)

# rename all columns
item.columns = [col + "_items" for col in item.columns]
price.columns = [col + "_price" for col in price.columns]
freight_value.columns = [col + "_freight" for col in freight_value.columns]
review_score.columns = [col + "_review" for col in review_score.columns]
shipping_time.columns = [col + "_shipping" for col in shipping_time.columns]

# turn week into a columns
item = item.reset_index()
price = price.reset_index()
freight_value = freight_value.reset_index()
review_score = review_score.reset_index()
shipping_time = shipping_time.reset_index()

# Merging all dataframes on 'week'
findal_df = (
    item
    .merge(price, on="week", how="left")
    .merge(freight_value, on="week", how="left")
    .merge(review_score, on="week", how="left")
    .merge(shipping_time, on="week", how="left")
)

print(findal_df.head(30))