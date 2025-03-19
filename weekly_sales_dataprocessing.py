import pandas as pd

import os
path = os.path.join(os.path.dirname(__file__), "data")
path2 = os.path.join(os.path.dirname(__file__), "processed_data")

# this should be able to work without names= but somehow doesnt
order_items = pd.read_csv(os.path.join(path, "olist_order_items_dataset.csv"), header=0)
order_reviews = pd.read_csv(os.path.join(path, "olist_order_reviews_dataset.csv"), header=0)
orders = pd.read_csv(os.path.join(path, "olist_orders_dataset.csv"), header=0)
products = pd.read_csv(os.path.join(path, "olist_products_dataset.csv"), header=0)
translation = pd.read_csv(os.path.join(path, "product_category_name_translation.csv"), header=0)

""" dataprocessing """
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
orders['week'] = orders['order_purchase_timestamp'].dt.to_period('W').apply(lambda r: r.start_time)

orders_items = pd.merge(orders[['order_id', 'week']], order_items, on='order_id')

orders_items_products = pd.merge(orders_items, products, on='product_id')
orders_items_products = pd.merge(orders_items_products, translation, on='product_category_name', how='left')

weekly_totals = orders_items_products.groupby('week').agg(
    total_items_sold=('order_item_id', 'count'),
    total_price=('price', 'sum')
).reset_index()

category_weekly = orders_items_products.groupby(['week', 'product_category_name_english']).agg(
    items_sold=('order_item_id', 'count'),
    price=('price', 'sum')
).reset_index()

items_sold_pivot = category_weekly.pivot(index='week', columns='product_category_name_english', values='items_sold')
items_sold_pivot = items_sold_pivot.rename(columns=lambda x: f"items_sold_{x}")

price_pivot = category_weekly.pivot(index='week', columns='product_category_name_english', values='price')
price_pivot = price_pivot.rename(columns=lambda x: f"price_{x}")

most_items = category_weekly.loc[category_weekly.groupby('week')['items_sold'].idxmax()].set_index('week')
most_items = most_items['product_category_name_english'].rename('most_items_category')

most_price = category_weekly.loc[category_weekly.groupby('week')['price'].idxmax()].set_index('week')
most_price = most_price['product_category_name_english'].rename('most_price_category')

final_dataset = weekly_totals.set_index('week').join(items_sold_pivot).join(price_pivot).join(most_items).join(most_price).reset_index()

final_dataset.fillna(0, inplace=True)
final_dataset = final_dataset[final_dataset["total_items_sold"] >= 10]

final_dataset.to_csv(os.path.join(path2, "weekly_sales_per_category.csv"), index=False)

print(final_dataset)
