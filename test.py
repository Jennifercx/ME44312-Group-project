import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
path = os.path.join(os.path.dirname(__file__), "data")

# fix mistake review_id to order_id is 1 to many

# this should be able to work without names= but somehow doesnt
customers = pd.read_csv(os.path.join(path, "olist_customers_dataset.csv"), header=1, names=['customer_id', 'customer_unique_id', 'customer_zip_code_prefix', 'customer_city', 'customer_state'])
orders = pd.read_csv(os.path.join(path, "olist_orders_dataset.csv"), header=1, names=["order_id", "customer_id", "order_status", "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date", "order_estimated_delivery_date"])
order_items = pd.read_csv(os.path.join(path, "olist_order_items_dataset.csv"), header=1, names=["order_id", "order_item_id", "product_id", "seller_id", "shipping_limit_date", "price","freight_value"])
products = pd.read_csv(os.path.join(path, "olist_products_dataset.csv"), header=1, names=["product_id", "product_category_name", "product_name_lenght", "product_description_lenght", "product_photos_qty", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"])
order_reviews = pd.read_csv(os.path.join(path, "olist_order_reviews_dataset.csv"), header=1, names=["review_id","order_id", "review_score", "review_comment_title", "review_comment_message", "review_creation_date", "review_answer_timestamp"])
translation = pd.read_csv(os.path.join(path, "product_category_name_translation.csv"), header=1, names=["product_category_name", "product_category_name_english"])

""" count how many people have order once or multiple times 
conclusion: only unique customer_ids so no customer has more than 1 order
"""
# order_counts = orders.groupby('customer_id').size()
# order_freq = order_counts.value_counts().sort_index()

# total_orders = orders['customer_id'].count()
# unique_customers = orders['customer_id'].nunique()
# print(total_orders, unique_customers)

# plt.figure(figsize=(10, 6))
# order_freq.plot(kind='bar')

# plt.xlabel('Number of Orders')
# plt.ylabel('Number of Customers')
# plt.title('Amount of orders per customer')
# plt.xticks(rotation=0)

# plt.show()

""" count amount of items per order """
# items_per_order = order_items.groupby("order_id").size()

# plt.figure(figsize=(10, 6))
# n, bins, patches = plt.hist(items_per_order, bins=range(1, items_per_order.max() + 2), edgecolor='black')

# plt.title("Amount of items per order")
# plt.xlabel("Amount of items")
# plt.ylabel("Frequency (Amount of items)")

# bin_center = [patch.get_x() + patch.get_width() / 2 for patch in patches]
# plt.xticks(bin_center, range(1, len(bin_center) + 1))

# for i in range(len(patches)):
#     height = patches[i].get_height()
#     plt.text(bin_center[i], height + 1, f'{int(height)}', ha='center', va='bottom', fontsize=10)

# plt.show()

""" count the amount of times a product has been bought """
# product_count = order_items['product_id'].value_counts()
# top_products = product_count.head(10)

# plt.figure(figsize=(12, 6))
# top_products.plot(kind='bar', color='skyblue', edgecolor='black')

# plt.title('Top 10 most bought products')
# plt.xlabel('Product ID')
# plt.ylabel('Number of times bought')
# plt.xticks(rotation=45, ha='right')

# plt.show()

""" number of times a category has been bought """
# products_category = pd.merge(products, translation, on='product_category_name', how='left')
# merged_df = pd.merge(order_items, products_category[['product_id', 'product_category_name_english']], on='product_id', how='left')

# category_count = merged_df['product_category_name_english'].value_counts()
# # top_10_categories = category_count.head(10)

# plt.figure(figsize=(12, 6))
# category_count.plot(kind='bar', color='skyblue', edgecolor='black')
# plt.title('Number of times bought per category')
# plt.xlabel('Product category')
# plt.ylabel('Number of times bought')
# plt.xticks(rotation=90, ha='right')
# plt.tight_layout()  # to make sure I can read the labels but it is ugly
# plt.show()

""" check if one review_id in reviews is linked to multiple order_id 
i dont know what to do with this information tbh"""
# order_review_count = order_reviews.groupby('review_id')['order_id'].nunique()

# order_count_frequency = order_review_count.value_counts().sort_index()

# plt.figure(figsize=(8, 6))
# ax = order_count_frequency.plot(kind='bar', color='skyblue', edgecolor='black')

# plt.title('Amount of order_id per review_id')
# plt.xlabel('frequency order_id per review')
# plt.ylabel('amount review_id')

# for i, count in enumerate(order_count_frequency):
#     ax.text(i, count + 0.2, str(count), ha='center', va='bottom', fontsize=12)

# plt.xticks(rotation=0)
# plt.tight_layout()

# plt.show()

""" connect customer_id to product_category """
customers_orders = pd.merge(customers, orders, on='customer_id', how='inner')
orders_order_items = pd.merge(customers_orders, order_items, on='order_id', how='inner')
order_items_products = pd.merge(orders_order_items, products, on='product_id', how='inner')
customer_product_category = pd.merge(order_items_products, translation, on='product_category_name', how='left')
# customer_product_category.to_csv(os.path.join(path, "customer_to_product_category.csv"), index=False)

""" connect product_category to review_score 
not sure if this is possible as reviews are only linked with order """
product_category_review = pd.merge(customer_product_category, order_reviews[['order_id', 'review_score']], on='order_id', how='left')
# product_category_review.to_csv(os.path.join(path, "customer_product_category_with_review.csv"), index=False)

""" connect review_score to seller_id 
did not connect sellers dataset yet"""
products_reviews = pd.merge(order_items_products, order_reviews[['order_id', 'review_score']], on='order_id', how='left')
review_seller = pd.merge(products_reviews, order_items[['order_id', 'seller_id']], on='order_id', how='left')
# final_df.to_csv(os.path.join(path, "review_score_with_seller.csv"), index=False)
