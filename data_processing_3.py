import pandas as pd
import os
import itertools


"""
Building a dataset for time series forecasting_______________________________________________________________________________________
"""
# Define file names
data_dir = os.path.join(os.getcwd(), "data")
processed_data_dir = os.path.join(data_dir, "processed_data")

file_names = {
    "customer": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "product_category_name": "product_category_name_translation.csv"
}

# Load CSVs into a dictionary
dfs = {key: pd.read_csv(os.path.join(data_dir, file)) for key, file in file_names.items()}

# Merge datasets incrementally
df_all = (
    dfs["customer"]
    .merge(dfs["orders"][["customer_id", "order_status", "order_id", "order_purchase_timestamp"]], on="customer_id", how="left")
    .merge(dfs["reviews"][["order_id", "review_score"]], on="order_id", how="left")
    .merge(dfs["payments"][["order_id"]], on="order_id", how="left")
    .merge(dfs["order_items"][["order_id", "product_id", "seller_id", "price", "freight_value", "order_item_id"]], on="order_id", how="left")
    .merge(dfs["sellers"][["seller_id", "seller_state"]], on="seller_id", how="left")
    .merge(dfs["products"][["product_id", "product_category_name"]], on="product_id", how="left")
    .merge(dfs["product_category_name"], on="product_category_name", how="left")
)

# Drop unnecessary columns
cols_to_drop = [
    "customer_id", "customer_unique_id", "order_id",
    "product_id", "seller_id", "product_category_name", 'customer_zip_code_prefix', 'customer_city', 
]
df_useful = df_all.drop(columns=cols_to_drop)

# Keep only delivered orders
df_useful = df_useful[df_useful["order_status"] == "delivered"]

# label missing values in product_category_name_english as unknown
#df_useful['product_category_name_english'].fillna('unknown', inplace=True)

# drop rows with missing values in product_category_name_english
df_useful = df_useful.dropna()

# drop orders with status other than delivered
df_useful = df_useful[df_useful['order_status'] == 'delivered']
df_useful.drop(['order_status'], axis=1, inplace=True)

# map categories to more general categories
category_mapping = {

    # Home & Furniture
    
    'furniture_decor': 'furniture',
    'furniture_living_room': 'furniture',
    'furniture_bedroom': 'furniture',
    'furniture_mattress_and_upholstery': 'furniture',
    'kitchen_dining_laundry_garden_furniture': 'furniture',
    'bed_bath_table': 'furniture',
    
    # Construction & Tools
    'costruction_tools_tools': 'construction_tools',
    'costruction_tools_garden': 'construction_tools',
    'construction_tools_lights': 'construction_tools',
    'construction_tools_construction': 'construction_tools',
    'construction_tools_safety': 'construction_tools',
    'home_construction': 'construction_tools',

    # Home Appliances
    'home_appliances_2': 'home_appliances',
    'small_appliances': 'home_appliances',
    'small_appliances_home_oven_and_coffee': 'home_appliances',
    'air_conditioning': 'home_appliances',
    'home_confort': 'home_appliances',
    'home_comfort_2': 'home_appliances',

    # Fashion
    'fashio_female_clothing': 'fashion',
    'fashion_male_clothing': 'fashion',
    'fashion_childrens_clothes': 'fashion',
    'fashion_shoes': 'fashion',
    'fashion_bags_accessories': 'fashion',
    'fashion_underwear_beach': 'fashion',
    'fashion_sport': 'fashion',

    # Electronics & Tech
    'electronics': 'electronics',
    'tablets_printing_image': 'electronics',
    'computers': 'electronics',
    'electronics_accessories': 'electronics',
    'computers_accessories': 'electronics',
    'audio': 'electronics',
    'telephony': 'telephony',
    'fixed_telephony': 'telephony',

    # Books & Entertainment
    'books_general_interest': 'entertainment',
    'books_imported': 'entertainment',
    'books_technical': 'entertainment',
    'cds_dvds_musicals': 'entertainment',
    'dvds_blu_ray': 'entertainment',
    'cine_photo': 'entertainment',
    'consoles_games': 'entertainment',
    'arts_and_craftmanship': 'entertainment',
    'art': 'entertainment',
    'music': 'entertainment',
    'musical_instruments': 'entertainment',
    
    # Automotive
    'auto': 'automotive',

    # Beauty & Health
    'health_beauty': 'beauty_health',
    'perfumery': 'beauty_health',
    'diapers_and_hygiene': 'beauty_health',

    # Food & Beverages
    'food_drink': 'food',
    'food': 'food',
    'drinks': 'food',
    'la_cuisine': 'food',

    # Miscellaneous
    'market_place': 'other',
    'cool_stuff': 'other',
    'security_and_services': 'other',
    'signaling_and_security': 'other',
    'party_supplies': 'other',
    'christmas_supplies': 'other',
    'watches_gifts': 'gifts',
    'flowers': 'gifts',
    'pet_shop': 'pets',
    'baby': 'baby',
    'sports_leisure': 'sports',
    'toys': 'toys',
    'stationery': 'office_supplies',
    'office_furniture': 'office_supplies',
    'luggage_accessories': 'luggage',
    'industry_commerce_and_business': 'other',
    'agro_industry_and_commerce': 'other',
}

df_useful['category'] = df_useful['product_category_name_english'].replace(category_mapping)
df_useful.drop(['product_category_name_english'], axis=1, inplace=True)

# convert date to datetime
df_useful['date'] = pd.to_datetime(df_useful['order_purchase_timestamp']).dt.strftime('%Y-%m-%d')
df_useful.drop(['order_purchase_timestamp'], axis=1, inplace=True)

# add day of week, day of the month and week of the year
df_useful['day_of_week'] = pd.to_datetime(df_useful['date']).dt.day_name()
df_useful['day_of_month'] = pd.to_datetime(df_useful['date']).dt.day
df_useful['week_of_year'] = pd.to_datetime(df_useful['date']).dt.isocalendar().week

# Extract year and week number from 'order_delivered_customer_date'
df_useful['year_week'] = pd.to_datetime(df_useful['date']).dt.to_period('W')

category_counts = df_useful['category'].value_counts()
print(category_counts)

# Filter out data from 2016 because highly discontinuous
df_useful = df_useful[pd.to_datetime(df_useful['date']).dt.year > 2016]

# Get unique weeks and categories
unique_weeks = df_useful['year_week'].unique()
nunique_weeks = len(unique_weeks)
unique_categories = df_useful['category'].unique()
nunique_categories = len(unique_categories)

# Create a full DataFrame with all possible (week, category) pairs
full_weeks_categories = pd.DataFrame(
    list(itertools.product(unique_weeks, unique_categories)), 
    columns=['year_week', 'category']
)

# Group by week and category
weekly_sales = df_useful.groupby(['year_week', 'category']).agg(
    num_items_sold=('order_item_id', 'count'),  # Count items sold (using order_item_id)
    total_price=('price', 'sum'),       # Sum of total price
    total_freight_value=('freight_value', 'sum'),  # Sum of total freight value
    most_sold_state=('seller_state', lambda x: x.mode()[0]),  # State that sold the most
    most_bought_state=('customer_state', lambda x: x.mode()[0]),  # State that bought the most
    avg_review_score=('review_score', 'mean'), # Average review score
    week_of_year=('week_of_year', 'first')  # Week of the year   
).reset_index()
weekly_sales = weekly_sales.sort_values(by=['year_week', 'category']).reset_index(drop=True)

# Merge with full DataFrame to fill missing values
weekly_sales_complete = full_weeks_categories.merge(weekly_sales, on=['year_week', 'category'], how='left')
weekly_sales_complete.fillna({
    'num_items_sold': 0,
    'total_price': 0.0,
    'total_freight_value': 0.0,
    'most_sold_state': 'Unknown',
    'most_bought_state': 'Unknown'
}, inplace=True)

# Sort by week 
weekly_sales_complete = weekly_sales_complete.sort_values(by=['year_week', 'category']).reset_index(drop=True)

# Compute average review score to fill missing values
overall_avg = df_useful['review_score'].mean()
weekly_sales_complete['avg_review_score'].fillna(overall_avg, inplace=True)

# fill missing values in week_of_year
weekly_sales_complete['week_of_year'] = weekly_sales_complete['year_week'].astype(str).str.split('-').str[1].astype(int)


print(weekly_sales.head(20))
print(weekly_sales_complete.head(20))

# Save to CSV to processed_data directory
weekly_sales_complete.to_csv(os.path.join(processed_data_dir, "weekly_sales_complete.csv"), index=False)
weekly_sales.to_csv(os.path.join(processed_data_dir, "weekly_sales_no_fill.csv"), index=False)


# take the last 17 weeks as test data and the rest as training data (approximately 80% training and 20% test)
weeks = weekly_sales_complete['year_week'].unique()
train_weeks = weeks[:-17]
test_weeks = weeks[-17:]
train_df = weekly_sales_complete[weekly_sales_complete['year_week'].isin(train_weeks)]
test_df = weekly_sales_complete[weekly_sales_complete['year_week'].isin(test_weeks)]

# Save to CSV 
train_df.to_csv(os.path.join(processed_data_dir, "train.csv"), index=False)
test_df.to_csv(os.path.join(processed_data_dir, "test.csv"), index=False)

print("")
print("Data Processing Completed")
print("")
print("Train dataframe shape: ", train_df.shape)
print(f"({70} weeks, {nunique_categories} categories)")
print("")
print("Test dataframe shape: ", test_df.shape)
print(f"({17} weeks, {nunique_categories} categories)")