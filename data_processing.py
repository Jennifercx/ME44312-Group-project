# --- import packages -------------------------------------------------------------------------------------------------
import pandas as pd
import os
import plotly.express as px
import numpy as np

# data directory
data_dir = os.path.join(os.getcwd(), "data")
processed_data_dir = os.path.join(data_dir, "processed_data")
print("Current Working Directory:", os.getcwd())

# import dataframes seperately
df_customer = pd.read_csv(os.path.join(data_dir, "olist_customers_dataset.csv"))
df_geolocation = pd.read_csv(os.path.join(data_dir, "olist_geolocation_dataset.csv"))
df_order_items = pd.read_csv(os.path.join(data_dir, "olist_order_items_dataset.csv"))
df_payments = pd.read_csv(os.path.join(data_dir, "olist_order_payments_dataset.csv"))
df_reviews = pd.read_csv(os.path.join(data_dir, "olist_order_reviews_dataset.csv"))
df_orders = pd.read_csv(os.path.join(data_dir, "olist_orders_dataset.csv"))
df_products = pd.read_csv(os.path.join(data_dir, "olist_products_dataset.csv"))
df_sellers = pd.read_csv(os.path.join(data_dir, "olist_sellers_dataset.csv"))
df_product_category_name = pd.read_csv(os.path.join(data_dir, "product_category_name_translation.csv"))

# combine all datasets and save to .csv file
df_all = df_customer.merge(df_orders, on='customer_id', how='left').merge(df_reviews, on='order_id', how='left').merge(df_payments, on='order_id', how='left').merge(df_order_items, on='order_id', how='left').merge(df_sellers, on='seller_id', how='left').merge(df_products, on='product_id', how='left').merge(df_product_category_name, on='product_category_name', how='left')
df_shipping_time = pd.DataFrame((pd.to_datetime(df_orders["order_delivered_customer_date"]) -pd.to_datetime(df_orders["order_purchase_timestamp"])).dt.total_seconds()/(24*3600))
df_all["shipping_time"] = pd.DataFrame((pd.to_datetime(df_orders["order_delivered_customer_date"]) -pd.to_datetime(df_orders["order_purchase_timestamp"])).dt.total_seconds()/(24*3600))
df_all.to_csv('data/merged_data.csv', index=False)

############################################################################################################################################################################################
# --- Data Cleaning -------------------------------------------------------------------------------------------------------
############################################################################################################################################################################################
visualise = False # Set to True to visualise the cleaning process

# Create a copy of the original dataframe before cleaning
df_orig = df_all.copy()

# Generate a mask for rows that satisfy all conditions:
clean_mask = (
    df_orig['product_category_name_english'].notna() & # drop rows with missing values in product_category_name_english
    (pd.to_datetime(df_orig["order_purchase_timestamp"]).dt.year > 2016) & # Filter out data from 2016 because highly discontinuous
    (df_orig["order_status"] == "delivered") & # Filter out orders that are not delivered
    (pd.to_datetime(df_orig["order_purchase_timestamp"]).dt.date < pd.to_datetime("2018-08-27").date())
)

# Save removed rows
df_trash = df_orig[~clean_mask]

# Apply the cleaning
df_all = df_orig[clean_mask]

# --- Visualisation to analyse the cleaning process -------------------------------------------------------------------------------------------------------
if visualise:
    # Convert timestamps to datetime objects
    sales_clean = (
        df_all.groupby(pd.to_datetime(df_all['order_purchase_timestamp']).dt.date)
            .size()
            .reset_index(name='count')
    )
    sales_clean.columns = ['order_date', 'count']
    sales_clean['type'] = 'Cleaned Sales'

    sales_trash = (
        df_trash.groupby(pd.to_datetime(df_trash['order_purchase_timestamp']).dt.date)
            .size()
            .reset_index(name='count')
    )
    sales_trash.columns = ['order_date', 'count']
    sales_trash['type'] = 'Removed Sales'
    # Combine the two datasets
    sales_df = pd.concat([sales_clean, sales_trash], ignore_index=True)
    # Convert order_date back to datetime for Plotly's x-axis
    sales_df['order_date'] = pd.to_datetime(sales_df['order_date'])
    # Create an interactive grouped bar chart with green and red bars
    fig = px.bar(
        sales_df,
        x='order_date',
        y='count',
        color='type',
        barmode='group',
        custom_data=['order_date', 'type'],  # Pass order_date and type for custom hover text
        hover_data={'count': True},           # Only include count from the data
        labels={'order_date': 'Date', 'count': 'Number of Sales'},
        title='Daily Sales: Cleaned vs Removed Data',
        color_discrete_map={'Cleaned Sales': 'green', 'Removed Sales': 'red'}
    )
    # Update hovertemplate to include the day of the week (formatted via strftime)
    fig.update_traces(
        hovertemplate=
        'Date: %{customdata[0]|%a, %Y-%m-%d}<br>' +
        'Count: %{y}<br>' +
        'Type: %{customdata[1]}'
    )
    # Update x-axis so dates include day-of-week (e.g., "Mon, 2025-03-25")
    fig.update_xaxes(tickformat="%a, %Y-%m-%d", tickangle=-45)
    # Set both the paper and plot backgrounds to white
    fig.update_layout(paper_bgcolor='#212121',plot_bgcolor='#262626',  font=dict(color='white'),title_font=dict(color='white'))
    fig.update_traces(marker_line_width=0)
    fig.show()
# map categories to more general categories -----------------------------------------------------------------------------------------------------------------------------------------------

category_mapping = {

    # Home & Furniture
    
    'furniture_decor': 'furniture',
    'furniture_living_room': 'furniture',
    'furniture_bedroom': 'furniture',
    'furniture_mattress_and_upholstery': 'furniture',
    'kitchen_dining_laundry_garden_furniture': 'furniture',
    'bed_bath_table': 'furniture',
    
    # Construction & Tools
    'costruction_tools_tools': 'tools',
    'costruction_tools_garden': 'tools',
    'construction_tools_lights': 'tools',
    'construction_tools_construction': 'tools',
    'construction_tools_safety': 'tools',
    'home_construction': 'tools',
    'garden_tools': 'tools',

    # Home
    'home_appliances_2': 'home',
    'small_appliances': 'home',
    'small_appliances_home_oven_and_coffee': 'home',
    'air_conditioning': 'home',
    'home_confort': 'home',
    'home_comfort_2': 'home',
    'home_appliances': 'home',
    'housewares': 'home',

    # Fashion
    'fashio_female_clothing': 'fashion',
    'fashion_male_clothing': 'fashion',
    'fashion_childrens_clothes': 'fashion',
    'fashion_shoes': 'fashion',
    'fashion_bags_accessories': 'fashion',
    'fashion_underwear_beach': 'fashion',
    'fashion_sport': 'fashion',
     'luggage_accessories': 'fashion',

    # Electronics & Tech
    'electronics': 'electronics',
    'tablets_printing_image': 'electronics',
    'computers': 'electronics',
    'electronics_accessories': 'electronics',
    'computers_accessories': 'electronics',
    'audio': 'electronics',
    'telephony': 'electronics',
    'fixed_telephony': 'electronics',

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
    'market_place': 'miscellaneous',
    'cool_stuff': 'miscellaneous',
    'security_and_services': 'miscellaneous',
    'signaling_and_security': 'miscellaneous',
    'party_supplies': 'miscellaneous',
    'christmas_supplies': 'miscellaneous',
    'watches_gifts': 'miscellaneous',
    'flowers': 'miscellaneous',
    'pet_shop': 'pets',
    'baby': 'baby',
    'sports_leisure': 'sports',
    'toys': 'toys',
    'stationery': 'office_supplies',
    'office_furniture': 'office_supplies',
    'industry_commerce_and_business': 'miscellaneous',
    'agro_industry_and_commerce': 'miscellaneous',
}

df_all['product_category_name_english'] = df_all['product_category_name_english'].replace(category_mapping)

# print the category count
category_counts = df_all['product_category_name_english'].value_counts()
print(category_counts)

############################################################################################################################################################################################
# --- create weekly dataset -------------------------------------------------------------------------------------------------------
############################################################################################################################################################################################

# select relevant columns only
df_data = df_all[["order_purchase_timestamp", "review_score", "price", "freight_value", "shipping_time", "product_category_name_english"]]
df_data.dropna(inplace=True)
df_data["week"] = pd.to_datetime(df_data['order_purchase_timestamp']).dt.to_period("W")

# per week and per item type dataframes
item = df_data.groupby(['week', 'product_category_name_english']).size().unstack(fill_value=0)
total_items = df_data.groupby('week').size()
#item = item.div(total_items, axis=0)

price = df_data.groupby(['week', 'product_category_name_english'])['price'].sum().unstack(fill_value=0)
total_price = df_data.groupby('week')['price'].sum()
#price = price.div(total_price, axis=0)

freight_value = df_data.groupby(['week', 'product_category_name_english'])['freight_value'].sum().unstack(fill_value=0)
total_freight_value = df_data.groupby('week')['price'].sum()
#freight_value = freight_value.div(total_freight_value, axis=0)

overall_avg = df_data['review_score'].mean() # overall average review score for filling missing values
review_score = df_data.groupby(['week', 'product_category_name_english'])['review_score'].mean().unstack(fill_value=overall_avg)

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
final_df = (
    item
    .merge(price, on="week", how="left")
    .merge(freight_value, on="week", how="left")
    .merge(review_score, on="week", how="left")
    .merge(shipping_time, on="week", how="left")
)
# add month and week of month columns
final_df['week'] = final_df['week'].astype(str)  # Convert to string
final_df['start_date'] = final_df['week'].str.split('/').str[0]  # Extract start date
final_df['start_date'] = pd.to_datetime(final_df['start_date'])  # Convert to datetime
final_df['month_of_year'] = final_df['start_date'].dt.month  # Month of year (1-12)
final_df['week_of_month'] = final_df['start_date'].apply(lambda x: (x.day - 1) // 7 + 1)  # Week of month (1-5)
final_df.drop(columns=['start_date'], inplace=True) 

cols = ['week', 'month_of_year', 'week_of_month'] + [col for col in final_df.columns if col not in ['week', 'month_of_year', 'week_of_month']]
final_df = final_df[cols]

############################################################################################################################################################################################
# --- split data into training and test data -------------------------------------------------------------------------------------------------------
############################################################################################################################################################################################

# take the last 17 weeks as test data and the rest as training data (approximately 80% training and 20% test)
weeks = final_df['week'].unique()
train_weeks = weeks[:-17] # 69 weeks
test_weeks = weeks[-17:]
train_df = final_df[final_df['week'].isin(train_weeks)]
test_df = final_df[final_df['week'].isin(test_weeks)]

final_df.to_csv(os.path.join(processed_data_dir, "processed_dataset.csv"), index=False)
train_df.to_csv(os.path.join(processed_data_dir, "train_data.csv"), index=False)
test_df.to_csv(os.path.join(processed_data_dir, "test_data.csv"), index=False)