import pandas as pd
import matplotlib.pyplot as plt
import os

# Define file paths (adjust if needed)
base_path = os.getcwd() + "/data/processed_data"
test_path = os.path.join(base_path, "test.csv")
predicted_path = os.path.join(base_path, "arima_prediction.csv")  # can change to arima_prediction_cv.csv

# Load test and predicted datasets
test_df = pd.read_csv(test_path)
pred_df = pd.read_csv(predicted_path)

# --- Process test file ---
# Define a cleaning function for the test file's "year_week" column.
# This function extracts the second part (after the '/') and removes the trailing '-1' if present.
def clean_year_week_test(s):
    parts = s.split('/')
    # Use the second part if it exists; otherwise use the whole string.
    date_part = parts[1] if len(parts) > 1 else parts[0]
    if date_part.endswith('-1'):
        date_part = date_part[:-2]
    return date_part

# Create a new column 'week' in test_df using the cleaning function and convert it to datetime.
test_df['week'] = pd.to_datetime(test_df['year_week'].apply(clean_year_week_test), format='%Y-%m-%d')

# --- Process predicted file ---
# The predicted file already has a "week" column as a string.
# Convert it to datetime.
pred_df['week'] = pd.to_datetime(pred_df['week'], format='%Y-%m-%d')

# --- Merge test and predicted datasets ---
# We assume both datasets have a "category" column and we merge on "week" and "category".
# In the merged dataframe, we'll have the actual total_price from test and the forecast total_price.
merged = pd.merge(
    test_df[['week', 'category', 'total_price']],
    pred_df[['week', 'category', 'total_price']],
    on=['week', 'category'],
    suffixes=('_actual', '_forecast')
)

print("Merged data sample:")
print(merged.head())

# --- Plot the results ---
categories = merged['category'].unique()

for cat in categories:
    cat_data = merged[merged['category'] == cat].sort_values('week')
    
    plt.figure(figsize=(10, 5))
    plt.plot(cat_data['week'], cat_data['total_price_actual'], marker='o', label='Actual')
    plt.plot(cat_data['week'], cat_data['total_price_forecast'], marker='x', label='Forecast')
    plt.title(f"Actual vs Forecast for Category: {cat}")
    plt.xlabel("Week (Start Date)")
    plt.ylabel("Total Price")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
