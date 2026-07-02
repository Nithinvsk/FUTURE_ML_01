import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =====================================================
# DISPLAY SETTINGS
# =====================================================

pd.set_option('display.max_rows', None)

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("data/superstore.csv", encoding="latin1")

print("\n===== DATASET LOADED SUCCESSFULLY =====")
print(df.head())

# =====================================================
# MISSING VALUE REPORT
# =====================================================

print("\n===== MISSING VALUE REPORT =====")
print(df.isnull().sum())

# =====================================================
# DATE CONVERSION
# =====================================================

df['Order Date'] = pd.to_datetime(df['Order Date'])

# =====================================================
# DAILY SALES AGGREGATION
# =====================================================

daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()

# =====================================================
# TIME-BASED FEATURE ENGINEERING
# =====================================================

daily_sales['Days'] = (
        daily_sales['Order Date']
        - daily_sales['Order Date'].min()
).dt.days

daily_sales['Month'] = daily_sales['Order Date'].dt.month
daily_sales['Year'] = daily_sales['Order Date'].dt.year
daily_sales['DayOfWeek'] = daily_sales['Order Date'].dt.dayofweek
daily_sales['Quarter'] = daily_sales['Order Date'].dt.quarter

print("\n===== SAMPLE TIME FEATURES =====")
print(
    daily_sales[
        ['Order Date',
         'Days',
         'Month',
         'Year',
         'DayOfWeek',
         'Quarter']
    ].head()
)

# =====================================================
# FEATURES AND TARGET
# =====================================================

X = daily_sales[
    ['Days',
     'Month',
     'Year',
     'DayOfWeek',
     'Quarter']
]

y = daily_sales['Sales']

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# =====================================================
# MODEL TRAINING
# =====================================================

model = LinearRegression()
model.fit(X_train, y_train)

# =====================================================
# PREDICTIONS
# =====================================================

predictions = model.predict(X_test)

# =====================================================
# MODEL EVALUATION
# =====================================================

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

print("\n===== MODEL PERFORMANCE =====")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))

# =====================================================
# HISTORICAL SALES TREND
# =====================================================

plt.figure(figsize=(12, 6))

plt.plot(
    daily_sales['Order Date'],
    daily_sales['Sales']
)

plt.title("Historical Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True)

plt.savefig("outputs/historical_sales.png")
plt.show()

# =====================================================
# MONTHLY SALES TREND (SEASONALITY ANALYSIS)
# =====================================================

monthly_sales = (
    df.set_index('Order Date')
      .resample('ME')['Sales']
      .sum()
)

plt.figure(figsize=(12, 6))

monthly_sales.plot()

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.grid(True)

plt.savefig("outputs/monthly_sales_trend.png")
plt.show()

# =====================================================
# FUTURE FORECAST (90 DAYS)
# =====================================================

future_dates = pd.date_range(
    start=daily_sales['Order Date'].max()
          + pd.Timedelta(days=1),
    periods=90
)

future_df = pd.DataFrame({
    'Order Date': future_dates
})

future_df['Days'] = (
        future_df['Order Date']
        - daily_sales['Order Date'].min()
).dt.days

future_df['Month'] = future_df['Order Date'].dt.month
future_df['Year'] = future_df['Order Date'].dt.year
future_df['DayOfWeek'] = future_df['Order Date'].dt.dayofweek
future_df['Quarter'] = future_df['Order Date'].dt.quarter

future_sales = model.predict(
    future_df[
        ['Days',
         'Month',
         'Year',
         'DayOfWeek',
         'Quarter']
    ]
)

# =====================================================
# FORECAST VISUALIZATION
# =====================================================

plt.figure(figsize=(12, 6))

plt.plot(
    daily_sales['Order Date'],
    daily_sales['Sales'],
    label='Historical Sales'
)

plt.plot(
    future_dates,
    future_sales,
    label='90-Day Forecast'
)

plt.title("Sales Forecast - Next 90 Days")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)

plt.savefig("outputs/sales_forecast.png")
plt.show()

# =====================================================
# SAVE FORECAST RESULTS
# =====================================================

forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Sales': future_sales
})

forecast_df['Predicted Sales'] = (
    forecast_df['Predicted Sales']
    .round(2)
)

forecast_df.to_csv(
    "outputs/forecast_results.csv",
    index=False
)

# =====================================================
# DATASET SUMMARY
# =====================================================

print("\n===== DATASET ROWS =====")

print("Original Dataset Rows:", len(df))
print("Daily Sales Rows:", len(daily_sales))
print("Forecast Rows:", len(forecast_df))

# =====================================================
# DATE RANGE
# =====================================================

print("\n===== DATASET DATE RANGE =====")

print("Earliest Date:", df['Order Date'].min())
print("Latest Date:", df['Order Date'].max())

# =====================================================
# BUSINESS INSIGHTS
# =====================================================

print("\n===== BUSINESS INSIGHTS =====")

print(
    f"Dataset covers years "
    f"{daily_sales['Year'].min()} "
    f"to "
    f"{daily_sales['Year'].max()}."
)

print(
    "The forecast estimates sales for the next 90 days."
)

print(
    "Businesses can use these forecasts for:"
)

print("- Inventory Planning")
print("- Demand Forecasting")
print("- Cash Flow Management")
print("- Staffing Decisions")
print("- Sales Target Planning")

# =====================================================
# FORECAST OUTPUT
# =====================================================

print("\nForecast saved successfully!")

print("\n===== 90 DAY FORECAST =====")
print(forecast_df)