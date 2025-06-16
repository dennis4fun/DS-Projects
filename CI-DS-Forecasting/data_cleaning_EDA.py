import pandas as pd
import os
import matplotlib.pyplot as plt # Added for plotting
import seaborn as sns # Added for enhanced plotting

# --- Configuration ---
# Define the input file and directory.
# The raw CSV data is expected to be in 'data/raw_stock_data.csv'.
input_dir = 'data'
input_csv_file = 'raw_stock_data.csv'
input_filepath = os.path.join(input_dir, input_csv_file)

# Define the output file for the processed data
output_processed_csv_file = 'processed_data.csv'
output_processed_filepath = os.path.join(input_dir, output_processed_csv_file)

# Define the directory for saving plots
plots_dir = os.path.join(input_dir, 'plots')


# --- Load Data ---
print(f"Attempting to load data from '{input_filepath}'...")
try:
    # Read the CSV file into a Pandas DataFrame.
    # We specify 'Date' as a parse_dates column so Pandas tries to convert it automatically.
    # Set 'Ticker' as a categorical type early for potential memory efficiency and specific operations.
    df = pd.read_csv(input_filepath, parse_dates=['Date'])
    print("Data loaded successfully.")
    print(f"Initial DataFrame shape: {df.shape}")
    print(f"Initial DataFrame columns: {df.columns.tolist()}")
    print("Sample of initial DataFrame (first 5 rows):\n", df.head().to_string())
except FileNotFoundError:
    print(f"Error: The file '{input_filepath}' was not found.")
    print("Please ensure the 'raw_stock_data.csv' file exists in the 'data/' directory and that the scraping script ran successfully.")
    exit() # Exit the script if the file is not found
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    exit() # Exit for other loading errors

# --- Data Cleaning and Type Conversions ---
print("\nStarting data cleaning and type conversions...")

# 1. Convert 'Date' column to datetime objects (if not already done by parse_dates)
# This is crucial for time-series analysis.
df['Date'] = pd.to_datetime(df['Date'])
print("Converted 'Date' column to datetime objects.")

# 2. Convert 'Ticker' column to categorical type
df['Ticker'] = df['Ticker'].astype('category')
print("Converted 'Ticker' column to categorical type.")

# 3. Handle 'Adj Close' column being empty/NaN if 'auto_adjust=True' was used in scraping
# If 'Adj Close' is entirely NaN and 'Close' exists, copy 'Close' to 'Adj Close'.
# This assumes 'Close' is already adjusted when auto_adjust=True in yfinance.
if 'Adj Close' in df.columns and df['Adj Close'].isnull().all() and 'Close' in df.columns:
    df['Adj Close'] = df['Close']
    print("Filled 'Adj Close' column by copying values from 'Close' (assuming 'Close' is adjusted).")
elif 'Adj Close' not in df.columns and 'Close' in df.columns:
    # If 'Adj Close' column doesn't exist at all, but 'Close' does, create it
    df['Adj Close'] = df['Close']
    print("Created 'Adj Close' column by copying values from 'Close'.")

# 4. Convert relevant numerical columns to appropriate numeric types.
# Use errors='coerce' to turn any non-convertible values into NaN.
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
for col in numeric_cols:
    if col in df.columns: # Ensure the column exists before trying to convert
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"Converted '{col}' to numeric type.")
    else:
        print(f"Warning: Numeric column '{col}' not found in DataFrame. Skipping conversion.")

# 5. Handle Missing Values (NaNs)
initial_nan_count = df.isnull().sum().sum()
if initial_nan_count > 0:
    print(f"\nDetected {initial_nan_count} missing values. Attempting to fill...")
    # For time-series data, forward-fill then backward-fill within each ticker group is robust.
    df = df.groupby('Ticker').ffill().bfill()
    print("Missing numerical values filled using forward/backward fill within each ticker group.")
else:
    print("\nNo missing values detected initially.")

# After filling, check if any NaNs remain (e.g., if entire columns/groups were NaN, or for Ticker/Date)
remaining_nan_count = df.isnull().sum().sum()
if remaining_nan_count > 0:
    print(f"Warning: {remaining_nan_count} missing values still remain after filling. Dropping rows with NaNs.")
    # Drop rows where critical columns (Date, Open, Close, Adj Close) might still have NaNs
    df.dropna(subset=['Date', 'Open', 'Close', 'Adj Close'], inplace=True)
    print(f"Remaining NaNs addressed by dropping rows. New shape: {df.shape}")
else:
    print("No missing values remain after filling.")


# 6. Remove Duplicate Rows (if any, unlikely but good practice)
initial_rows_after_nan_handle = df.shape[0]
df.drop_duplicates(inplace=True)
if df.shape[0] < initial_rows_after_nan_handle:
    print(f"Removed {initial_rows_after_nan_handle - df.shape[0]} duplicate rows.")
else:
    print("No duplicate rows found.")

# 7. Sort Data (Crucial for time-series)
# Sort by Ticker first, then by Date, to ensure time-series integrity for each stock.
df.sort_values(by=['Ticker', 'Date'], inplace=True)
print("Data sorted by 'Ticker' and 'Date'.")

# --- Basic Exploratory Data Analysis (EDA) ---
print("\n--- Cleaned Data Overview (EDA) ---")
print("DataFrame Info (data types and non-null counts):")
df.info()

print("\nDataFrame Description (descriptive statistics for numerical columns):")
print(df.describe())

print("\nFirst 5 rows of the cleaned DataFrame:")
print(df.head().to_string())

print("\nLast 5 rows of the cleaned DataFrame:")
print(df.tail().to_string())

print("\nNumber of unique tickers:", df['Ticker'].nunique())
print("Unique tickers found:", df['Ticker'].unique().tolist())

# Check data range
print(f"\nData covers from {df['Date'].min().date()} to {df['Date'].max().date()}")

# Example: Check for gaps in dates (per ticker)
print("\nChecking for potential date gaps (first 5 per ticker):")
for ticker in df['Ticker'].unique():
    ticker_df = df[df['Ticker'] == ticker].copy()
    if not ticker_df.empty:
        # Calculate daily frequency for each ticker
        all_dates = pd.date_range(start=ticker_df['Date'].min(), end=ticker_df['Date'].max(), freq='B') # 'B' for business day
        missing_dates = all_dates.difference(ticker_df['Date'])
        if len(missing_dates) > 0:
            print(f"  {ticker}: Found {len(missing_dates)} potential missing business days (e.g., holidays, market closures, actual data gaps). First 5 missing: {missing_dates.tolist()[:5]}")
        else:
            print(f"  {ticker}: No missing business days detected.")


print("\n--- Data Visualization (EDA) ---")

# Ensure the plots directory exists
os.makedirs(plots_dir, exist_ok=True)
print(f"Ensured plots directory '{plots_dir}' exists for saving visualizations.")

# 1. Stock Price Timeline (Adj Close over time for all tickers)
plt.figure(figsize=(15, 7))
sns.lineplot(data=df, x='Date', y='Adj Close', hue='Ticker', marker='o', markersize=4, linewidth=1)
plt.title('Adjusted Closing Price Over Time for Selected S&P 500 Stocks')
plt.xlabel('Date')
plt.ylabel('Adjusted Closing Price (USD)')
plt.legend(title='Ticker', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'stock_price_timeline.png')) # Save plot instead of showing
plt.close() # Close the plot to free memory
print(f"Generated and saved: Stock Price Timeline to {os.path.join(plots_dir, 'stock_price_timeline.png')}")

# 2. Daily Trading Volume Over Time
plt.figure(figsize=(15, 7))
sns.lineplot(data=df, x='Date', y='Volume', hue='Ticker', linewidth=1)
plt.title('Daily Trading Volume Over Time for Selected S&P 500 Stocks')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend(title='Ticker', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'daily_trading_volume.png')) # Save plot instead of showing
plt.close() # Close the plot to free memory
print(f"Generated and saved: Daily Trading Volume Timeline to {os.path.join(plots_dir, 'daily_trading_volume.png')}")


# 3. Distribution of Daily Returns
# Calculate daily returns for each stock
df['Daily Return'] = df.groupby('Ticker')['Adj Close'].pct_change()

plt.figure(figsize=(15, 7))
sns.histplot(data=df, x='Daily Return', hue='Ticker', kde=True, bins=50, palette='viridis', alpha=0.6)
plt.title('Distribution of Daily Returns for Selected S&P 500 Stocks')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Ticker', bbox_to_anchor=(1.05, 1), loc='upper left') # Ensure legend visibility
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'daily_returns_distribution.png')) # Save plot instead of showing
plt.close() # Close the plot to free memory
print(f"Generated and saved: Distribution of Daily Returns to {os.path.join(plots_dir, 'daily_returns_distribution.png')}")


# --- Note on Confusion Matrix ---
print("\n--- Note on Confusion Matrix ---")
print("A Confusion Matrix is typically used for evaluating classification models, where you predict discrete categories (e.g., 'stock goes up' or 'stock goes down').")
print("For stock price forecasting (predicting the exact future price), which is usually a regression problem, a Confusion Matrix is not directly applicable.")
print("If you later transform your forecasting problem into a classification problem (e.g., predicting daily price movement direction), then a Confusion Matrix would be a relevant visualization.")

print("\n--- Data Prepped for ML Forecasting ---")
print("The DataFrame 'df' is now cleaned and ready for feature engineering and ML model training.")

# --- Save Cleaned and Processed Data ---
os.makedirs(input_dir, exist_ok=True) # Ensure data directory exists (redundant but safe)
df.to_csv(output_processed_filepath, index=False)
print(f"\nCleaned and processed data saved successfully to '{output_processed_filepath}'")

print("\nEDA and Data Cleaning script execution finished.")
