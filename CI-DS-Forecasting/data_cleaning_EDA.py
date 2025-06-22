import pandas as pd
import os
import matplotlib.pyplot as plt # Added for plotting
import seaborn as sns # Added for enhanced plotting
import numpy as np # Import numpy for np.nan

# --- Configuration ---
# --- Path Correction to ensure output goes to repository root ---
# Get the absolute path of the current script file
script_path = os.path.abspath(__file__)
# Get the directory of the script
script_dir = os.path.dirname(script_path)
# Go up one level to reach the repository root
repo_root = os.path.dirname(script_dir)

# Define the input file and directory, now relative to the repository root
input_dir = os.path.join(repo_root, 'data')
input_csv_file = 'raw_stock_data.csv'
input_filepath = os.path.join(input_dir, input_csv_file)

# Define the output file for the processed data, relative to the repository root
output_processed_csv_file = 'processed_data.csv'
output_processed_filepath = os.path.join(input_dir, output_processed_csv_file)

# Define the directory for saving plots, relative to the repository root
plots_dir = os.path.join(input_dir, 'plots')

# Define the minimum number of data points a ticker must have to be included in processed data
MIN_DATA_POINTS_PER_TICKER = 5 

# --- Load Data ---
print(f"Attempting to load data from '{input_filepath}'...")
try:
    df = pd.read_csv(input_filepath, parse_dates=['Date'])
    print("Data loaded successfully.")
    print(f"Initial DataFrame shape: {df.shape}")
    print(f"Initial DataFrame columns: {df.columns.tolist()}")
    print("Sample of initial DataFrame (first 5 rows):\n", df.head().to_string())
except FileNotFoundError:
    print(f"Error: The file '{input_filepath}' was not found.")
    print("Please ensure the 'raw_stock_data.csv' file exists in the 'data/' directory and that the scraping script ran successfully.")
    df = pd.DataFrame() # Initialize an empty DataFrame if file not found
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    df = pd.DataFrame() # Initialize an empty DataFrame for other loading errors

# Early exit if DataFrame is empty after loading
if df.empty:
    print("\nInitial DataFrame is empty after loading. Skipping all cleaning, EDA, and saving steps.")
    print("EDA and Data Cleaning script execution finished (early exit due to empty initial data).")
    # Save an empty CSV file with expected columns to avoid errors in subsequent steps
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    # Define expected columns for an empty DataFrame
    expected_cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'VIX_Close', 'TNX_Close']
    pd.DataFrame(columns=expected_cols).to_csv(output_processed_filepath, index=False)
    print(f"An empty processed_data.csv with expected columns was saved to '{output_processed_filepath}'.")
    exit()


# --- Data Cleaning and Type Conversions ---
print("\nStarting data cleaning and type conversions...")

# 1. Convert 'Date' column to datetime objects
df['Date'] = pd.to_datetime(df['Date'])
print("Converted 'Date' column to datetime objects.")

# 2. Convert 'Ticker' column to categorical type
df['Ticker'] = df['Ticker'].astype('category')
print("Converted 'Ticker' column to categorical type.")

# 3. Handle 'Adj Close' column being empty/NaN (for stock data)
if 'Adj Close' not in df.columns and 'Close' in df.columns:
    df['Adj Close'] = df['Close']
    print("Created 'Adj Close' column by copying values from 'Close'.")
elif 'Adj Close' in df.columns and df['Adj Close'].isnull().all() and 'Close' in df.columns:
    df['Adj Close'] = df['Close']
    print("Filled 'Adj Close' column by copying values from 'Close' (assuming 'Close' is adjusted).")
elif 'Adj Close' not in df.columns and 'Close' not in df.columns:
    df['Adj Close'] = np.nan
    print("Warning: Neither 'Close' nor 'Adj Close' found. 'Adj Close' set to NaN.")


# 4. Convert relevant numerical columns to appropriate numeric types.
# Ensure all columns in this list are handled gracefully if they don't exist
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'VIX_Close', 'TNX_Close'] 
for col in numeric_cols:
    if col in df.columns: 
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"Converted '{col}' to numeric type.")
    else:
        print(f"Warning: Numeric column '{col}' not found in DataFrame. Skipping conversion.")

# 5. Handle Missing Values (NaNs)
initial_nan_count_total = df.isnull().sum().sum()
if initial_nan_count_total > 0:
    print(f"\nDetected {initial_nan_count_total} missing values. Attempting to fill...")
    
    # Fill stock-specific data (Open, High, Low, Close, Volume, Adj Close) per ticker
    stock_value_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    for col in stock_value_cols:
        if col in df.columns:
            if df[col].isnull().any():
                df[col] = df.groupby('Ticker')[col].ffill().bfill()
                print(f"  Filled NaNs for '{col}' (stock-specific, ffill/bfill).")
        
    # For macro data (VIX_Close, TNX_Close), which are common across tickers, fill globally.
    # IMPORTANT: Only fill macro columns if they actually contain *some* valid data.
    macro_cols = ['VIX_Close', 'TNX_Close']
    for macro_col in macro_cols:
        if macro_col in df.columns:
            if df[macro_col].isnull().any():
                # Check if there's any non-null data to propagate BEFORE attempting ffill/bfill
                if df[macro_col].first_valid_index() is not None:
                    df[macro_col] = df[macro_col].ffill().bfill()
                    print(f"  Filled NaNs for '{macro_col}' (global, ffill/bfill).")
                else:
                    print(f"  Warning: Column '{macro_col}' is entirely NaN after initial processing, cannot fill. It will remain NaN for now.")
            else:
                print(f"  No NaNs in '{macro_col}'.")
    
    final_nan_count_total = df.isnull().sum().sum()
    if final_nan_count_total < initial_nan_count_total:
        print(f"Missing numerical values filled. Remaining NaNs: {final_nan_count_total}")
    else:
        print("No further missing values could be filled.")
else:
    print("\nNo missing values detected initially.")


# 6. Remove Duplicate Rows
initial_rows_after_nan_handle = df.shape[0]
df.drop_duplicates(inplace=True)
if df.shape[0] < initial_rows_after_nan_handle:
    print(f"Removed {initial_rows_after_nan_handle - df.shape[0]} duplicate rows.")
else:
    print("No duplicate rows found.")

# 7. Sort Data (Crucial for time-series)
df.sort_values(by=['Ticker', 'Date'], inplace=True)
print("Data sorted by 'Ticker' and 'Date'.")

# --- Refined NaN Dropping Strategy ---
# Define essential columns that MUST have data for a row to be considered valid for *modeling*.
essential_cols_for_row_validity = ['Date', 'Adj Close', 'Ticker'] 

initial_rows_before_final_dropna = df.shape[0]
# Use .copy() to avoid SettingWithCopyWarning later if `df` is a view
df = df.dropna(subset=essential_cols_for_row_validity).copy()

if df.shape[0] < initial_rows_before_final_dropna:
    print(f"Dropped {initial_rows_before_final_dropna - df.shape[0]} rows due to NaNs in essential columns ('Date', 'Adj Close', 'Ticker'). New shape: {df.shape}")
elif df.empty:
    print("CRITICAL WARNING: DataFrame became empty after dropping rows with NaNs in essential columns. This indicates severe data loss.")
else:
    print("No rows dropped due to NaNs in essential columns after primary filling.")


# --- Filter out tickers with insufficient data for analysis/modeling ---
original_tickers = df['Ticker'].unique().tolist() if not df.empty else []

if not df.empty:
    df_filtered_by_length = df.groupby('Ticker', observed=False).filter(lambda x: len(x) >= MIN_DATA_POINTS_PER_TICKER).copy()
    df = df_filtered_by_length # Assign the filtered DataFrame back
    
    filtered_tickers = df['Ticker'].unique().tolist() if not df.empty else []

    if len(original_tickers) != len(filtered_tickers):
        removed_tickers = set(original_tickers) - set(filtered_tickers)
        print(f"Removed tickers with less than {MIN_DATA_POINTS_PER_TICKER} data points after cleaning: {list(removed_tickers)}. Original tickers: {original_tickers}, Filtered tickers: {filtered_tickers}")
else:
    print("DataFrame is empty, skipping ticker filtering by length.")


# --- Final Check if DataFrame is empty after all cleaning/filtering ---
if df.empty:
    print("\nFINAL WARNING: DataFrame is empty after all cleaning and filtering. No processed data will be saved, and no plots will be generated.")
    # Save an empty CSV file with expected columns if the DataFrame is empty
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    expected_cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'VIX_Close', 'TNX_Close']
    pd.DataFrame(columns=expected_cols).to_csv(output_processed_filepath, index=False)
    print(f"An empty processed_data.csv with expected columns was saved to '{output_processed_filepath}'.")
    print("EDA and Data Cleaning script execution finished (empty DataFrame).")
    exit() # Exit to prevent further errors with empty DF


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

print(f"\nData covers from {df['Date'].min().date()} to {df['Date'].max().date()}")

# Example: Check for gaps in dates (per ticker)
print("\nChecking for potential date gaps (first 5 per ticker):")
for ticker in df['Ticker'].unique():
    ticker_df = df[df['Ticker'] == ticker].copy()
    if not ticker_df.empty:
        # Create a complete date range between min and max date for the ticker, including weekends for completeness check
        # Then filter for business days only for comparison
        full_date_range = pd.date_range(start=ticker_df['Date'].min(), end=ticker_df['Date'].max(), freq='D')
        business_days_in_range = full_date_range[full_date_range.weekday < 5] # Exclude Saturday (5) and Sunday (6)
        
        # Actual dates in the dataset for this ticker
        actual_dates = ticker_df['Date'].unique()
        
        # Dates that are in business_days_in_range but not in actual_dates
        missing_dates = pd.DatetimeIndex(business_days_in_range).difference(actual_dates)
        
        if len(missing_dates) > 0:
            print(f"  {ticker}: Found {len(missing_dates)} potential missing business days (e.g., holidays, market closures, actual data gaps). First 5 missing: {missing_dates.date.tolist()[:5]}")
        else:
            print(f"  {ticker}: No missing business days detected.")


print("\n--- Data Visualization (EDA) ---")

# Ensure the plots directory exists
os.makedirs(plots_dir, exist_ok=True)
print(f"Ensured plots directory '{plots_dir}' exists for saving visualizations.")

# 1. Stock Price Timeline (Adj Close over time for all tickers)
plt.figure(figsize=(15, 7))
if df['Ticker'].nunique() > 0 and 'Adj Close' in df.columns and not df['Adj Close'].dropna().empty: 
    sns.lineplot(data=df, x='Date', y='Adj Close', hue='Ticker', marker='o', markersize=4, linewidth=1)
    plt.title('Adjusted Closing Price Over Time for Selected S&P 500 Stocks')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Closing Price (USD)')
    if df['Ticker'].nunique() > 1: # Only add legend if multiple tickers for hue
        plt.legend(title='Ticker', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'stock_price_timeline.png'))
    plt.close()
    print(f"Generated and saved: Stock Price Timeline to {os.path.join(plots_dir, 'stock_price_timeline.png')}")
else:
    print("Skipping Stock Price Timeline plot: Insufficient data or 'Adj Close' column for plotting.")
    plt.close()

# 2. Daily Trading Volume Over Time
plt.figure(figsize=(15, 7))
if df['Ticker'].nunique() > 0 and 'Volume' in df.columns and not df['Volume'].dropna().empty: 
    sns.lineplot(data=df, x='Date', y='Volume', hue='Ticker', linewidth=1)
    plt.title('Daily Trading Volume Over Time for Selected S&P 500 Stocks')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    if df['Ticker'].nunique() > 1: 
        plt.legend(title='Ticker', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'daily_trading_volume.png'))
    plt.close()
    print(f"Generated and saved: Daily Trading Volume Timeline to {os.path.join(plots_dir, 'daily_trading_volume.png')}")
else:
    print("Skipping Daily Trading Volume plot: Insufficient data or 'Volume' column for plotting.")
    plt.close()

# 3. Distribution of Daily Returns
df_for_returns = df.groupby('Ticker', observed=False).filter(lambda x: len(x) > 1).copy()
if not df_for_returns.empty and 'Adj Close' in df_for_returns.columns:
    df_for_returns['Daily Return'] = df_for_returns.groupby('Ticker', observed=False)['Adj Close'].pct_change()
    plot_df_returns = df_for_returns.dropna(subset=['Daily Return']).copy()
    
    if not plot_df_returns.empty and len(plot_df_returns['Daily Return'].dropna()) > 1:
        plt.figure(figsize=(15, 7))
        sns.histplot(data=plot_df_returns, x='Daily Return', hue='Ticker', kde=True, bins=50, palette='viridis', alpha=0.6)
        plt.title('Distribution of Daily Returns for Selected S&P 500 Stocks')
        plt.xlabel('Daily Return')
        plt.ylabel('Frequency')
        if plot_df_returns['Ticker'].nunique() > 1:
            plt.legend(title='Ticker', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'daily_returns_distribution.png'))
        plt.close()
        print(f"Generated and saved: Distribution of Daily Returns to {os.path.join(plots_dir, 'daily_returns_distribution.png')}")
    else:
        print("Skipping Distribution of Daily Returns plot: Insufficient data for valid KDE or histogram after cleaning.")
        plt.close()
else:
    print("Skipping Daily Return calculation or plot: No tickers with sufficient data points for pct_change or 'Adj Close' column missing.")
    plt.close()

# --- NEW: EDA for Macroeconomic Data ---
# Plot VIX Close over time
if 'VIX_Close' in df.columns and df['VIX_Close'].first_valid_index() is not None and len(df['VIX_Close'].dropna().unique()) > 1:
    plt.figure(figsize=(15, 7))
    # Drop duplicates by date for plotting macro data, as it's common across tickers
    sns.lineplot(data=df.drop_duplicates(subset=['Date']), x='Date', y='VIX_Close', color='purple', linewidth=1.5)
    plt.title('VIX (Volatility Index) Close Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('VIX Close')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'vix_close_timeline.png'))
    plt.close()
    print(f"Generated and saved: VIX Close Timeline to {os.path.join(plots_dir, 'vix_close_timeline.png')}")
else:
    print("Skipping VIX Close Timeline plot: 'VIX_Close' column missing or has insufficient data after cleaning.")
    plt.close()

# Plot TNX Close over time
if 'TNX_Close' in df.columns and df['TNX_Close'].first_valid_index() is not None and len(df['TNX_Close'].dropna().unique()) > 1:
    plt.figure(figsize=(15, 7))
    # Drop duplicates by date for plotting macro data, as it's common across tickers
    sns.lineplot(data=df.drop_duplicates(subset=['Date']), x='Date', y='TNX_Close', color='darkorange', linewidth=1.5)
    plt.title('10-Year US Treasury Yield (TNX) Close Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('TNX Close (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'tnx_close_timeline.png'))
    plt.close()
    print(f"Generated and saved: TNX Close Timeline to {os.path.join(plots_dir, 'tnx_close_timeline.png')}")
else:
    print("Skipping TNX Close Timeline plot: 'TNX_Close' column missing or has insufficient data after cleaning.")
    plt.close()


print("\n--- Note on Confusion Matrix ---")
print("A Confusion Matrix is typically used for evaluating classification models, where you predict discrete categories (e.g., 'stock goes up' or 'stock goes down').")
print("For stock price forecasting (predicting the exact future price), which is usually a regression problem, a Confusion Matrix is not directly applicable.")
print("If you later transform your forecasting problem into a classification problem (e.g., predicting daily price movement direction), then a Confusion Matrix would be a relevant visualization.")

print("\n--- Data Prepped for ML Forecasting ---")
if not df.empty:
    print("The DataFrame 'df' is now cleaned and ready for feature engineering and ML model training.")
else:
    print("The DataFrame 'df' is empty. Cannot proceed with ML Forecasting.")


# --- Save Cleaned and Processed Data ---
if not df.empty:
    os.makedirs(input_dir, exist_ok=True) # Ensure data directory exists (redundant but safe)
    df.to_csv(output_processed_filepath, index=False)
    print(f"\nCleaned and processed data saved successfully to '{output_processed_filepath}'")
else:
    print(f"\nDataFrame is empty. Skipping saving processed data to '{output_processed_filepath}'.")

print("\nEDA and Data Cleaning script execution finished.")