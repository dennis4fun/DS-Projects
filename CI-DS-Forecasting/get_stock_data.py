import pandas as pd
import yfinance as yf
import datetime
import os
import time
import numpy as np # Import numpy for np.nan

# --- Configuration ---
SP500_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM']

end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=365 * 2) # Fetch data for the last 2 years

# --- Path Correction to ensure output goes to repository root ---
# Get the absolute path of the current script file (e.g., /path/to/DS-Projects/CI-DS-Forecasting/get_stock_data.py)
script_path = os.path.abspath(__file__)
# Get the directory of the script (e.g., /path/to/DS-Projects/CI-DS-Forecasting/)
script_dir = os.path.dirname(script_path)
# Go up one level to reach the repository root (e.g., /path/to/DS-Projects/)
repo_root = os.path.dirname(script_dir)

output_dir = os.path.join(repo_root, 'data') # <--- CORRECTED: Now always points to DS-Projects/data
output_csv_file = 'raw_stock_data.csv'
output_filepath = os.path.join(output_dir, output_csv_file)

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

# Define the expected columns in the final output DataFrame.
# Adding new expected columns for macroeconomic data.
EXPECTED_COLUMNS = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'VIX_Close', 'TNX_Close'] # NEW: Added VIX and TNX

# --- Ensure Output Directory Exists and Delete Old File ---
os.makedirs(output_dir, exist_ok=True)
print(f"Ensured directory '{output_dir}' exists for data storage.")

# Snippet to delete existing raw_stock_data.csv
if os.path.exists(output_filepath):
    os.remove(output_filepath)
    print(f"Deleted existing '{output_filepath}' to ensure a clean run.")

# --- Helper function to fetch single ticker data ---
def fetch_single_ticker_data(ticker, start_date, end_date, max_retries, retry_delay):
    """Fetches historical data for a single ticker with retry logic."""
    retries = 0
    while retries < max_retries:
        try:
            print(f"  Attempting to fetch data for {ticker} (Attempt {retries + 1}/{max_retries})...")
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if df.empty:
                raise ValueError(f"No data returned for {ticker}.")
            
            df.reset_index(inplace=True)
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]
            
            rename_map = {
                'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low',
                'close': 'Close', 'volume': 'Volume', 'adj_close': 'Adj Close'
            }
            df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
            
            df['Ticker'] = ticker
            
            # Ensure only relevant columns from original download are returned, before merging macro data
            # Check for 'Close' or 'Adj Close' for main stock data
            stock_cols = [c for c in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'] if c in df.columns]
            df = df[stock_cols]
            
            print(f"  Successfully fetched data for {ticker}.")
            return df

        except Exception as e:
            retries += 1
            print(f"  Failed to fetch data for {ticker}: {e}")
            if retries < max_retries:
                time.sleep(retry_delay)
            else:
                print(f"  Max retries reached for {ticker}. Skipping.")
                return pd.DataFrame() # Return empty DataFrame on failure

# --- NEW: Helper function to fetch macroeconomic data ---
def fetch_macro_data(tickers, start_date, end_date, max_retries, retry_delay):
    """Fetches macroeconomic data (VIX, TNX) with retry logic."""
    macro_data = pd.DataFrame()
    macro_tickers = {
        '^VIX': 'VIX',  # CBOE Volatility Index
        '^TNX': 'TNX'   # 10 Year Treasury Yield
    }
    
    macro_dfs = []
    for yf_ticker, col_prefix in macro_tickers.items():
        df_macro = fetch_single_ticker_data(yf_ticker, start_date, end_date, max_retries, retry_delay)
        if not df_macro.empty:
            # For VIX and TNX, 'Adj Close' might not exist, use 'Close' instead.
            # Ensure 'Close' exists before attempting to rename.
            if 'Close' in df_macro.columns:
                df_macro = df_macro[['Date', 'Close']].rename(columns={'Close': f'{col_prefix}_Close'})
                macro_dfs.append(df_macro)
            else:
                print(f"  Warning: Neither 'Adj Close' nor 'Close' found for {yf_ticker}. Skipping.")
    
    if macro_dfs:
        # Merge all macroeconomic dataframes on 'Date'
        macro_data = macro_dfs[0]
        for i in range(1, len(macro_dfs)):
            macro_data = pd.merge(macro_data, macro_dfs[i], on='Date', how='outer')
        print(f"Successfully fetched and merged macroeconomic data. Shape: {macro_data.shape}")
        print("Macro data head:\n", macro_data.head().to_string())
    else:
        print("No macroeconomic data fetched successfully.")
        
    return macro_data.sort_values('Date').reset_index(drop=True)


# --- Data Fetching Logic ---
print(f"Starting historical stock data retrieval from {start_date} to {end_date}...")

collected_dfs = []
for ticker in SP500_TICKERS:
    df_stock = fetch_single_ticker_data(ticker, start_date, end_date, MAX_RETRIES, RETRY_DELAY_SECONDS)
    if not df_stock.empty:
        collected_dfs.append(df_stock)

# Fetch macroeconomic data
print("\nStarting macroeconomic data retrieval...")
df_macro = fetch_macro_data(SP500_TICKERS, start_date, end_date, MAX_RETRIES, RETRY_DELAY_SECONDS)


# --- Combine All Collected DataFrames and Merge Macro Data ---
all_stocks_data = pd.DataFrame() 

if collected_dfs:
    print("\nAll individual stock DataFrames collected. Performing final concatenation...")
    all_stocks_data = pd.concat(collected_dfs, ignore_index=True)
    all_stocks_data['Date'] = pd.to_datetime(all_stocks_data['Date'])
    df_macro['Date'] = pd.to_datetime(df_macro['Date'])

    # Merge stock data with macroeconomic data
    if not df_macro.empty:
        print(f"Merging stock data with macroeconomic data (VIX, TNX)...")
        # Use left merge to keep all stock data, and add macro data where available
        all_stocks_data = pd.merge(all_stocks_data, df_macro, on='Date', how='left')
        print(f"Combined DataFrame shape after macro merge: {all_stocks_data.shape}")
        print(f"Combined DataFrame columns after macro merge: {all_stocks_data.columns.tolist()}")
    else:
        print("Macroeconomic data is empty, skipping merge.")

    # Ensure all EXPECTED_COLUMNS are present and in the correct order
    for col in EXPECTED_COLUMNS:
        if col not in all_stocks_data.columns:
            all_stocks_data[col] = np.nan # Use np.nan for numerical columns
            print(f"Warning: Expected column '{col}' not found after merge. Added as NaN.")

    all_stocks_data = all_stocks_data[EXPECTED_COLUMNS]
    all_stocks_data.sort_values(by=['Ticker', 'Date'], inplace=True)

    print(f"Final combined DataFrame shape: {all_stocks_data.shape}")
    print(f"Final combined DataFrame columns: {all_stocks_data.columns.tolist()}")
    print(f"Sample of final combined DataFrame (first 5 rows):\n{all_stocks_data.head().to_string()}")
    print(f"Sample of final combined DataFrame (last 5 rows):\n{all_stocks_data.tail().to_string()}")
else:
    print("\nNo stock data was successfully collected for any ticker. The output CSV file was not created.")


# --- Save Collected Data to CSV ---
if not all_stocks_data.empty:
    all_stocks_data.to_csv(output_filepath, index=False)
    print(f"\nAll collected stock data saved successfully to '{output_filepath}'")
    print(f"Total data points (rows) collected: {len(all_stocks_data)}")
else:
    print("\nNo combined stock data to save. The output CSV file was not created.")

print("\nStock data scraping script execution finished.")