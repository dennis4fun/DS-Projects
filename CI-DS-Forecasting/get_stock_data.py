import pandas as pd
import yfinance as yf
import datetime
import os
import time

# --- Configuration ---
SP500_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM']

end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=365 * 2) # Fetch data for the last 2 years

output_dir = 'data'
output_csv_file = 'raw_stock_data.csv'
output_filepath = os.path.join(output_dir, output_csv_file)

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

# Define the expected columns in the final output DataFrame.
# This strictly enforces the schema we want for the final CSV.
EXPECTED_COLUMNS = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']

# --- Ensure Output Directory Exists and Delete Old File ---
os.makedirs(output_dir, exist_ok=True)
print(f"Ensured directory '{output_dir}' exists for data storage.")

# Snippet to delete existing raw_stock_data.csv
if os.path.exists(output_filepath):
    os.remove(output_filepath)
    print(f"Deleted existing '{output_filepath}' to ensure a clean run.")

# --- Data Fetching Logic ---
# Initialize an empty list to store DataFrames for each ticker.
collected_dfs = []

print(f"Starting historical stock data retrieval from {start_date} to {end_date}...")

for ticker in SP500_TICKERS:
    retries = 0
    df = pd.DataFrame() # Initialize df for each ticker for the loop

    while retries < MAX_RETRIES:
        try:
            print(f"\nAttempting to fetch data for ticker: {ticker} (Attempt {retries + 1}/{MAX_RETRIES})...")
            # The FutureWarning indicates auto_adjust is True by default, which should give 'Adj Close'
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if df.empty:
                raise ValueError(f"No data returned for {ticker}. It might be delisted or ticker is wrong.")
            
            # --- Aggressive Column Normalization and Standardization ---
            # Debugging: Print initial columns and their type BEFORE any modification
            print(f"Original columns for {ticker}: {df.columns}")
            print(f"Type of columns for {ticker}: {type(df.columns)}")

            # 1. Reset index to turn 'Date' into a regular column from the index.
            # This step should always be done first if Date is the index.
            df.reset_index(inplace=True)
            
            # 2. **Crucial Fix:** Flatten MultiIndex columns if yfinance returned them in a multi-level way.
            # Based on your output, df.columns is MultiIndex with names=['Price', 'Ticker']
            # where level 0 contains ('Close', 'High', etc.) and level 1 contains ('AAPL', etc.).
            if isinstance(df.columns, pd.MultiIndex):
                # We want the 'Price' level (level 0) for column names.
                df.columns = df.columns.get_level_values(0)
                print(f"Columns for {ticker} after MultiIndex flattening (using level 0): {df.columns.tolist()}")
            else:
                print(f"Columns for {ticker} are already single-level. Current columns: {df.columns.tolist()}")

            # 3. Clean up column names: strip whitespace, replace spaces with underscores, convert to lowercase.
            # Apply this to the (now flattened) column names.
            df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]

            # 4. Rename specific columns to our standard expected names.
            # This map ensures that 'date' becomes 'Date', 'adj_close' becomes 'Adj Close', etc.
            rename_map = {
                'date': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'adj_close': 'Adj Close' # Matches the lowercase, underscore-replaced 'adj_close'
            }
            # Only rename columns that actually exist in the DataFrame after flattening and lowercasing
            df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
            
            # 5. Add 'Ticker' column to identify the stock.
            df['Ticker'] = ticker
            
            # --- Ensure all EXPECTED_COLUMNS are present and in the correct order ---
            # First, filter to only include columns that are in EXPECTED_COLUMNS AND are actually present in df
            cols_to_keep = [col for col in EXPECTED_COLUMNS if col in df.columns]
            df = df[cols_to_keep]

            # Then, re-add any missing EXPECTED_COLUMNS as NaN to ensure consistent schema for pd.concat
            for col in EXPECTED_COLUMNS:
                if col not in df.columns:
                    print(f"Warning: Expected column '{col}' still not found for ticker {ticker}. Adding as NaN.")
                    df[col] = pd.NA # Use pandas' native missing value type for consistency

            # Finally, ensure the exact order of EXPECTED_COLUMNS for the DataFrame
            df = df[EXPECTED_COLUMNS]

            print(f"Columns for {ticker} after final schema enforcement: {df.columns.tolist()}")
            print(f"First 5 rows for {ticker} after final processing:\n{df.head().to_string()}")

            # If successful, add the processed DataFrame to our list and break retry loop.
            collected_dfs.append(df)
            print(f"Successfully processed and added data for {ticker} to list.")
            break # Exit retry loop on success

        except Exception as e:
            retries += 1
            print(f"Failed to fetch data for {ticker}: {e}")
            if retries < MAX_RETRIES:
                print(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                print(f"Max retries reached for {ticker}. Skipping this ticker. Error: {e}")
    
# --- Combine All Collected DataFrames ---
all_stocks_data = pd.DataFrame() # Re-initialize as empty before final concat

if collected_dfs: # Check if the list of DataFrames is not empty
    print("\nAll individual stock DataFrames collected. Performing final concatenation...")
    # Concatenate all DataFrames in the list vertically.
    # ignore_index=True ensures a clean, continuous index for the combined DataFrame.
    all_stocks_data = pd.concat(collected_dfs, ignore_index=True) 
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
