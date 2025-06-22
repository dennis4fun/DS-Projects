import pandas as pd
import numpy as np
import os
import joblib # For saving and loading models
from prophet import Prophet # Import Prophet library

# --- Configuration ---
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
repo_root = os.path.dirname(script_dir)

input_dir = os.path.join(repo_root, 'data')
processed_csv_file = 'processed_data.csv'
input_filepath = os.path.join(input_dir, processed_csv_file)

ml_artifacts_dir = os.path.join(repo_root, 'ml_artifacts')

# --- Ensure Output Directories Exist ---
os.makedirs(input_dir, exist_ok=True)
os.makedirs(ml_artifacts_dir, exist_ok=True)
print(f"Ensured directories '{input_dir}' and '{ml_artifacts_dir}' exist.")

# --- Load Data ---
print(f"Attempting to load processed data from '{input_filepath}'...")
try:
    df = pd.read_csv(input_filepath, parse_dates=['Date'])
    print("Processed data loaded successfully.")
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {df.columns.tolist()}")
    print("Sample of DataFrame (first 5 rows):\n", df.head().to_string())
except FileNotFoundError:
    print(f"Error: The file '{input_filepath}' was not found.")
    print("Please ensure 'processed_data.csv' exists in the 'data/' directory (run data_cleaning_EDA.py first).")
    exit()
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    exit()

if df.empty:
    print("DataFrame is empty after loading. No models will be trained.")
    exit()

# --- Model Training with Prophet (per ticker) ---
print("\nStarting Model Training (Prophet) for each ticker...")

trained_models = {}

# Define regressors. These must be columns in your processed_data.csv.
# 'Adj Close' is the target 'y', so we don't include it here.
# 'Ticker' and 'Date' are for grouping/ds, not regressors.
# We include all numerical columns except 'Adj Close' itself as potential regressors.
# We also want to ensure that these columns are indeed present in the DataFrame
# before adding them as regressors.
all_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
REGRESSOR_COLS = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'VIX_Close', 'TNX_Close'] if col in all_numeric_cols and col != 'Adj Close']

print(f"Regressors identified for Prophet: {REGRESSOR_COLS}")

for ticker in df['Ticker'].unique():
    print(f"\nTraining Prophet model for ticker: {ticker}")
    
    ticker_df = df[df['Ticker'] == ticker].copy()
    
    ticker_df['ds'] = pd.to_datetime(ticker_df['Date'])
    ticker_df['y'] = ticker_df['Adj Close']
    
    # Select only required columns for Prophet plus regressors
    prophet_df = ticker_df[['ds', 'y'] + REGRESSOR_COLS].copy()

    # Drop rows with NaNs in 'y' or regressors for this specific ticker's training
    initial_prophet_df_rows = prophet_df.shape[0]
    prophet_df.dropna(subset=['y'] + REGRESSOR_COLS, inplace=True)
    if prophet_df.shape[0] < initial_prophet_df_rows:
        print(f"  Dropped {initial_prophet_df_rows - prophet_df.shape[0]} rows due to NaNs in target or regressors for {ticker}.")

    if prophet_df.empty:
        print(f"  WARNING: No valid data remaining for {ticker} after dropping NaNs for Prophet. Skipping training for this ticker.")
        continue # Skip to the next ticker

    # Initialize Prophet model with more flexible parameters
    m = Prophet(
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.1,  # Increased for more flexibility in trend changes
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,      # Often not meaningful for daily stock closes
        # Optionally, you can add `interval_width` here for the uncertainty intervals,
        # but it's more about visualization than the mean forecast itself.
        # interval_width=0.90 # Default is 0.80 for 80% CI.
    )
    
    # Add external regressors
    for regressor in REGRESSOR_COLS:
        m.add_regressor(regressor)
        print(f"  Added regressor: {regressor}")

    # Fit the model
    try:
        m.fit(prophet_df)
        
        model_filename = os.path.join(ml_artifacts_dir, f'prophet_model_{ticker}.joblib')
        joblib.dump(m, model_filename)
        trained_models[ticker] = m
        print(f"Prophet model for {ticker} trained and saved to '{model_filename}'")
    except Exception as e:
        print(f"  Error training model for {ticker}: {e}")
        print(f"  Skipping model saving for {ticker} due to training error.")

if not trained_models:
    print("\nNo models were successfully trained. Please check your data and logs.")

print("\nModel training script execution finished.")