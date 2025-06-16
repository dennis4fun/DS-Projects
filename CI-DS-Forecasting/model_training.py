import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # For saving and loading models
from prophet import Prophet # Import Prophet library

# --- Configuration ---
input_dir = 'data'
processed_csv_file = 'processed_data.csv'
input_filepath = os.path.join(input_dir, processed_csv_file)

# Define the directory for saving models
ml_artifacts_dir = 'ml_artifacts'

# --- Ensure Output Directories Exist ---
os.makedirs(input_dir, exist_ok=True) # Ensure data dir exists
os.makedirs(ml_artifacts_dir, exist_ok=True) # Ensure ML artifacts dir exists
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

# --- Model Training with Prophet (per ticker) ---
print("\nStarting Model Training (Prophet) for each ticker...")

# Prophet requires columns named 'ds' (datestamp) and 'y' (value)
# We will train a separate model for each stock ticker.
trained_models = {}

for ticker in df['Ticker'].unique():
    print(f"\nTraining Prophet model for ticker: {ticker}")
    
    # Prepare data for Prophet: 'ds' for Date, 'y' for 'Adj Close'
    ticker_df = df[df['Ticker'] == ticker].copy()
    
    # Ensure 'Date' column is datetime and sorted for Prophet
    ticker_df['ds'] = pd.to_datetime(ticker_df['Date'])
    ticker_df['y'] = ticker_df['Adj Close']
    
    # Select only required columns for Prophet
    prophet_df = ticker_df[['ds', 'y']]

    # Initialize Prophet model
    # You can add more parameters here to tune Prophet, e.g.,
    # seasonality_mode='multiplicative', changepoint_prior_scale=0.05,
    # daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True
    m = Prophet(seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05) # Increased flexibility for changepoints
    
    # Add external regressors if desired (e.g., Volume, Open, High, Low, Close)
    # This aligns with some of your previous features, allowing Prophet to use them.
    # Note: Prophet expects these regressors to be present in both historical and future dataframes.
    
    # We will add them to Prophet's "extra regressors"
    # Ensure these are aligned with the columns available in your processed_data.csv
    # For now, let's include 'Volume', 'Open', 'High', 'Low', 'Close' as extra regressors.
    # These should be present and valid in your 'processed_data.csv' for all dates.
    # Note: Prophet will use the *actual* values of these regressors from your historical data.
    # For future forecasting, you'll need to provide *future* values for these regressors.
    # For simplicity, we'll start without extra regressors in this first Prophet implementation.
    # If you later want to add them, you'd uncomment and extend this section.
    
    # Example of adding a regressor (uncomment if you want to include other features)
    # m.add_regressor('Volume') 
    # m.add_regressor('Open')
    # m.add_regressor('High')
    # m.add_regressor('Low')
    # m.add_regressor('Close')


    # Fit the model
    m.fit(prophet_df)
    
    # Save the trained Prophet model
    model_filename = os.path.join(ml_artifacts_dir, f'prophet_model_{ticker}.joblib')
    joblib.dump(m, model_filename)
    trained_models[ticker] = m
    print(f"Prophet model for {ticker} trained and saved to '{model_filename}'")

print("\nModel training script execution finished.")
