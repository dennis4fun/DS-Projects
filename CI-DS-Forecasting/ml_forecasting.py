import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import joblib # For loading models

# --- Configuration ---
input_dir = 'data'
processed_csv_file = 'processed_data.csv'
input_filepath = os.path.join(input_dir, processed_csv_file)

# Define the directory where models and plots are saved
ml_artifacts_dir = 'ml_artifacts'
model_file = 'stock_forecaster_model.joblib'
scaler_file = 'scaler.joblib'
features_list_file = 'features.joblib' # NEW: File to load feature names
output_future_forecast_plot = 'future_stock_price_forecast.png' # Plot for future forecast
output_future_only_plot = 'future_forecast_only.png' # NEW: Plot for future forecast only

# --- Ensure Output Directory Existss ---
os.makedirs(ml_artifacts_dir, exist_ok=True)
print(f"Ensured directory '{ml_artifacts_dir}' exists for saving forecasts.")

# --- Load Data ---
# We need historical data to get the last known values for feature engineering for future dates
print(f"Attempting to load processed data from '{input_filepath}' for feature context...")
try:
    df_historical = pd.read_csv(input_filepath, parse_dates=['Date'])
    print("Historical data loaded successfully.")
    df_historical.sort_values(by=['Ticker', 'Date'], inplace=True)
except FileNotFoundError:
    print(f"Error: The file '{input_filepath}' was not found.")
    print("Please ensure 'processed_data.csv' exists in the 'data/' directory.")
    exit()
except Exception as e:
    print(f"An error occurred while loading historical data: {e}")
    exit()

# --- Load Trained Model, Scaler, and Feature Names ---
print("\nAttempting to load trained model, scaler, and feature names...")
model_filepath = os.path.join(ml_artifacts_dir, model_file)
scaler_filepath = os.path.join(ml_artifacts_dir, scaler_file)
features_list_filepath = os.path.join(ml_artifacts_dir, features_list_file) # NEW

try:
    loaded_model = joblib.load(model_filepath)
    loaded_scaler = joblib.load(scaler_filepath)
    loaded_features = joblib.load(features_list_filepath) # NEW: Load feature names
    print("Loaded trained model, scaler, and feature names for future forecasting.")
except FileNotFoundError:
    print(f"Error: Model, scaler, or feature list not found. Please ensure all exist in '{ml_artifacts_dir}'.")
    print("Run the 'model_training.py' script first to train and save them.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the model/scaler/features: {e}")
    exit()

# Use loaded_features directly
features = loaded_features 
print(f"Loaded features for prediction: {features}")


# --- Future Stock Price Forecasting ---
print("\n--- Starting Future Stock Price Forecasting ---")

last_historical_date = df_historical['Date'].max()
forecast_end_date = datetime.date(2025, 12, 31)

future_dates = pd.date_range(start=last_historical_date + pd.Timedelta(days=1),
                             end=forecast_end_date, freq='B')

print(f"Forecasting from {last_historical_date.date()} to {forecast_end_date} for all tickers.")
print(f"Number of future business days to forecast: {len(future_dates)}")

future_forecast_df_list = []

# Determine the maximum look-back window required by features
relevant_window_sizes = []
for f in features:
    if f.startswith('Adj_Close_Lag_') or f.startswith('Volume_Lag_'):
        try:
            relevant_window_sizes.append(int(f.split('_')[-1]))
        except ValueError:
            pass # Not a numerical lag, ignore
    elif f.startswith('MA_') and f.endswith('_Day') and 'Volume' not in f:
        try:
            # For 'MA_X_Day', 'X' is the second to last element after splitting by '_'
            relevant_window_sizes.append(int(f.split('_')[-2]))
        except ValueError:
            pass
    elif f.startswith('Volume_MA_') and f.endswith('_Day'):
        try:
            relevant_window_sizes.append(int(f.split('_')[-2]))
        except ValueError:
            pass


# Add a default minimum if no relevant features are found, otherwise take the max
# Adding a buffer (e.g., +5) for rolling windows and lags to ensure enough history
max_lookback_window = max(relevant_window_sizes) if relevant_window_sizes else 10 # Default to 10
# **Adjusted for better plotting visibility if data is short:**
# Ensure minimum history is at least 15-20 days to allow some initial feature calculation
min_history_required = max(20, max_lookback_window + 5) # Ensure at least 20 days of history for forecasting to start properly
print(f"Determined maximum look-back window for features: {max_lookback_window} days. Requiring {min_history_required} historical points (min_history_required enforced).")


for ticker in df_historical['Ticker'].unique():
    print(f"\nForecasting for ticker: {ticker}")
    
    historical_ticker_df = df_historical[df_historical['Ticker'] == ticker].sort_values(by='Date')
    
    if len(historical_ticker_df) < min_history_required:
        print(f"Warning: Not enough historical data for {ticker} ({len(historical_ticker_df)} days) to satisfy feature look-back ({min_history_required} days). Skipping forecast for this ticker.")
        continue # Skip to the next ticker
        
    # Use actual historical data for the initial fill of recent_adj_closes and recent_volumes
    # Ensure we get enough points based on max_lookback_window + buffer
    recent_adj_closes = historical_ticker_df['Adj Close'].tail(min_history_required).tolist() 
    recent_volumes = historical_ticker_df['Volume'].tail(min_history_required).tolist() 
    
    ticker_forecasts = [] # To store forecasts for the current ticker

    for future_date in future_dates:
        future_features = {}
        
        # Time-based features for the *future_date*
        future_features['Year'] = future_date.year
        future_features['Month'] = future_date.month
        future_features['Day'] = future_date.day
        future_features['DayOfWeek'] = future_date.dayofweek
        future_features['DayOfYear'] = future_date.dayofyear
        future_features['WeekOfYear'] = future_date.isocalendar().week 

        # Lag Features: Use the `recent_adj_closes` and `recent_volumes` lists.
        # Ensure that `recent_adj_closes` and `recent_volumes` always have enough elements
        # by falling back to the earliest available value if the list is too short.
        for i in [1, 2, 3, 5, 7, 10]: # These are the specific lags used in model_training.py
            if len(recent_adj_closes) >= i:
                future_features[f'Adj_Close_Lag_{i}'] = recent_adj_closes[-i]
            else:
                # If history is too short for a lag, use the oldest value available
                future_features[f'Adj_Close_Lag_{i}'] = recent_adj_closes[0] if recent_adj_closes else np.nan

        if len(recent_volumes) >= 1: # Only need lag 1 for volume
            future_features['Volume_Lag_1'] = recent_volumes[-1]
        else:
            future_features['Volume_Lag_1'] = recent_volumes[0] if recent_volumes else np.nan

        # Rolling Averages: Recalculate based on the `recent_adj_closes` (which will include predictions)
        # Use a temporary Series based on the recent history (actuals + predictions)
        temp_ma_adj_close_series = pd.Series(recent_adj_closes)
        future_features['MA_5_Day'] = temp_ma_adj_close_series.rolling(window=5, min_periods=1).mean().iloc[-1]
        future_features['MA_10_Day'] = temp_ma_adj_close_series.rolling(window=10, min_periods=1).mean().iloc[-1]
        
        temp_ma_volume_series = pd.Series(recent_volumes)
        future_features['Volume_MA_5_Day'] = temp_ma_volume_series.rolling(window=5, min_periods=1).mean().iloc[-1]


        # Create a DataFrame for the current future features, ensuring correct column order
        future_features_df = pd.DataFrame([future_features], columns=features)
        
        # Scale the features
        # Convert scaled features back to DataFrame with explicit feature names
        future_features_scaled = pd.DataFrame(loaded_scaler.transform(future_features_df), columns=features, index=future_features_df.index)
        
        # Predict the price
        predicted_price = loaded_model.predict(future_features_scaled)[0]
        
        # Store the forecast
        ticker_forecasts.append({
            'Ticker': ticker,
            'Date': future_date,
            'Predicted_Adj_Close': predicted_price,
            'Type': 'Forecast'
        })
        
        # DEBUG PRINT: Show prediction for current date and ticker
        print(f"  {ticker} - {future_date.strftime('%Y-%m-%d')}: Predicted Price = {predicted_price:.2f}")

        # Update `recent_adj_closes` and `recent_volumes` for the next iteration's lags/MAs
        recent_adj_closes.append(predicted_price)
        # For volume, a simplistic approach is to just repeat the last known volume.
        recent_volumes.append(recent_volumes[-1] if recent_volumes else np.nan) 

    future_forecast_df_list.extend(ticker_forecasts)

future_forecast_df = pd.DataFrame(future_forecast_df_list)

# --- Prepare data for plotting ---
plot_data_parts = []

# 1. Historical Actuals (from df_historical)
historical_actuals_for_plot = df_historical[['Ticker', 'Date', 'Adj Close']].copy()
historical_actuals_for_plot.rename(columns={'Adj Close': 'Predicted_Adj_Close'}, inplace=True) # Rename to align with others
historical_actuals_for_plot['Type'] = 'Actual'
plot_data_parts.append(historical_actuals_for_plot)

# 2. Test Set Actuals and Predictions
# This section is commented out to simplify the plot for debugging the future forecast lines.
# If needed, you can re-enable this section and handle its plotting separately.
# To get the original test set data (features and target) from historical_df for plotting
# Re-apply feature engineering steps to temp_df_full *before* splitting to ensure correct context
# temp_df_full = df_historical.copy()

# Note: The feature engineering here needs to exactly mirror model_training.py
# for ticker in temp_df_full['Ticker'].unique():
#     ticker_mask = (temp_df_full['Ticker'] == ticker)
#     ticker_df_part = temp_df_full.loc[ticker_mask].copy()
#     for i in [1, 2, 3, 5, 7, 10]:
#         ticker_df_part[f'Adj_Close_Lag_{i}'] = ticker_df_part['Adj Close'].shift(i)
#     ticker_df_part['Volume_Lag_1'] = ticker_df_part['Volume'].shift(1)
#     # Update df in place
#     for col in ticker_df_part.columns:
#         if col not in temp_df_full.columns:
#             temp_df_full[col] = pd.NA
#         temp_df_full.loc[ticker_mask, col] = ticker_df_part[col]

# temp_df_full['MA_5_Day'] = temp_df_full.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
# temp_df_full['MA_10_Day'] = temp_df_full.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
# temp_df_full['Volume_MA_5_Day'] = temp_df_full.groupby('Ticker')['Volume'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

# temp_df_full['Year'] = temp_df_full['Date'].dt.year
# temp_df_full['Month'] = temp_df_full['Date'].dt.month
# temp_df_full['Day'] = temp_df_full['Date'].dt.day
# temp_df_full['DayOfWeek'] = temp_df_full['Date'].dt.dayofweek
# temp_df_full['DayOfYear'] = temp_df_full['Date'].dt.dayofyear
# temp_df_full['WeekOfYear'] = temp_df_full['Date'].dt.isocalendar().week.astype(int)
# temp_df_full['Target_Adj_Close'] = temp_df_full.groupby('Ticker')['Adj Close'].shift(-1) # Target needs to be created for dropna context

# Drop NaNs that would have been dropped during training to get matching X_plot/y_plot indices
# temp_df_full.dropna(subset=temp_df_full.columns.difference(['Daily Return']), inplace=True)

# Define X_plot and y_plot from temp_df_full using the loaded_features (which comes from model_training)
# X_plot = temp_df_full[features] # Use loaded_features here
# y_plot = temp_df_full['Target_Adj_Close']

# Re-create the test_indices based on the fully processed temp_df_full
# test_size_ratio = 0.2
# test_indices = []
# for ticker in temp_df_full['Ticker'].unique():
#     ticker_data_indices = temp_df_full.loc[temp_df_full['Ticker'] == ticker].index
#     split_index = int(len(ticker_data_indices) * (1 - test_size_ratio))
#     test_indices.extend(ticker_data_indices[split_index:].tolist())

# Scale test features and predict
# X_test_scaled_for_plot_pred = pd.DataFrame(loaded_scaler.transform(X_plot.loc[test_indices]), columns=features, index=X_plot.loc[test_indices].index)
# y_test_pred_for_plot = loaded_model.predict(X_test_scaled_for_plot_pred)

# Add test predictions
# test_set_pred_for_plot = pd.DataFrame({
#     'Ticker': temp_df_full.loc[test_indices]['Ticker'],
#     'Date': temp_df_full.loc[test_indices]['Date'],
#     'Predicted_Adj_Close': y_test_pred_for_plot, 
#     'Type': 'Predicted (Test Set)'
# })
# plot_data_parts.append(test_set_pred_for_plot)

# Add test actuals (important for comparison)
# test_set_actual_for_plot = pd.DataFrame({
#     'Ticker': temp_df_full.loc[test_indices]['Ticker'],
#     'Date': temp_df_full.loc[test_indices]['Date'],
#     'Predicted_Adj_Close': y_plot.loc[test_indices], 
#     'Type': 'Actual (Test Set)'
# })
# plot_data_parts.append(test_set_actual_for_plot)

# 3. Future Forecasts (from future_forecast_df)
plot_data_parts.append(future_forecast_df)

# Combine all parts for plotting
combined_forecast_df = pd.concat(plot_data_parts, ignore_index=True)
combined_forecast_df['Predicted_Adj_Close'] = pd.to_numeric(combined_forecast_df['Predicted_Adj_Close'], errors='coerce') # Ensure numeric
combined_forecast_df.sort_values(by=['Ticker', 'Date', 'Type'], inplace=True) # Sort by Type for consistent plotting order

# DEBUG PRINTS for combined_forecast_df
print("\n--- Debugging combined_forecast_df before plotting ---")
print("Combined DataFrame Info:")
combined_forecast_df.info()
print("\nCombined DataFrame Head:")
print(combined_forecast_df.head().to_string())
print("\nCombined DataFrame Tail:")
print(combined_forecast_df.tail().to_string())
print("\nUnique Types in Combined DataFrame:", combined_forecast_df['Type'].unique())
print("\nMin/Max Dates in Combined DataFrame:")
print(f"Min Date: {combined_forecast_df['Date'].min()}")
print(f"Max Date: {combined_forecast_df['Date'].max()}")
print("--- End Debugging ---")


# --- Visualization of Combined Forecast (Current Plot) ---
plt.figure(figsize=(18, 9))

# Define line styles for different 'Type' categories
# Simplified as we are only plotting 'Actual' and 'Forecast' for now
line_styles = {
    'Actual': '-',
    'Forecast': '-.'
}

# Define color palette if you want consistent colors for tickers
palette = sns.color_palette('tab10', n_colors=df_historical['Ticker'].nunique()) # Use a distinct palette for tickers

# Explicitly plot each Ticker's data to ensure continuity for each type
for ticker in combined_forecast_df['Ticker'].unique():
    ticker_data = combined_forecast_df[combined_forecast_df['Ticker'] == ticker]
    
    # Get the color for the current ticker
    ticker_color = palette[list(df_historical['Ticker'].unique()).index(ticker)]

    # Plot Actual history
    sns.lineplot(data=ticker_data[ticker_data['Type'] == 'Actual'],
                 x='Date', y='Predicted_Adj_Close', color=ticker_color,
                 linestyle=line_styles['Actual'], linewidth=1.5, markers=False, label=f'{ticker} Actual')
    
    # Plot Future Forecasts
    if not ticker_data[ticker_data['Type'] == 'Forecast'].empty:
        sns.lineplot(data=ticker_data[ticker_data['Type'] == 'Forecast'],
                     x='Date', y='Predicted_Adj_Close', color=ticker_color,
                     linestyle=line_styles['Forecast'], linewidth=1.5, markers=False, label=f'{ticker} Forecast')


plt.title('Historical and Future Stock Price Forecasts (to end of 2025)') # Updated title
plt.xlabel('Date')
plt.ylabel('Adjusted Closing Price (USD)')

# Create a more structured legend with Ticker and Type
# This now only needs to represent 'Actual' and 'Forecast' types, and the tickers.
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='black', lw=1.5, linestyle='-', label='Actual'),
    Line2D([0], [0], color='black', lw=1.5, linestyle='-.', label='Forecast')
]

# Get the existing ticker legend handles/labels (from the loop-based plotting)
current_handles, current_labels = plt.gca().get_legend_handles_labels()

# Separate ticker labels from type labels that are automatically generated by seaborn
# We want to remove the 'Ticker Actual' and 'Ticker Forecast' labels, and only keep unique ticker colors
# The loop above creates labels like "AAPL Actual", "AAPL Forecast"
# We need to extract just the unique ticker parts to make the first legend.
unique_ticker_labels = []
unique_ticker_handles = []
seen_tickers = set()

for h, l in zip(current_handles, current_labels):
    if ' Actual' in l or ' Forecast' in l: # These are the labels generated by the loop
        ticker_name = l.replace(' Actual', '').replace(' Forecast', '')
        if ticker_name not in seen_tickers:
            unique_ticker_labels.append(ticker_name)
            unique_ticker_handles.append(h) # Use the handle for the ticker's color
            seen_tickers.add(ticker_name)
    else: # This path might not be hit if we only plot with explicit labels in the loop
        pass # Handle other labels if any, but for this simplified case, it's just tickers and types

# Sort ticker labels alphabetically for consistent legend order
ticker_labels_sorted, ticker_handles_sorted = zip(*sorted(zip(unique_ticker_labels, unique_ticker_handles)))


# Create the first legend for Tickers (colors)
first_legend = plt.legend(ticker_handles_sorted, ticker_labels_sorted, title='Ticker', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.gca().add_artist(first_legend) # Add the first legend to the figure

# Create the second legend for Types (line styles)
plt.legend(handles=legend_elements, title='Type', bbox_to_anchor=(1.02, 0.7), loc='upper left')


# Set X-axis limits explicitly to ensure full range is shown
# Get the first and last date from the combined data
min_date = combined_forecast_df['Date'].min()
max_date = combined_forecast_df['Date'].max()
plt.xlim(min_date, max_date)


plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for legends
plt.savefig(os.path.join(ml_artifacts_dir, output_future_forecast_plot))
plt.close()
print(f"Generated and saved: Future Stock Price Forecast Plot to {os.path.join(ml_artifacts_dir, output_future_forecast_plot)}")


# --- NEW: Visualization of Future Forecast Only ---
print("\n--- Generating Future Forecast Only Plot ---")
plt.figure(figsize=(15, 7))

# Filter combined_forecast_df to only include 'Forecast' type
future_only_df = combined_forecast_df[combined_forecast_df['Type'] == 'Forecast'].copy()

if not future_only_df.empty:
    # Explicitly plot each Ticker's forecast to ensure continuity
    for ticker in future_only_df['Ticker'].unique():
        ticker_future_data = future_only_df[future_only_df['Ticker'] == ticker].sort_values(by='Date')
        
        # Get the color for the current ticker
        ticker_color = palette[list(df_historical['Ticker'].unique()).index(ticker)]

        sns.lineplot(data=ticker_future_data,
                     x='Date', y='Predicted_Adj_Close', color=ticker_color,
                     linestyle=line_styles['Forecast'], linewidth=1.5, markers=False, label=f'{ticker} Forecast')

    plt.title('Future Stock Price Forecasts (Current Date to End of 2025)')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Closing Price (USD)')

    # Set X-axis limits to cover only the forecast period
    # Use the `future_dates` range for precise limits
    plt.xlim(future_dates.min(), future_dates.max())

    # Create a legend for Tickers for this plot
    handles, labels = plt.gca().get_legend_handles_labels()
    # Extract unique ticker labels and their corresponding handles (colors)
    unique_ticker_handles_future = []
    unique_ticker_labels_future = []
    seen_tickers_future = set()
    for h, l in zip(handles, labels):
        ticker_name = l.replace(' Forecast', '')
        if ticker_name not in seen_tickers_future:
            unique_ticker_labels_future.append(ticker_name)
            unique_ticker_handles_future.append(h)
            seen_tickers_future.add(ticker_name)
    
    ticker_labels_sorted_future, ticker_handles_sorted_future = zip(*sorted(zip(unique_ticker_labels_future, unique_ticker_handles_future)))

    plt.legend(ticker_handles_sorted_future, ticker_labels_sorted_future, title='Ticker', bbox_to_anchor=(1.02, 1), loc='upper left')


    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for legend
    plt.savefig(os.path.join(ml_artifacts_dir, output_future_only_plot))
    plt.close()
    print(f"Generated and saved: Future Forecast Only Plot to {os.path.join(ml_artifacts_dir, output_future_only_plot)}")
else:
    print("No future forecast data to plot in 'Future Forecast Only' graph.")


print("\nFuture Forecasting complete.")
print("ML Forecasting script execution finished.")