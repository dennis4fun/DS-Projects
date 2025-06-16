import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import joblib # For loading models
from prophet import Prophet # Import Prophet library

# --- Configuration ---
input_dir = 'data'
processed_csv_file = 'processed_data.csv'
input_filepath = os.path.join(input_dir, processed_csv_file)

# Define the directory where models and plots are saved
ml_artifacts_dir = 'ml_artifacts'
output_historical_future_forecast_plot = 'historical_and_future_forecast.png' # Plot for historical + future forecast
output_future_only_plot = 'future_forecast_only.png' # Plot for future forecast only
output_quarterly_changes_plot = 'quarterly_forecast_changes.png' # NEW: Plot for quarterly changes

# --- Ensure Output Directory Exists ---
os.makedirs(ml_artifacts_dir, exist_ok=True)
print(f"Ensured directory '{ml_artifacts_dir}' exists for saving forecasts.")

# --- Load Data ---
# We need historical data to get the last known values for Prophet to extend from
print(f"Attempting to load processed data from '{input_filepath}' for historical context...")
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

# --- Load Trained Prophet Models ---
print("\nAttempting to load trained Prophet models...")
loaded_models = {}
for ticker in df_historical['Ticker'].unique():
    model_filepath = os.path.join(ml_artifacts_dir, f'prophet_model_{ticker}.joblib')
    try:
        loaded_model = joblib.load(model_filepath)
        loaded_models[ticker] = loaded_model
        print(f"Loaded Prophet model for {ticker}.")
    except FileNotFoundError:
        print(f"Error: Model for {ticker} not found at '{model_filepath}'. Please ensure 'model_training.py' was run successfully.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the model for {ticker}: {e}")
        exit()

if not loaded_models:
    print("No models were loaded. Exiting forecasting script.")
    exit()

# --- Future Stock Price Forecasting with Prophet ---
print("\n--- Starting Future Stock Price Forecasting with Prophet ---")

last_historical_date = df_historical['Date'].max()
forecast_end_date = datetime.date(2025, 12, 31)

future_dates = pd.date_range(start=last_historical_date + pd.Timedelta(days=1),
                             end=forecast_end_date, freq='B')

print(f"Forecasting from {last_historical_date.date()} to {forecast_end_date} ({len(future_dates)} business days) for all tickers.")

future_forecast_df_list = []

for ticker, model in loaded_models.items():
    print(f"\nGenerating forecast for ticker: {ticker}")
    
    ticker_historical_df = df_historical[df_historical['Ticker'] == ticker].copy()
    ticker_historical_df['ds'] = pd.to_datetime(ticker_historical_df['Date'])
    ticker_historical_df['y'] = ticker_historical_df['Adj Close'] 

    # Create future DataFrame for forecasting
    future_prophet_df = model.make_future_dataframe(periods=len(future_dates), freq='B', include_history=False)
    
    forecast = model.predict(future_prophet_df)
    
    # Extract predicted values AND confidence intervals
    ticker_forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
        columns={'ds': 'Date', 'yhat': 'Predicted_Adj_Close'}
    )
    ticker_forecast_data['Ticker'] = ticker
    ticker_forecast_data['Type'] = 'Forecast'
    
    future_forecast_df_list.append(ticker_forecast_data)
    
    print(f"  {ticker} forecast generated for {len(ticker_forecast_data)} periods. Sample tail:\n{ticker_forecast_data.tail().to_string()}")

future_forecast_df = pd.concat(future_forecast_df_list, ignore_index=True)


# --- Prepare data for plotting ---
plot_data_parts = []

# 1. Historical Actuals (from df_historical)
historical_actuals_for_plot = df_historical[['Ticker', 'Date', 'Adj Close']].copy()
historical_actuals_for_plot.rename(columns={'Adj Close': 'Predicted_Adj_Close'}, inplace=True)
historical_actuals_for_plot['Type'] = 'Actual'
# For historical data, yhat_lower and yhat_upper are not applicable, fill with NaN
historical_actuals_for_plot['yhat_lower'] = np.nan 
historical_actuals_for_plot['yhat_upper'] = np.nan
plot_data_parts.append(historical_actuals_for_plot)

# 2. Future Forecasts (from future_forecast_df)
plot_data_parts.append(future_forecast_df)

# Combine all parts for plotting
combined_forecast_df = pd.concat(plot_data_parts, ignore_index=True)
combined_forecast_df['Predicted_Adj_Close'] = pd.to_numeric(combined_forecast_df['Predicted_Adj_Close'], errors='coerce')
combined_forecast_df.sort_values(by=['Ticker', 'Date', 'Type'], inplace=True)


# --- Debugging combined_forecast_df before plotting ---
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


# --- Visualization 1: Historical and Future Forecast ---
plt.figure(figsize=(18, 9)) # Slightly adjusted figure size for better readability

line_styles = {
    'Actual': '-',
    'Forecast': '-.'
}

# Using a different seaborn palette
palette = sns.color_palette('tab10', n_colors=df_historical['Ticker'].nunique()) 


for ticker in combined_forecast_df['Ticker'].unique():
    ticker_data = combined_forecast_df[combined_forecast_df['Ticker'] == ticker]
    ticker_color = palette[list(df_historical['Ticker'].unique()).index(ticker)]

    # Plot Actual history
    sns.lineplot(data=ticker_data[ticker_data['Type'] == 'Actual'],
                 x='Date', y='Predicted_Adj_Close', color=ticker_color,
                 linestyle=line_styles['Actual'], linewidth=2, # Increased linewidth
                 markers=False, label=f'{ticker} Actual')
    
    # Plot Future Forecasts
    forecast_data_ticker = ticker_data[ticker_data['Type'] == 'Forecast']
    if not forecast_data_ticker.empty:
        sns.lineplot(data=forecast_data_ticker,
                     x='Date', y='Predicted_Adj_Close', color=ticker_color,
                     linestyle=line_styles['Forecast'], linewidth=2, # Increased linewidth
                     markers=False, label=f'{ticker} Forecast')
        
        # Plot confidence interval as shaded region
        plt.fill_between(forecast_data_ticker['Date'], 
                         forecast_data_ticker['yhat_lower'], 
                         forecast_data_ticker['yhat_upper'], 
                         color=ticker_color, alpha=0.15, label=f'{ticker} Forecast Interval')

# Add a vertical line at the forecast start date for clarity
plt.axvline(x=last_historical_date, color='gray', linestyle='--', linewidth=1.5, label='Forecast Start')


plt.title('Stock Price Historical Data and Prophet Forecasts (with Confidence Interval)', fontsize=16) # More descriptive title
plt.xlabel('Date', fontsize=12)
plt.ylabel('Adjusted Closing Price (USD)', fontsize=12)

# Create a more structured legend with Ticker, Type, and Forecast Interval
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_elements = [
    Line2D([0], [0], color='black', lw=2, linestyle='-', label='Actual Price'),
    Line2D([0], [0], color='black', lw=2, linestyle='-.', label='Forecasted Price'),
    Patch(facecolor='gray', edgecolor='black', alpha=0.15, label='Forecast Confidence Interval')
]

# Get the existing ticker legend handles/labels (from the loop-based plotting)
current_handles, current_labels = plt.gca().get_legend_handles_labels()

unique_ticker_labels = []
unique_ticker_handles = []
seen_tickers = set()

for h, l in zip(current_handles, current_labels):
    # This logic needs to correctly extract only the ticker label part
    # And ensure we only add each ticker once for its color entry in the legend
    if ' Actual' in l or ' Forecast' in l:
        ticker_name = l.replace(' Actual', '').replace(' Forecast', '')
        if ticker_name not in seen_tickers:
            unique_ticker_labels.append(ticker_name)
            unique_ticker_handles.append(h)
            seen_tickers.add(ticker_name)
    elif 'Forecast Interval' in l: # Skip adding interval labels from automatic generation
        pass
    else: 
        pass 

ticker_labels_sorted, ticker_handles_sorted = zip(*sorted(zip(unique_ticker_labels, unique_ticker_handles)))


# Create the first legend for Tickers (colors)
first_legend = plt.legend(ticker_handles_sorted, ticker_labels_sorted, title='Ticker', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='medium', title_fontsize='large')
plt.gca().add_artist(first_legend) 

# Create the second legend for Types (line styles and shaded area)
# Filter out handles from the ticker legend if they were auto-generated for types
plt.legend(handles=legend_elements, title='Line Type', bbox_to_anchor=(1.02, 0.6), loc='upper left', fontsize='medium', title_fontsize='large')


# Set X-axis limits explicitly to ensure full range is shown
plt.xlim(combined_forecast_df['Date'].min(), combined_forecast_df['Date'].max())

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for legends
plt.savefig(os.path.join(ml_artifacts_dir, output_historical_future_forecast_plot))
plt.close()
print(f"Generated and saved: Historical and Future Stock Price Forecast Plot to {os.path.join(ml_artifacts_dir, output_historical_future_forecast_plot)}")


# --- Visualization 2: Future Forecast Only ---
print("\n--- Generating Future Forecast Only Plot ---")
plt.figure(figsize=(15, 7)) # Adjust figure size

if not future_forecast_df.empty:
    for ticker in future_forecast_df['Ticker'].unique():
        ticker_future_data = future_forecast_df[future_forecast_df['Ticker'] == ticker].sort_values(by='Date')
        ticker_color = palette[list(df_historical['Ticker'].unique()).index(ticker)]

        sns.lineplot(data=ticker_future_data,
                     x='Date', y='Predicted_Adj_Close', color=ticker_color,
                     linestyle=line_styles['Forecast'], linewidth=2, # Increased linewidth
                     markers=False, label=f'{ticker} Forecast')
        
        # Plot confidence interval
        plt.fill_between(ticker_future_data['Date'], 
                         ticker_future_data['yhat_lower'], 
                         ticker_future_data['yhat_upper'], 
                         color=ticker_color, alpha=0.15, label=f'{ticker} Forecast Interval')


    plt.title('Future Stock Price Forecasts (Current Date to End of 2025) with Confidence Interval', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Adjusted Closing Price (USD)', fontsize=12)

    # Set X-axis limits to cover only the forecast period
    plt.xlim(future_dates.min(), future_dates.max())

    # Dynamically set Y-axis limits to zoom into the forecast range
    min_y_forecast = future_forecast_df['Predicted_Adj_Close'].min()
    max_y_forecast = future_forecast_df['Predicted_Adj_Close'].max()
    y_buffer = (max_y_forecast - min_y_forecast) * 0.1 # 10% buffer
    plt.ylim(min_y_forecast - y_buffer, max_y_forecast + y_buffer)


    # Create a legend for Tickers and Type for this plot
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_ticker_handles_future = []
    unique_ticker_labels_future = []
    seen_tickers_future = set()
    for h, l in zip(handles, labels):
        # Extract ticker name from labels like "AAPL Forecast" or "AAPL Forecast Interval"
        if ' Forecast' in l:
            ticker_name = l.replace(' Forecast', '')
            if ' Interval' in ticker_name: # Handle "AAPL Forecast Interval"
                ticker_name = ticker_name.replace(' Interval', '')
            if ticker_name not in seen_tickers_future:
                unique_ticker_labels_future.append(ticker_name)
                unique_ticker_handles_future.append(h)
                seen_tickers_future.add(ticker_name)
        else:
            pass # Ignore other labels if any

    ticker_labels_sorted_future, ticker_handles_sorted_future = zip(*sorted(zip(unique_ticker_labels_future, unique_ticker_handles_future)))

    # Combined legend for this "Future Only" plot
    legend_elements_future_only = [
        Line2D([0], [0], color='black', lw=2, linestyle='-.', label='Forecasted Price'),
        Patch(facecolor='gray', edgecolor='black', alpha=0.15, label='Forecast Confidence Interval')
    ]

    # Create the first legend for Tickers (colors)
    first_legend_future = plt.legend(ticker_handles_sorted_future, ticker_labels_sorted_future, title='Ticker', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='medium', title_fontsize='large')
    plt.gca().add_artist(first_legend_future) 

    # Create the second legend for Types (line styles and shaded area)
    plt.legend(handles=legend_elements_future_only, title='Line Type', bbox_to_anchor=(1.02, 0.6), loc='upper left', fontsize='medium', title_fontsize='large')


    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(os.path.join(ml_artifacts_dir, output_future_only_plot))
    plt.close()
    print(f"Generated and saved: Future Forecast Only Plot to {os.path.join(ml_artifacts_dir, output_future_only_plot)}")
else:
    print("No future forecast data to plot in 'Future Forecast Only' graph.")


# --- NEW Visualization 3: Quarterly Forecast Changes ---
print("\n--- Generating Quarterly Forecast Changes Plot ---")
quarterly_changes_list = []

# Define the end dates for the remaining quarters of 2025
# Current date is assumed to be mid-June 2025 based on previous context.
# Q2 ends June 30, Q3 ends Sept 30, Q4 ends Dec 31
q2_end_date = datetime.date(2025, 6, 30)
q3_end_date = datetime.date(2025, 9, 30)
q4_end_date = datetime.date(2025, 12, 31)

# Find the last actual historical price for each ticker as the baseline for Q2 change
last_actual_prices = df_historical.groupby('Ticker')['Adj Close'].last().reset_index()
last_actual_prices.rename(columns={'Adj Close': 'Start_Price'}, inplace=True)

for ticker in future_forecast_df['Ticker'].unique():
    ticker_forecast_data = future_forecast_df[future_forecast_df['Ticker'] == ticker].sort_values(by='Date')
    
    # Get prices at quarter end dates (or closest available business day)
    # Use ffill to get the last known forecast up to that date, then bfill for next known
    q2_price = ticker_forecast_data.loc[ticker_forecast_data['Date'] <= pd.Timestamp(q2_end_date)]['Predicted_Adj_Close'].iloc[-1] if not ticker_forecast_data.loc[ticker_forecast_data['Date'] <= pd.Timestamp(q2_end_date)].empty else np.nan
    q3_price = ticker_forecast_data.loc[ticker_forecast_data['Date'] <= pd.Timestamp(q3_end_date)]['Predicted_Adj_Close'].iloc[-1] if not ticker_forecast_data.loc[ticker_forecast_data['Date'] <= pd.Timestamp(q3_end_date)].empty else np.nan
    q4_price = ticker_forecast_data.loc[ticker_forecast_data['Date'] <= pd.Timestamp(q4_end_date)]['Predicted_Adj_Close'].iloc[-1] if not ticker_forecast_data.loc[ticker_forecast_data['Date'] <= pd.Timestamp(q4_end_date)].empty else np.nan

    # Get the last historical price for this ticker
    start_q2_price = last_actual_prices[last_actual_prices['Ticker'] == ticker]['Start_Price'].iloc[0] if not last_actual_prices[last_actual_prices['Ticker'] == ticker].empty else np.nan

    # Calculate changes
    if not np.isnan(start_q2_price) and not np.isnan(q2_price):
        q2_change = ((q2_price - start_q2_price) / start_q2_price) * 100
        quarterly_changes_list.append({'Ticker': ticker, 'Quarter': 'Q2 2025', 'Change (%)': q2_change})
    
    if not np.isnan(q2_price) and not np.isnan(q3_price):
        q3_change = ((q3_price - q2_price) / q2_price) * 100
        quarterly_changes_list.append({'Ticker': ticker, 'Quarter': 'Q3 2025', 'Change (%)': q3_change})

    if not np.isnan(q3_price) and not np.isnan(q4_price):
        q4_change = ((q4_price - q3_price) / q3_price) * 100
        quarterly_changes_list.append({'Ticker': ticker, 'Quarter': 'Q4 2025', 'Change (%)': q4_change})

quarterly_changes_df = pd.DataFrame(quarterly_changes_list)

if not quarterly_changes_df.empty:
    plt.figure(figsize=(14, 8)) # Adjusted for horizontal bars
    
    # Sort for better visualization
    quarterly_changes_df['Abs_Change'] = quarterly_changes_df['Change (%)'].abs()
    quarterly_changes_df.sort_values(by=['Quarter', 'Abs_Change'], ascending=[True, False], inplace=True)

    # Define colors for positive/negative change
    colors = ['green' if x >= 0 else 'red' for x in quarterly_changes_df['Change (%)']]
    
    sns.barplot(data=quarterly_changes_df, x='Change (%)', y='Ticker', hue='Quarter', palette='viridis', dodge=True)
    
    plt.title('Projected Quarterly Stock Price Changes (Q2-Q4 2025)', fontsize=16)
    plt.xlabel('Projected Change (%)', fontsize=12)
    plt.ylabel('Ticker', fontsize=12)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8) # Line at 0% change
    
    # Add percentage labels to bars
    for container in plt.gca().containers:
        plt.bar_label(container, fmt='%.2f%%', label_type='edge', padding=3)

    plt.legend(title='Quarter', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(os.path.join(ml_artifacts_dir, output_quarterly_changes_plot))
    plt.close()
    print(f"Generated and saved: Quarterly Forecast Changes Plot to {os.path.join(ml_artifacts_dir, output_quarterly_changes_plot)}")
else:
    print("No quarterly forecast data to plot in 'Quarterly Forecast Changes' graph.")

print("\nFuture Forecasting complete.")
print("ML Forecasting script execution finished.")
