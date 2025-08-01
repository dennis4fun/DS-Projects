import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import joblib # For loading models
from prophet import Prophet # Import Prophet library

# --- Configuration ---
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
repo_root = os.path.dirname(script_dir)

input_dir = os.path.join(repo_root, 'data')
processed_csv_file = 'processed_data.csv'
input_filepath = os.path.join(input_dir, processed_csv_file)

ml_artifacts_dir = os.path.join(repo_root, 'ml_artifacts')

output_historical_future_forecast_plot = 'historical_and_future_forecast.png'
output_future_only_plot = 'future_forecast_only.png'
output_quarterly_changes_plot = 'quarterly_forecast_changes.png'

# Define regressors, consistent with model_training.py
REGRESSOR_COLS = ['Open', 'High', 'Low', 'Close', 'Volume', 'VIX_Close', 'TNX_Close']

# --- Ensure Output Directory Exists ---
os.makedirs(ml_artifacts_dir, exist_ok=True)
print(f"Ensured directory '{ml_artifacts_dir}' exists for saving models and plots.")

# --- Load Data ---
print(f"Attempting to load processed data from '{input_filepath}' for historical context...")
try:
    df_historical = pd.read_csv(input_filepath, parse_dates=['Date'])
    print("Historical data loaded successfully.")
    df_historical.sort_values(by=['Ticker', 'Date'], inplace=True)
    print(f"Historical DataFrame shape: {df_historical.shape}")
    print(f"Historical DataFrame columns: {df_historical.columns.tolist()}")
    print("Sample of Historical DataFrame (last 5 rows):\n", df_historical.tail().to_string())
except FileNotFoundError:
    print(f"Error: The file '{input_filepath}' was not found.")
    print("Please ensure 'processed_data.csv' exists in the 'data/' directory.")
    exit()
except Exception as e:
    print(f"An error occurred while loading historical data: {e}")
    exit()

if df_historical.empty:
    print("Historical DataFrame is empty after loading. Cannot proceed with forecasting.")
    exit()

# Filter REGRESSOR_COLS to only include those present in df_historical
REGRESSOR_COLS_PRESENT = [col for col in REGRESSOR_COLS if col in df_historical.columns]
if len(REGRESSOR_COLS_PRESENT) < len(REGRESSOR_COLS):
    missing_regressors = set(REGRESSOR_COLS) - set(REGRESSOR_COLS_PRESENT)
    print(f"Warning: The following regressors were not found in historical data and will be skipped: {list(missing_regressors)}")
REGRESSOR_COLS = REGRESSOR_COLS_PRESENT


# --- Load Trained Prophet Models ---
print("\nAttempting to load trained Prophet models...")
loaded_models = {}
unique_tickers_in_data = df_historical['Ticker'].unique()

if not unique_tickers_in_data.size > 0:
    print("No unique tickers found in historical data. Cannot load models.")
    exit()

for ticker in unique_tickers_in_data:
    model_filepath = os.path.join(ml_artifacts_dir, f'prophet_model_{ticker}.joblib')
    try:
        loaded_model = joblib.load(model_filepath)
        loaded_models[ticker] = loaded_model
        print(f"Loaded Prophet model for {ticker}.")
    except FileNotFoundError:
        print(f"Error: Model for {ticker} not found at '{model_filepath}'. Please ensure 'model_training.py' was run successfully and created models for all tickers.")
    except Exception as e:
        print(f"An error occurred while loading the model for {ticker}: {e}")

if not loaded_models:
    print("No models were loaded. Exiting forecasting script.")
    exit()

# --- Future Stock Price Forecasting with Prophet ---
print("\n--- Starting Future Stock Price Forecasting with Prophet ---")

last_historical_date = df_historical['Date'].max()
forecast_end_date = datetime.date(2025, 12, 31)

future_dates = pd.date_range(start=last_historical_date + pd.Timedelta(days=1),
                             end=forecast_end_date, freq='B')

if future_dates.empty:
    print(f"No future business days to forecast between {last_historical_date.date()} and {forecast_end_date}. Forecasting will be skipped.")
    future_forecast_df = pd.DataFrame(columns=['Date', 'Predicted_Adj_Close', 'yhat_lower', 'yhat_upper', 'Ticker', 'Type'])
else:
    print(f"Forecasting from {last_historical_date.date()} to {forecast_end_date} for all tickers.")
    print(f"Number of future business days to forecast: {len(future_dates)}")

    future_forecast_df_list = []

    ROLLING_WINDOW_DAYS = 20 # Use last 20 trading days for a rolling average of regressors

    for ticker, model in loaded_models.items():
        print(f"\nGenerating forecast for ticker: {ticker}")
        
        ticker_historical_df = df_historical[df_historical['Ticker'] == ticker].copy()
        
        future_prophet_df = model.make_future_dataframe(periods=len(future_dates), freq='B', include_history=False)
        
        # --- Populate future regressors for prediction with rolling averages ---
        # Get the recent historical data to calculate rolling averages
        recent_historical = ticker_historical_df.tail(ROLLING_WINDOW_DAYS).copy()
        
        for col in REGRESSOR_COLS:
            if col in recent_historical.columns:
                # Calculate rolling mean for the historical period
                rolling_mean_val = recent_historical[col].rolling(window=min(ROLLING_WINDOW_DAYS, len(recent_historical)), min_periods=1).mean().iloc[-1]
                future_prophet_df[col] = rolling_mean_val
            else:
                future_prophet_df[col] = np.nan
        
        future_prophet_df[REGRESSOR_COLS] = future_prophet_df[REGRESSOR_COLS].fillna(0)

        # Predict
        try:
            forecast = model.predict(future_prophet_df)
            
            ticker_forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
                columns={'ds': 'Date', 'yhat': 'Predicted_Adj_Close'}
            )
            ticker_forecast_data['Ticker'] = ticker
            ticker_forecast_data['Type'] = 'Forecast'
            
            future_forecast_df_list.append(ticker_forecast_data)
            
            print(f"  {ticker} forecast generated for {len(ticker_forecast_data)} periods. Sample tail:\n{ticker_forecast_data.tail().to_string()}")
        except Exception as e:
            print(f"  Error generating forecast for {ticker}: {e}")
            print(f"  Skipping forecast for {ticker}.")

    if not future_forecast_df_list:
        print("No forecasts were generated for any ticker. Exiting forecasting script.")
        exit()

    future_forecast_df = pd.concat(future_forecast_df_list, ignore_index=True)


# --- Prepare data for plotting ---
plot_data_parts = []

historical_actuals_for_plot = df_historical[['Ticker', 'Date', 'Adj Close']].copy()
historical_actuals_for_plot.rename(columns={'Adj Close': 'Predicted_Adj_Close'}, inplace=True)
historical_actuals_for_plot['Type'] = 'Actual'
historical_actuals_for_plot['yhat_lower'] = np.nan 
historical_actuals_for_plot['yhat_upper'] = np.nan
plot_data_parts.append(historical_actuals_for_plot)

plot_data_parts.append(future_forecast_df)

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
plt.figure(figsize=(18, 9))

line_styles = {
    'Actual': '-',
    'Forecast': '-.'
}

unique_tickers = combined_forecast_df['Ticker'].unique()
palette = sns.color_palette('tab10', n_colors=len(unique_tickers)) 
ticker_color_map = {ticker: palette[i] for i, ticker in enumerate(unique_tickers)}


for ticker in unique_tickers:
    ticker_data = combined_forecast_df[combined_forecast_df['Ticker'] == ticker]
    ticker_color = ticker_color_map[ticker]

    sns.lineplot(data=ticker_data[ticker_data['Type'] == 'Actual'],
                 x='Date', y='Predicted_Adj_Close', color=ticker_color,
                 linestyle=line_styles['Actual'], linewidth=2, 
                 markers=False, label=f'{ticker} Actual')
    
    forecast_data_ticker = ticker_data[ticker_data['Type'] == 'Forecast']
    if not forecast_data_ticker.empty:
        sns.lineplot(data=forecast_data_ticker,
                     x='Date', y='Predicted_Adj_Close', color=ticker_color,
                     linestyle=line_styles['Forecast'], linewidth=2, 
                     markers=False, label=f'{ticker} Forecast')
        
        # Emphasize confidence interval to show potential range
        plt.fill_between(forecast_data_ticker['Date'], 
                         forecast_data_ticker['yhat_lower'], 
                         forecast_data_ticker['yhat_upper'], 
                         color=ticker_color, alpha=0.25, label=f'{ticker} Forecast Interval') # Increased alpha
        # Add thin line at yhat_lower and yhat_upper for better visibility of boundaries
        plt.plot(forecast_data_ticker['Date'], forecast_data_ticker['yhat_lower'], color=ticker_color, linestyle=':', linewidth=0.8, alpha=0.7)
        plt.plot(forecast_data_ticker['Date'], forecast_data_ticker['yhat_upper'], color=ticker_color, linestyle=':', linewidth=0.8, alpha=0.7)


plt.axvline(x=last_historical_date, color='gray', linestyle='--', linewidth=1.5, label='Forecast Start')


plt.title('Stock Price Historical Data and Prophet Forecasts (with Confidence Interval)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Adjusted Closing Price (USD)', fontsize=12)

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_elements = [
    Line2D([0], [0], color='black', lw=2, linestyle='-', label='Actual Price'),
    Line2D([0], [0], color='black', lw=2, linestyle='-.', label='Forecasted Price'),
    Patch(facecolor='gray', edgecolor='black', alpha=0.25, label='Forecast Confidence Interval'),
    Line2D([0], [0], color='gray', lw=0.8, linestyle=':', label='Confidence Bounds')
]

current_handles, current_labels = plt.gca().get_legend_handles_labels()

unique_ticker_legend_handles = []
unique_ticker_legend_labels = []
seen_tickers_in_legend = set()

for h, l in zip(current_handles, current_labels):
    if ' Actual' in l or ' Forecast' in l:
        ticker_name = l.split(' ')[0]
        if ticker_name not in seen_tickers_in_legend:
            unique_ticker_legend_labels.append(ticker_name)
            unique_ticker_legend_handles.append(h)
            seen_tickers_in_legend.add(ticker_name)

sorted_ticker_info = sorted(zip(unique_ticker_legend_labels, unique_ticker_legend_handles), key=lambda x: x[0])
ticker_labels_sorted = [item[0] for item in sorted_ticker_info]
ticker_handles_sorted = [item[1] for item in sorted_ticker_info]

first_legend = plt.legend(ticker_handles_sorted, ticker_labels_sorted, title='Ticker', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='medium', title_fontsize='large')
plt.gca().add_artist(first_legend) 

plt.legend(handles=legend_elements, title='Line Type', bbox_to_anchor=(1.02, 0.6), loc='upper left', fontsize='medium', title_fontsize='large')


plt.xlim(combined_forecast_df['Date'].min(), combined_forecast_df['Date'].max())

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig(os.path.join(ml_artifacts_dir, output_historical_future_forecast_plot))
plt.close()
print(f"Generated and saved: Historical and Future Stock Price Forecast Plot to {os.path.join(ml_artifacts_dir, output_historical_future_forecast_plot)}")


# --- Visualization 2: Future Forecast Only ---
print("\n--- Generating Future Forecast Only Plot ---")
plt.figure(figsize=(15, 7))

if not future_forecast_df.empty:
    for ticker in unique_tickers:
        ticker_future_data = future_forecast_df[future_forecast_df['Ticker'] == ticker].sort_values(by='Date')
        ticker_color = ticker_color_map[ticker]

        sns.lineplot(data=ticker_future_data,
                     x='Date', y='Predicted_Adj_Close', color=ticker_color,
                     linestyle=line_styles['Forecast'], linewidth=2, 
                     markers=False, label=f'{ticker} Forecast')
        
        plt.fill_between(ticker_future_data['Date'], 
                         ticker_future_data['yhat_lower'], 
                         ticker_future_data['yhat_upper'], 
                         color=ticker_color, alpha=0.25, label=f'{ticker} Forecast Interval')
        plt.plot(ticker_future_data['Date'], ticker_future_data['yhat_lower'], color=ticker_color, linestyle=':', linewidth=0.8, alpha=0.7)
        plt.plot(ticker_future_data['Date'], ticker_future_data['yhat_upper'], color=ticker_color, linestyle=':', linewidth=0.8, alpha=0.7)


    plt.title('Future Stock Price Forecasts (Current Date to End of 2025) with Confidence Interval', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Adjusted Closing Price (USD)', fontsize=12)

    plt.xlim(future_dates.min(), future_dates.max())

    min_y_forecast = future_forecast_df['yhat_lower'].min() 
    max_y_forecast = future_forecast_df['yhat_upper'].max()
    y_buffer = (max_y_forecast - min_y_forecast) * 0.15
    plt.ylim(min_y_forecast - y_buffer, max_y_forecast + y_buffer)


    handles, labels = plt.gca().get_legend_handles_labels()
    
    unique_ticker_handles_future = []
    unique_ticker_labels_future = []
    seen_tickers_future = set()

    for h, l in zip(handles, labels):
        if ' Forecast' in l:
            ticker_name = l.split(' ')[0]
            if ticker_name not in seen_tickers_future:
                unique_ticker_labels_future.append(ticker_name)
                unique_ticker_handles_future.append(h)
                seen_tickers_future.add(ticker_name)

    sorted_ticker_info_future = sorted(zip(unique_ticker_labels_future, unique_ticker_handles_future), key=lambda x: x[0])
    ticker_labels_sorted_future = [item[0] for item in sorted_ticker_info_future]
    ticker_handles_sorted_future = [item[1] for item in sorted_ticker_info_future]


    legend_elements_future_only = [
        Line2D([0], [0], color='black', lw=2, linestyle='-.', label='Forecasted Price'),
        Patch(facecolor='gray', edgecolor='black', alpha=0.25, label='Forecast Confidence Interval'),
        Line2D([0], [0], color='gray', lw=0.8, linestyle=':', label='Confidence Bounds')
    ]

    first_legend_future = plt.legend(ticker_handles_sorted_future, ticker_labels_sorted_future, title='Ticker', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='medium', title_fontsize='large')
    plt.gca().add_artist(first_legend_future) 

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

q2_end_date = datetime.date(2025, 6, 30)
q3_end_date = datetime.date(2025, 9, 30)
q4_end_date = datetime.date(2025, 12, 31)

if future_forecast_df.empty:
    print("Future forecast DataFrame is empty. Cannot calculate quarterly changes.")
else:
    last_actual_prices = df_historical.groupby('Ticker')['Adj Close'].last().reset_index()
    last_actual_prices.rename(columns={'Adj Close': 'Start_Price'}, inplace=True)

    for ticker in future_forecast_df['Ticker'].unique():
        ticker_forecast_data = future_forecast_df[future_forecast_df['Ticker'] == ticker].sort_values(by='Date')
        
        start_q2_price = last_actual_prices[last_actual_prices['Ticker'] == ticker]['Start_Price'].iloc[0] if not last_actual_prices[last_actual_prices['Ticker'] == ticker].empty else np.nan

        def get_forecast_price_at_date(forecast_df, target_date):
            closest_date_idx = (forecast_df['Date'] <= pd.Timestamp(target_date)).idxmax() if any(forecast_df['Date'] <= pd.Timestamp(target_date)) else None
            if closest_date_idx is not None:
                return forecast_df.loc[closest_date_idx, 'Predicted_Adj_Close']
            return np.nan

        q2_forecast_price = get_forecast_price_at_date(ticker_forecast_data, q2_end_date)
        q3_forecast_price = get_forecast_price_at_date(ticker_forecast_data, q3_end_date)
        q4_forecast_price = get_forecast_price_at_date(ticker_forecast_data, q4_end_date)

        if not np.isnan(start_q2_price) and not np.isnan(q2_forecast_price):
            q2_change = ((q2_forecast_price - start_q2_price) / start_q2_price) * 100
            quarterly_changes_list.append({'Ticker': ticker, 'Quarter': 'Q2 2025', 'Change (%)': q2_change})
        
        if not np.isnan(q2_forecast_price) and not np.isnan(q3_forecast_price):
            q3_change = ((q3_forecast_price - q2_forecast_price) / q2_forecast_price) * 100
            quarterly_changes_list.append({'Ticker': ticker, 'Quarter': 'Q3 2025', 'Change (%)': q3_change})

        if not np.isnan(q3_forecast_price) and not np.isnan(q4_forecast_price):
            q4_change = ((q4_forecast_price - q3_forecast_price) / q3_forecast_price) * 100
            quarterly_changes_list.append({'Ticker': ticker, 'Quarter': 'Q4 2025', 'Change (%)': q4_change})

quarterly_changes_df = pd.DataFrame(quarterly_changes_list)

if not quarterly_changes_df.empty:
    plt.figure(figsize=(14, 8)) 
    
    quarter_order = ['Q2 2025', 'Q3 2025', 'Q4 2025']
    quarterly_changes_df['Quarter'] = pd.Categorical(quarterly_changes_df['Quarter'], categories=quarter_order, ordered=True)
    quarterly_changes_df.sort_values(by=['Ticker', 'Quarter', 'Change (%)'], ascending=[True, True, False], inplace=True)

    # Plot the bar chart first
    sns.barplot(data=quarterly_changes_df, x='Change (%)', y='Ticker', hue='Quarter', 
                palette='viridis', dodge=True) # Use a generic palette initially

    # --- FIX for IndexError: Iterate through all patches and color based on their width ---
    # This is more robust as it doesn't rely on the DataFrame's row order matching patch order.
    for bar in plt.gca().patches:
        # Get the x-value (Change %) for each bar
        if bar.get_width() >= 0:
            bar.set_color('green')
        else:
            bar.set_color('red')


    plt.title('Projected Quarterly Stock Price Changes (Q2-Q4 2025)', fontsize=16)
    plt.xlabel('Projected Change (%)', fontsize=12)
    plt.ylabel('Ticker', fontsize=12)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    
    for container in plt.gca().containers:
        plt.bar_label(container, fmt='%.2f%%', label_type='edge', padding=3, fontsize=9)

    # Create a custom legend for quarters with their actual colors
    # This is necessary because we manually overrode bar colors.
    legend_handles = []
    legend_labels = []
    
    # Manually create patch objects for the legend
    for quarter in quarter_order:
        # Get a representative bar for this quarter to extract its color
        # This part is a bit tricky; we'll assume the first bar found for each quarter has the correct color after manual override
        # A safer approach might be to pre-define the legend colors or use a dummy plot.
        # For simplicity, we'll create patches with the desired green/red.
        dummy_patch_positive = Patch(facecolor='green', label=f'{quarter} (Positive)')
        dummy_patch_negative = Patch(facecolor='red', label=f'{quarter} (Negative)')
        # Add only if that type of change exists in the data for the quarter
        if any((quarterly_changes_df['Quarter'] == quarter) & (quarterly_changes_df['Change (%)'] >= 0)):
             legend_handles.append(Patch(facecolor='green', label=f'{quarter} (Gain)'))
        if any((quarterly_changes_df['Quarter'] == quarter) & (quarterly_changes_df['Change (%)'] < 0)):
             legend_handles.append(Patch(facecolor='red', label=f'{quarter} (Loss)'))
        
    # Simplify the legend: just show the quarter labels if we have a direct color mapping
    # Since we're coloring based on value, the hue legend is less useful for color itself.
    # We can create a simple legend for the 'Quarter' labels.
    # The actual color for positive/negative will be visually apparent.
    # For the legend, let's just show the default palette colors for 'Quarter' to identify the groups.
    # Or, we can just remove the hue legend entirely if the green/red is the primary visual.
    
    # Re-introducing the simplified legend. The colors in the legend will be Seaborn's default palette for hue,
    # but the bars themselves will be green/red. This might be slightly confusing but common.
    # A cleaner approach would be a custom legend showing "Positive Change (Green)" and "Negative Change (Red)".
    
    # Custom legend elements for positive/negative change, distinct from 'Quarter' hue
    custom_legend_elements = [
        Patch(facecolor='green', label='Projected Gain'),
        Patch(facecolor='red', label='Projected Loss')
    ]
    
    # Get original hue handles/labels for the Quarter legend
    q_handles, q_labels = plt.gca().get_legend_handles_labels()
    
    # Filter for the hue legend entries (Quarters)
    # This requires carefully extracting only the 'Quarter' labels and their corresponding default colors from Seaborn
    quarter_legend_handles = []
    quarter_legend_labels = []
    for h, l in zip(q_handles, q_labels):
        if l in quarter_order: # Only pick up the quarter labels
            quarter_legend_handles.append(h)
            quarter_legend_labels.append(l)

    # Create the first legend for Quarters
    first_legend = plt.legend(quarter_legend_handles, quarter_legend_labels, title='Quarter', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.gca().add_artist(first_legend) # Add this legend back to the figure

    # Create the second legend for Positive/Negative change
    plt.legend(handles=custom_legend_elements, title='Change Type', bbox_to_anchor=(1.02, 0.6), loc='upper left')


    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(os.path.join(ml_artifacts_dir, output_quarterly_changes_plot))
    plt.close()
    print(f"Generated and saved: Quarterly Forecast Changes Plot to {os.path.join(ml_artifacts_dir, output_quarterly_changes_plot)}")
else:
    print("No quarterly forecast data to plot in 'Quarterly Forecast Changes' graph.")

print("\nFuture Forecasting complete.")
print("ML Forecasting script execution finished.")