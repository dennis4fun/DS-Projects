import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib # For saving and loading models

# --- Configuration ---
input_dir = 'data'
processed_csv_file = 'processed_data.csv'
input_filepath = os.path.join(input_dir, processed_csv_file)

# Define the directory for saving plots and models
output_dir = 'ml_artifacts'
output_model_file = 'stock_forecaster_model.joblib'
output_scaler_file = 'scaler.joblib'
output_features_list_file = 'features.joblib' # NEW: File to save feature names
output_forecast_plot = 'forecast_vs_actual.png' # For test set visualization
output_feature_importance_plot = 'feature_importance.png'

# --- Ensure Output Directories Exist ---
os.makedirs(input_dir, exist_ok=True) # Ensure data dir exists
os.makedirs(output_dir, exist_ok=True) # Ensure ML artifacts dir exists
print(f"Ensured directories '{input_dir}' and '{output_dir}' exist.")

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

# --- Feature Engineering ---
print("\nStarting Feature Engineering...")

# Sort data - crucial for time series operations
df.sort_values(by=['Ticker', 'Date'], inplace=True)

# 1. Lag Features: Previous day's closing prices and volumes
print("Generating Lag Features...")
df_temp = df.copy() 

for ticker in df_temp['Ticker'].unique():
    ticker_mask = (df_temp['Ticker'] == ticker)
    ticker_df_part = df_temp.loc[ticker_mask].copy() 
    
    for i in [1, 2, 3, 5, 7, 10]:
        ticker_df_part[f'Adj_Close_Lag_{i}'] = ticker_df_part['Adj Close'].shift(i)
    
    ticker_df_part['Volume_Lag_1'] = ticker_df_part['Volume'].shift(1)

    for col in ticker_df_part.columns:
        if col not in df_temp.columns:
            df_temp[col] = pd.NA
        df_temp.loc[ticker_mask, col] = ticker_df_part[col]
df = df_temp

# 2. Rolling Window Statistics (Moving Averages)
print("Generating Rolling Window Statistics...")
df['MA_5_Day'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
df['MA_10_Day'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
df['Volume_MA_5_Day'] = df.groupby('Ticker')['Volume'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

# 3. Time-based Features
print("Generating Time-based Features...")
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['DayOfYear'] = df['Date'].dt.dayofyear
df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)

# Target Variable: Predict the 'Adj Close' price of the *next* day
df['Target_Adj_Close'] = df.groupby('Ticker')['Adj Close'].shift(-1)
TARGET = 'Target_Adj_Close'

# Drop rows with NaN values created by lag features, rolling statistics, or target shift
initial_rows = df.shape[0]
df.dropna(subset=df.columns.difference(['Daily Return']), inplace=True) 
print(f"Dropped {initial_rows - df.shape[0]} rows due to NaN values (from feature engineering / target shift).")
print(f"DataFrame shape after feature engineering and NaN removal: {df.shape}")

# Define features (X) and target (y)
features = [col for col in df.columns if col not in ['Date', 'Ticker', 'Adj Close', TARGET, 'Daily Return']]
X = df[features]
y = df[TARGET]

print(f"Features (X) selected: {X.columns.tolist()}")
print(f"Target (y) selected: {TARGET}")

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=X.index) 

print("Features scaled using StandardScaler.")

# --- Train/Test Split (Time Series Aware) ---
# Split chronologically, using the last 20% of the data as test set for each ticker.
test_size_ratio = 0.2
train_df_list = []
test_df_list = []

for ticker in df['Ticker'].unique():
    ticker_data = X_scaled_df.loc[df['Ticker'] == ticker]
    ticker_target = y.loc[df['Ticker'] == ticker]

    split_index = int(len(ticker_data) * (1 - test_size_ratio))
    
    train_df_list.append(ticker_data.iloc[:split_index])
    test_df_list.append(ticker_data.iloc[split_index:])

X_train = pd.concat(train_df_list)
X_test = pd.concat(test_df_list)

y_train_list = []
y_test_list = []
for ticker in df['Ticker'].unique():
    ticker_target = y.loc[df['Ticker'] == ticker]
    split_index = int(len(ticker_target) * (1 - test_size_ratio))
    y_train_list.append(ticker_target.iloc[:split_index])
    y_test_list.append(ticker_target.iloc[split_index:])

y_train = pd.concat(y_train_list)
y_test = pd.concat(y_test_list)

print(f"\nData split into training and testing sets:")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# --- Model Training ---
print("\nStarting Model Training (RandomForestRegressor)...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) 
model.fit(X_train, y_train)
print("Model training complete.")

# --- Model Evaluation ---
print("\nStarting Model Evaluation...")
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# --- Feature Importance ---
if hasattr(model, 'feature_importances_'):
    feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nFeature Importances:")
    print(feature_importances.to_string())

    plt.figure(figsize=(12, 7))
    sns.barplot(x=feature_importances.values, y=feature_importances.index, palette='viridis')
    plt.title('Feature Importances from Random Forest Model')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_feature_importance_plot))
    plt.close()
    print(f"Generated and saved: Feature Importance Plot to {os.path.join(output_dir, output_feature_importance_plot)}")

# --- Visualization of Forecast vs. Actual (for a single ticker on test set) ---
print("\nGenerating Test Set Forecast Visualization...")
sample_ticker = df['Ticker'].unique()[0] if df['Ticker'].nunique() > 0 else None

if sample_ticker:
    plot_df = pd.DataFrame({
        'Date': df.loc[y_test.index]['Date'], 
        'Actual': y_test,
        'Predicted': y_pred
    })
    
    plot_df_sample = plot_df.loc[df['Ticker'] == sample_ticker].sort_values(by='Date')

    plt.figure(figsize=(15, 7))
    plt.plot(plot_df_sample['Date'], plot_df_sample['Actual'], label='Actual Price', color='blue', marker='o', markersize=4, linewidth=1)
    plt.plot(plot_df_sample['Date'], plot_df_sample['Predicted'], label='Predicted Price', color='red', linestyle='--', marker='x', markersize=4, linewidth=1)
    plt.title(f'Stock Price Forecast vs. Actual (Test Set) for {sample_ticker}')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Closing Price (USD)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_forecast_plot))
    plt.close()
    print(f"Generated and saved: Forecast vs Actual Plot for {sample_ticker} (Test Set) to {os.path.join(output_dir, output_forecast_plot)}")
else:
    print("Not enough unique tickers to generate sample forecast visualization for test set.")


# --- Save Model and Scaler ---
joblib.dump(model, os.path.join(output_dir, output_model_file))
joblib.dump(scaler, os.path.join(output_dir, output_scaler_file))
joblib.dump(features, os.path.join(output_dir, output_features_list_file)) # NEW: Save feature names
print(f"\nTrained model saved to '{os.path.join(output_dir, output_model_file)}'")
print(f"Scaler saved to '{os.path.join(output_dir, output_scaler_file)}'")
print(f"Feature names saved to '{os.path.join(output_dir, output_features_list_file)}'")

print("\nModel training script execution finished.")
