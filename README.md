## Sample CI Project with Data Science Elements

## 📦 Python Environment Setup with Pipenv

This project uses [Pipenv](https://pipenv.pypa.io/en/latest/) for streamlined dependency management and isolated virtual environments, ensuring our development setup is always consistent.

Prerequisites
To get started, make sure you have pipenv installed on your system:

```bash
pip install --user pipenv
```

Getting Started (For Cloned Repositories)

If you've just cloned this repository, navigate to the project's root directory in your terminal and run:

```bash
pipenv install
```

This command will automatically create the necessary virtual environment and install all project dependencies as specified in the `Pipfile.lock` file.

## 📊 Data Pipeline Overview

This project includes two main Python scripts that form the initial data pipeline:

### 1. `get_stock_data.py` **(Data Collection)**

This script is responsible for fetching historical stock price data for a predefined list of S&P 500 companies.

- **Purpose:** Web scrapes daily historical stock data (Open, High, Low, Close, Volume, Adjusted Close) from a reliable online source (using `yfinance`).

- **Output:** Creates a data/raw_stock_data.csv file. This CSV contains all collected stock data in a "long" format, where each row represents a daily record for a specific stock, clearly identified by a 'Ticker' column.

- **Overwrite Policy:** Automatically deletes any existing raw_stock_data.csv before a new run to ensure a fresh dataset.

- **Robustness:** Includes retry mechanisms for failed data fetches and handles potential column inconsistencies from the data source.

**To run this script:**

```bash
pipenv run python CI-DS-Forecasting/get_stock_data.py

```

### 2. `data_cleaning_EDA.py` **(Data Cleaning & EDA)**

This script takes the raw stock data, performs essential cleaning and transformations, and generates insightful visualizations.

- **Purpose:** Loads the `raw_stock_data.csv`, converts data types (e.g., 'Date' to datetime, prices to numeric), handles missing values, and performs basic data quality checks.

- **Transformations:** Common data prep cleaning.

  - **Ensures** 'Date' and all price/volume columns are in the correct data types.

  - **Fills missing values** (e.g., stock market holidays) using forward and backward fill within each stock's data.

  - **Calculates** 'Daily Return' as a new feature for analysis.

- **Visualizations:** Generates and saves the following charts into the `data/plots/` directory:

  - **Stock Price Timeline:** Visualizes the Adjusted Closing Price for all selected stocks over time, allowing for easy comparison.

  - **Daily Trading Volume:** Shows the trading volume trends for each stock.

  - **Distribution of Daily Returns:** Provides histograms to understand the return volatility for each stock.

- **Output:** Saves the cleaned and processed data into `data/processed_data.csv`. All generated plots are saved as PNG image files in the `data/plots/` subdirectory.

  - **Overwrite Policy:** Automatically overwrites processed_data.csv and the plot image files on each run.

**To run this script:**

```bash
pipenv run python CI-DS-Forecasting/get_stock_data.py
```

## 🚀 Running the Project Scripts:

Follow these steps sequentially to run the full forecasting pipeline:

### 1. Fetch Raw Stock Data:

This script scrapes historical stock data for predefined tickers (e.g., S&P 500 components) and saves it as `raw_stock_data.cs`v in the `data/` directory.

```bash
pipenv run python get_stock_data.py
```

### 2. Clean & Perform EDA:

This script loads the raw stock data, performs necessary cleaning and preprocessing steps, conducts Exploratory Data Analysis (EDA), and saves the processed data as `processed_data.csv` in the `data/ directory`. It also generates several EDA plots (e.g., daily returns distribution) in `data/plots/.`

```bash
pipenv run python data_cleaning_EDA.py
```

### 3. Train Forecasting Models:

This script utilizes the `processed_data.csv` to train individual Prophet time-series models for each stock ticker. The trained models are then saved as `.joblib` files in the `ml_artifacts/` directory.

```bash
pipenv run python model_training.py
```

### 4. Generate Stock Price Forecasts & Reports:

This script loads the trained Prophet models, generates future stock price forecasts until the end of 2025, and produces three types of visualizations in the `ml_artifacts/` directory:

- `historical_and_future_forecast.png:` Shows historical prices along with future forecasts and confidence intervals.

- `future_forecast_only.png:` A zoomed-in view of only the future forecasts and their confidence intervals.

- `quarterly_forecast_changes.png:` A horizontal bar graph comparing projected quarterly percentage changes for each stock.

```bash
pipenv run python ml_forecasting.py
```

### Future Enhancements:

Integration with a CI/CD pipeline (e.g., GitHub Actions) for automated execution.

Advanced model evaluation metrics and backtesting.

Incorporating external regressors (e.g., news sentiment, macroeconomic indicators) into the Prophet model.

Generating a comprehensive financial report using an LLM.
