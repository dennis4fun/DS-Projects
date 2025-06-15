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
