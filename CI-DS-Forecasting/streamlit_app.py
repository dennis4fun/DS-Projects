import streamlit as st
import pandas as pd
import os
import subprocess
import requests
import json
import time
from dotenv import load_dotenv

# --- Load environment variables from .env file ---
load_dotenv()

# --- Configuration ---
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
repo_root = os.path.dirname(script_dir)

DATA_DIR = os.path.join(repo_root, 'data')
ML_ARTIFACTS_DIR = os.path.join(repo_root, 'ml_artifacts')
PLOTS_DIR = os.path.join(DATA_DIR, 'plots')

# Paths to your core Python scripts
GET_STOCK_DATA_SCRIPT = os.path.join(script_dir, 'get_stock_data.py')
DATA_CLEANING_EDA_SCRIPT = os.path.join(script_dir, 'data_cleaning_EDA.py')
MODEL_TRAINING_SCRIPT = os.path.join(script_dir, 'model_training.py')
ML_FORECASTING_SCRIPT = os.path.join(script_dir, 'ml_forecasting.py')

# GitHub API Configuration
GITHUB_REPO_OWNER = "dennis4fun"
GITHUB_REPO_NAME = "DS-Projects"
GITHUB_WORKFLOW_NAME = "Stock Price Forecasting CI/CD"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# --- Helper Functions ---

def run_script_and_capture_output(script_path, script_name):
    """Runs a Python script using pipenv and captures its output."""
    st.info(f"Running {script_name}...")
    try:
        process = subprocess.run(
            ['pipenv', 'run', 'python', script_path],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_root
        )
        st.success(f"{script_name} completed successfully!")
        st.text_area(f"{script_name} Output:", process.stdout, height=200)
        if process.stderr:
            st.warning(f"{script_name} had stderr output:\n{process.stderr}")
        return True, process.stdout + process.stderr
    except subprocess.CalledProcessError as e:
        st.error(f"{script_name} failed with an error!")
        st.exception(e)
        st.text_area(f"{script_name} Error Output:", e.stderr, height=300)
        return False, e.stdout + e.stderr
    except FileNotFoundError:
        st.error(f"Error: pipenv or python not found. Ensure pipenv is installed and in your PATH.")
        return False, "pipenv or python not found."

@st.cache_data(ttl=60)
def get_latest_github_workflow_run_status():
    """Fetches the status of the latest workflow run from GitHub API."""
    if not GITHUB_TOKEN:
        return {"error": "GITHUB_TOKEN environment variable not set. Cannot fetch GitHub status."}

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    url = f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/actions/workflows/main.yml/runs"
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        runs_data = response.json()

        latest_run = None
        for run in runs_data.get('workflow_runs', []):
            if run['name'] == GITHUB_WORKFLOW_NAME:
                latest_run = run
                break

        if latest_run:
            return {
                "status": latest_run.get('status', 'N/A'),
                "conclusion": latest_run.get('conclusion', 'N/A'),
                "run_number": latest_run.get('run_number', 'N/A'),
                "html_url": latest_run.get('html_url', '#'),
                "created_at": latest_run.get('created_at', 'N/A'),
                "updated_at": latest_run.get('updated_at', 'N/A')
            }
        else:
            return {"message": "No runs found for the specified workflow."}

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching GitHub workflow status: {e}")
        return {"error": f"Failed to connect to GitHub API: {e}"}
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return {"error": f"An unexpected error occurred: {e}"}

def load_and_display_image(image_path, caption, width=800):
    """Loads and displays an image from a given path."""
    if os.path.exists(image_path):
        st.image(image_path, caption=caption, width=width)
    else:
        st.warning(f"Image not found: {image_path}. Please run the forecasting scripts.")

@st.cache_data(ttl=300) # Cache dataframes for 5 minutes
def load_csv_data(filepath):
    """Loads a CSV file into a pandas DataFrame."""
    if os.path.exists(filepath):
        try:
            # Removed errors='coerce' as it's not supported in older pandas versions with parse_dates
            df = pd.read_csv(filepath, parse_dates=['Date'])
            return df
        except Exception as e:
            st.error(f"Error loading {os.path.basename(filepath)}: {e}")
            return pd.DataFrame()
    else:
        return pd.DataFrame()

# --- Moved display_filterable_dataframe here ---
def display_filterable_dataframe(df, key_suffix):
    """Displays a DataFrame with filtering options by Ticker and Date range."""
    st.markdown("---")
    st.markdown("**Filter Data:**")

    col1, col2 = st.columns(2)

    with col1:
        # Ticker filter
        all_tickers = ['All'] + sorted(df['Ticker'].unique().tolist()) if 'Ticker' in df.columns else ['All']
        selected_ticker = st.selectbox("Select Ticker", all_tickers, key=f"ticker_filter_{key_suffix}")
        
    with col2:
        # Date range filter
        min_date = df['Date'].min().date() if 'Date' in df.columns and not df['Date'].empty else pd.Timestamp.today().date() - pd.Timedelta(days=365)
        max_date = df['Date'].max().date() if 'Date' in df.columns and not df['Date'].empty else pd.Timestamp.today().date()
        
        start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date, key=f"start_date_{key_suffix}")
        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date, key=f"end_date_{key_suffix}")

    filtered_df = df.copy()

    # Apply Ticker filter
    if selected_ticker != 'All' and 'Ticker' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Ticker'] == selected_ticker]

    # Apply Date range filter
    if 'Date' in filtered_df.columns:
        # Ensure date columns are datetime objects for proper comparison
        filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        filtered_df = filtered_df[(filtered_df['Date'] >= start_dt) & (filtered_df['Date'] <= end_dt)]

    st.write(f"Filtered Rows: {len(filtered_df)}")
    st.dataframe(filtered_df, height=300)
# --- End of moved display_filterable_dataframe ---


# --- Streamlit UI Layout ---

st.set_page_config(
    page_title="Stock Price Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Stock Price Forecasting Dashboard")
st.markdown("""
This dashboard allows you to:
- **View** the latest stock price forecasts and EDA plots.
- **Run** the entire data pipeline locally with a single click.
- **Check** the status of the automated CI/CD pipeline on GitHub Actions.
- **Inspect** the raw and processed data files.
""")

# --- Sidebar for Navigation/Actions ---
st.sidebar.header("Navigation & Actions")
selected_section = st.sidebar.radio(
    "Go to:",
    ["View Forecasts & EDA", "View Data Files", "Run Local Pipeline", "GitHub Actions Status"]
)

# --- Main Content Area ---

if selected_section == "View Forecasts & EDA":
    st.header("📊 Latest Stock Forecasts & EDA Plots")
    st.markdown("Here you can see the results from the latest local run of your forecasting pipeline.")

    # Forecast Plots
    st.subheader("Future Stock Price Forecasts")
    load_and_display_image(os.path.join(ML_ARTIFACTS_DIR, 'historical_and_future_forecast.png'), 
                           "Historical and Future Stock Price Forecasts with Confidence Intervals")
    load_and_display_image(os.path.join(ML_ARTIFACTS_DIR, 'future_forecast_only.png'), 
                           "Future Stock Price Forecasts Only (Zoomed View)")
    load_and_display_image(os.path.join(ML_ARTIFACTS_DIR, 'quarterly_forecast_changes.png'), 
                           "Projected Quarterly Stock Price Changes (Q2-Q4 2025)")

    # EDA Plots (assuming they are in data/plots)
    st.subheader("Exploratory Data Analysis (EDA) Plots")
    load_and_display_image(os.path.join(PLOTS_DIR, 'stock_price_timeline.png'), 
                           "Stock Price Timeline for Selected S&P 500 Stocks")
    load_and_display_image(os.path.join(PLOTS_DIR, 'daily_trading_volume.png'), 
                           "Daily Trading Volume Over Time")
    load_and_display_image(os.path.join(PLOTS_DIR, 'daily_returns_distribution.png'), 
                           "Distribution of Daily Returns")
    load_and_display_image(os.path.join(PLOTS_DIR, 'vix_close_timeline.png'), 
                           "VIX (Volatility Index) Close Price Over Time")
    load_and_display_image(os.path.join(PLOTS_DIR, 'tnx_close_timeline.png'), 
                           "10-Year US Treasury Yield (TNX) Close Price Over Time")


elif selected_section == "View Data Files": # NEW SECTION FOR DATA FILES
    st.header("🗂️ Raw and Processed Data Files")
    st.markdown("Inspect the raw and processed stock data. Use the filters to explore specific tickers or dates.")

    raw_csv_path = os.path.join(DATA_DIR, 'raw_stock_data.csv')
    processed_csv_path = os.path.join(DATA_DIR, 'processed_data.csv')

    st.subheader("Raw Stock Data")
    raw_df = load_csv_data(raw_csv_path)
    if not raw_df.empty:
        st.write(f"Displaying {len(raw_df)} rows from `raw_stock_data.csv`")
        display_filterable_dataframe(raw_df, "raw_data_filter")
    else:
        st.warning(f"Raw data file not found or is empty: `{raw_csv_path}`. Please run the pipeline locally.")

    st.subheader("Processed Stock Data")
    processed_df = load_csv_data(processed_csv_path)
    if not processed_df.empty:
        st.write(f"Displaying {len(processed_df)} rows from `processed_data.csv`")
        display_filterable_dataframe(processed_df, "processed_data_filter")
    else:
        st.warning(f"Processed data file not found or is empty: `{processed_csv_path}`. Please run the pipeline locally and ensure `data_cleaning_EDA.py` produces output.")

elif selected_section == "Run Local Pipeline":
    st.header("⚙️ Run Full Data Pipeline Locally")
    st.markdown("""
    Click the button below to run the entire stock forecasting pipeline on your local machine.
    This will:
    1.  Fetch fresh raw stock data.
    2.  Clean, process, and generate EDA plots.
    3.  Train Prophet models.
    4.  Generate forecasts and new report plots.
    """)

    if st.button("🚀 Run Full Pipeline"):
        st.write("Starting local pipeline execution...")
        status_placeholder = st.empty()
        status_message = ""
        all_output = []

        # Step 1: Run get_stock_data.py
        status_message += "Running Data Scraping...\n"
        status_placeholder.text_area("Pipeline Progress:", status_message, height=150)
        success, output = run_script_and_capture_output(GET_STOCK_DATA_SCRIPT, "get_stock_data.py")
        all_output.append(output)
        if not success:
            st.error("Data Scraping Failed. Check logs above.")
            st.stop()
        
        # Step 2: Run data_cleaning_EDA.py
        status_message += "Running Data Cleaning & EDA...\n"
        status_placeholder.text_area("Pipeline Progress:", status_message, height=150)
        success, output = run_script_and_capture_output(DATA_CLEANING_EDA_SCRIPT, "data_cleaning_EDA.py")
        all_output.append(output)
        if not success:
            st.error("Data Cleaning & EDA Failed. Check logs above.")
            st.stop()

        # Step 3: Run model_training.py
        status_message += "Running Model Training...\n"
        status_placeholder.text_area("Pipeline Progress:", status_message, height=150)
        success, output = run_script_and_capture_output(MODEL_TRAINING_SCRIPT, "model_training.py")
        all_output.append(output)
        if not success:
            st.error("Model Training Failed. Check logs above.")
            st.stop()

        # Step 4: Run ml_forecasting.py
        status_message += "Running ML Forecasting & Reporting...\n"
        status_placeholder.text_area("Pipeline Progress:", status_message, height=150)
        success, output = run_script_and_capture_output(ML_FORECASTING_SCRIPT, "ml_forecasting.py")
        all_output.append(output)
        if not success:
            st.error("ML Forecasting & Reporting Failed. Check logs above.")
            st.stop()

        st.success("🎉 Full Pipeline Run Completed Successfully Locally! 🎉")
        st.write("The plots and processed data in your local `data/` and `ml_artifacts/` folders have been updated.")
        
elif selected_section == "GitHub Actions Status":
    st.header("🌐 GitHub Actions Workflow Status")
    st.markdown(f"""
    This section shows the status of the latest run of your '{GITHUB_WORKFLOW_NAME}' workflow on GitHub.
    It runs automatically every Friday at 5:00 PM EST (22:00 UTC).
    
    **Important:** To fetch status, you need to set a GitHub Personal Access Token (PAT) 
    as an environment variable named `GITHUB_TOKEN`.
    
    1.  Go to [GitHub Settings > Developer settings > Personal access tokens](https://github.com/settings/tokens).
    2.  Generate a new token with at least `repo` scope.
    3.  Set it as an environment variable (e.g., `export GITHUB_TOKEN='your_token'` in terminal, or in a `.env` file for `python-dotenv`).
    """)

    if st.button("🔄 Check Latest GitHub Run"):
        with st.spinner("Fetching latest workflow run status..."):
            status_result = get_latest_github_workflow_run_status()
            
            if "error" in status_result:
                st.error(status_result["error"])
            elif "message" in status_result:
                st.info(status_result["message"])
            else:
                st.subheader(f"Latest Run: #{status_result['run_number']}")
                st.write(f"**Status:** `{status_result['status']}`")
                st.write(f"**Conclusion:** `{status_result['conclusion']}`")
                st.write(f"**Created At (UTC):** {status_result['created_at']}")
                st.write(f"**Last Updated (UTC):** {status_result['updated_at']}")
                st.markdown(f"**[View Full Run Details on GitHub]({status_result['html_url']})**")
