# Home.py (Main script for multi-page app, updated for pre-processed data, header image, and favicon)

import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path
import datetime
import nltk

def download_nltk_resources() -> bool:
    """
    Downloads required NLTK resources for text processing.
    Returns True if successful, False otherwise.
    """
    try:
        for resource, download_name in [
            ('corpora/stopwords', 'stopwords'),
            ('corpora/wordnet', 'wordnet'),
            ('corpora/omw-1.4', 'omw-1.4'),
            ('tokenizers/punkt', 'punkt'),
            ('tokenizers/punkt_tab', 'punkt_tab')
        ]:
            try:
                nltk.data.find(resource)
            except LookupError:
                nltk.download(download_name, quiet=True)
        return True
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")
        return False

# --- Define Base Directory ---
BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "madam_logo_01.png"   # Path for the favicon
HEADER_IMAGE_PATH = BASE_DIR / "madam_header.png" # Path for the main header image
DATA_FILE_PATH = BASE_DIR / "Madam_dataset_cleaned.csv" # Path for the cleaned data

# --- Page Configuration (Global - Called ONLY ONCE in the main app file) ---
page_icon_value = "📊" # Default to a generic emoji icon
try:
    if LOGO_PATH.exists(): # Check if the favicon file exists
        img_logo_icon = Image.open(LOGO_PATH)
        page_icon_value = img_logo_icon
except Exception: # Catch any error during image loading
    page_icon_value = "📊" # Fallback to emoji

st.set_page_config(
    page_title="Review Analysis Dashboard",
    page_icon=page_icon_value,
    layout="wide"
)

# Call centralized NLTK resource download
if not download_nltk_resources():
    st.error("Failed to download required NLTK resources. Please check your internet connection or NLTK setup.")
    st.stop() # Stop the app execution if resources aren't available

# --- Load Data ---
# Ensure this path is correct relative to your app's execution location
try:
    df_home = pd.read_csv(DATA_FILE_PATH)
except FileNotFoundError:
    st.error(f"Error: Data file not found at {DATA_FILE_PATH}. Please ensure the data cleaning process was successful and the file exists.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading data: {e}")
    st.stop()

# Data Preprocessing for display in Home page
df_home['Rating'] = pd.to_numeric(df_home['Rating'].astype(str).str.extract(r'(\d)').fillna(0).astype(int), errors='coerce')

if 'Time' in df_home.columns:
    df_home['Time'] = pd.to_datetime(df_home['Time'], errors='coerce')
    df_home = df_home.dropna(subset=['Time']) # Remove rows where Time is NaT
else:
    st.warning("'Time' column not found in the dataset for Home page. Date-based filtering and analysis will be limited.")

# --- Display Header Image ---
if HEADER_IMAGE_PATH.exists():
    st.image(str(HEADER_IMAGE_PATH), use_column_width=True)
else:
    st.title("Review Analysis Dashboard") # Fallback title if header image not found

st.markdown("---")

st.subheader("Welcome to the Review Analysis Dashboard!")
st.write("""
This dashboard provides insights into customer reviews, helping you understand sentiment, identify key trends,
and explore customer feedback over time. Use the filters in the sidebar to refine your view.
""")

# --- Sidebar for filters (Home Page specific) ---
st.sidebar.header("Home Page Filters")

# Date range filter
if 'Time' in df_home.columns and not df_home['Time'].empty:
    min_date = df_home['Time'].min().to_pydatetime()
    max_date = df_home['Time'].max().to_pydatetime()

    start_date, end_date = st.sidebar.slider(
        "Select Date Range:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )

    if start_date > end_date:
        st.sidebar.error("Start date cannot be after end date.")
        filtered_data = pd.DataFrame(columns=df_home.columns) # Empty DataFrame
    else:
        # Include full end day by adding nearly 24 hours
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        filtered_data = df_home[df_home['Time'].between(start_datetime, end_datetime)]
else:
    st.sidebar.warning("Date filter cannot be applied: 'Time' column not available or empty in data.")
    filtered_data = df_home.copy() # Use full data if date column is missing

# --- Display Filtered Data Preview ---
st.subheader("▸ Filtered Data Preview 🔍")

if not filtered_data.empty:
    st.write(f"Displaying **{len(filtered_data)}** reviews from **{start_date.strftime('%Y-%m-%d')}** to **{end_date.strftime('%Y-%m-%d')}**.")
    cols_to_show = ['Name', 'Rating', 'Time', 'Review', 'Language', 'compound', 'label']
    # Ensure all cols_to_show are in filtered_data before selecting
    cols_to_show = [col for col in cols_to_show if col in filtered_data.columns]
    st.dataframe(filtered_data[cols_to_show])
else:
    st.info("No reviews found for the selected date range or data not available.")

st.markdown("""
    **Understanding the Sentiment Columns:**
    * **`compound` score**: An overall sentiment score from -1 (very negative) to +1 (very positive), generated by a BERT model.
    * **`label`**: A simplified category (positive, negative, or neutral) assigned based on the sentiment analysis.
    """)