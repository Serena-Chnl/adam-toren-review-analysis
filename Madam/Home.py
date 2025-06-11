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
DATA_FILE_PATH = BASE_DIR / "Madam_dataset_cleaned.csv"

# --- Page Configuration (Global) ---
try:
    # Use madam_logo_01.png as the page icon (favicon)
    img_logo_icon = Image.open(LOGO_PATH)
    st.set_page_config(
        page_title="Madam Review Insights", # This sets the browser tab title
        page_icon=img_logo_icon, # Set favicon using madam_logo_01.png
        layout="wide"
    )
except FileNotFoundError:
    # Fallback if madam_logo_01.png is not found
    st.set_page_config(
        page_title="Madam Review Insights",
        page_icon="🍽️", # Fallback emoji icon
        layout="wide"
    )

# --- Header Image Section ---
# This section exclusively handles displaying madam_header.png as the main page title image.
try:
    madam_header_display = Image.open(HEADER_IMAGE_PATH)
    st.image(madam_header_display, use_container_width=True) # Display madam_header.png as the main title, taking full column width
except FileNotFoundError:
    # Fallback if the header image is not found.
    # We are explicitly asked to remove the text header "Madam Review Insights",
    # so we will use a more generic title here as a fallback or a warning.
    st.title("Review Insights Dashboard")
    st.warning(f"Madam header image not found at {HEADER_IMAGE_PATH}. Displaying a generic text title instead.")

# --- Introduction ---

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""

<span style="font-size:20px; color:#962965">**♦︎ Overview**</span>: Snapshot of key trends, sentiment, and ratings at a glance.
<span style="font-size:20px; color:#962965">**♦︎ Feedback Trending**</span>: Spot peaks of praise or moments to polish with time-based insights.
<span style="font-size:20px; color:#962965">**♦︎ Review Behaviour**</span>: Uncover when and how guests share their love (or gripes!).
<span style="font-size:20px; color:#962965">**♦︎ Keyword Insights**</span>: Find out what makes guests rave or rethink their visit.
<span style="font-size:20px; color:#962965">**♦︎ Customer Profile**</span>: Meet the global fans behind the reviews.
<span style="font-size:20px; color:#962965">**♦︎ Staff Performance**</span>: Celebrate the team stars sparking unforgettable moments.
<span style="font-size:20px; color:#962965">**♦︎ Forecast Analysis**</span>: Predicts future review trends and highlights days.

""", unsafe_allow_html=True)

st.markdown("<h1 style='font-size:20px; color:#962965;'>Use the sidebar to dive in each page and turn feedback into action!</h1>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Data Loading (Simplified) ---
@st.cache_data
def load_cleaned_data(file_path):
    """Loads the pre-processed CSV data and converts the 'Time' column."""
    try:
        df = pd.read_csv(file_path)
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')

        required_cols = ['Name', 'Rating', 'Time', 'Review', 'Language', 'compound', 'label']
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            st.error(f"CRITICAL ERROR: Cleaned data is missing required columns: {', '.join(missing_cols)}. Please re-run the data cleaning script.")
            return None

        nat_count = df['Time'].isna().sum()
        if nat_count > 0:
            st.warning(f"{nat_count} date/time entries could not be converted. These rows might be excluded from date-sensitive analyses.")
        return df
    except FileNotFoundError:
        st.error(f"CRITICAL ERROR: The data file '{file_path}' was not found. Please ensure 'main.py' has been run successfully.")
        return None
    except Exception as e:
        st.error(f"CRITICAL ERROR: An unexpected error occurred while loading the data: {e}")
        return None

# --- Initialize Session State for Data ---
if 'processed_data' not in st.session_state:
    df = load_cleaned_data(DATA_FILE_PATH)
    if df is not None:
        st.session_state.processed_data = df
    else:
        st.session_state.processed_data = None
        st.error("Failed to load pre-processed data. The dashboard cannot operate.")

# --- Sidebar and Data Filtering ---
st.sidebar.info("Select a page to begin your exploration!")
st.sidebar.header("Review Explorer Filters")

data = st.session_state.get('processed_data')
filtered_data = data

if data is not None and not data.empty:
    # Set min_date to January 1st of the earliest year
    earliest_year = data['Time'].min().year if pd.notna(data['Time'].min()) else datetime.datetime.now().year
    min_date = datetime.date(earliest_year, 1, 1)
    max_date_data = data['Time'].max().date() if pd.notna(data['Time'].max()) else datetime.date.today()

    default_start = min_date
    default_end = max_date_data

    start_date = st.sidebar.date_input("Start date", default_start, min_value=min_date, max_value=max_date_data)
    end_date = st.sidebar.date_input("End date", default_end, min_value=min_date, max_value=max_date_data)

    if start_date > end_date:
        st.sidebar.error("Error: Start date cannot be after end date.")
        filtered_data = pd.DataFrame(columns=data.columns)
    else:
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        filtered_data = data[data['Time'].between(start_datetime, end_datetime)]
else:
    st.sidebar.warning("Date filter cannot be applied: Data is not available.")

# --- Display Filtered Data Preview ---
st.subheader("▸ Filtered Data Preview 🔍")

if filtered_data is not None:
    st.write(f"Displaying **{len(filtered_data)}** reviews from **{start_date.strftime('%Y-%m-%d')}** to **{end_date.strftime('%Y-%m-%d')}**.")

    if not filtered_data.empty:
        cols_to_show = ['Name', 'Rating', 'Time', 'Review', 'Language', 'compound', 'label']
        st.dataframe(filtered_data[cols_to_show])
    else:
        st.info("No reviews found for the selected date range.")

    st.markdown("""
        **Understanding the Sentiment Columns:**
        * **`compound` score**: An overall sentiment score from -1 (very negative) to +1 (very positive), generated by a BERT model.
        * **`label`**: A simplified category (positive, negative, or neutral) assigned based on the sentiment analysis.
        """)
else:
    st.error("Data is not available for display. Please check the loading status above.")

st.info("The pre-processed data is ready in `st.session_state.processed_data` for all other dashboard pages.")
st.markdown("---")
