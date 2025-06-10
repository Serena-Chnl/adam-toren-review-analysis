# Home.py (Main script for A'DAM LOOKOUT multi-page Streamlit dashboard)

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
HEADER_IMAGE_PATH = BASE_DIR / "lookout_header.png"
LOOKOUT_LOGO_PATH = BASE_DIR / "lookout_logo_01.png" # Add this line for the new logo path
DATA_FILE_PATH = BASE_DIR / "Lookout_dataset_cleaned.csv"

# --- Page Configuration (Global) ---
page_icon = "🌆"  # Default emoji icon
try:
    # Use the new logo path for the page_icon
    img_icon = Image.open(LOOKOUT_LOGO_PATH)
    page_icon = img_icon
except Exception as e:
    st.warning(f"Cannot use '{LOOKOUT_LOGO_PATH.name}' as page icon: {e}. Using default emoji icon.")

st.set_page_config(
    page_title="A'DAM LOOKOUT Review Insights",
    page_icon=page_icon, # This will now use lookout_logo_01.png
    layout="wide"
)

# 强制浅色主题的 CSS
st.markdown("""
    <style>
    /* 主页面和应用程序背景 */
    body, .stApp {
        background-color: #f9fafb !important;
    }
    /* 侧边栏背景 */
    [data-testid="stSidebar"] {
        background-color: #e5e7eb !important;
    }
    /* 文本颜色 */
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #374151 !important;
    }
    /* 表格和数据框架背景 */
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        background-color: #ffffff !important;
    }
    /* Plotly 图表背景 */
    .plotly-graph-div, .js-plotly-plot .plotly .main-svg {
        background-color: #ffffff !important;
    }
    /* 增强侧边栏过滤器边框（如日期选择器） */
    [data-testid="stDateInput"] input, [data-testid="stSelectbox"] div, [data-testid="stSlider"] div {
        border: 1px solid #d1d5db !important;
        border-radius: 4px !important;
        background-color: #ffffff !important;
        color: #374151 !important;
    }
    /* 侧边栏输入框焦点样式 */
    [data-testid="stDateInput"] input:focus, [data-testid="stSelectbox"] div:focus, [data-testid="stSlider"] div:focus {
        border-color: initial !important; /* 或 simple `border: none !important;` */
        box-shadow: 0 0 0 2px rgba(90, 90, 90, 0.1) !important;
    }
    /* 侧边栏文本和标签 */
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown {
        color: #374151 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header Image Section ---
try:
    header_image = Image.open(HEADER_IMAGE_PATH)
    st.image(header_image, use_container_width=True)
except Exception as e:
    st.warning(f"Failed to load header image 'lookout_heading.png': {e}. Please ensure the file is in the correct folder.")

# --- Introduction ---
st.markdown("""
Welcome to the A'DAM LOOKOUT Review Insights Dashboard! **Dive in to discover:**
- <span style="color:#5a5a5a">**Overview**</span>: Quick view of trends, sentiment, and ratings for A'DAM Lookout.
- <span style="color:#5a5a5a">**Feedback Trending**</span>: Track peaks in praise or concerns over time.
- <span style="color:#5a5a5a">**Review Behaviour**</span>: Understand when and how visitors share their feedback.
- <span style="color:#5a5a5a">**Keyword Insights**</span>: Identify what visitors love or dislike.
- <span style="color:#5a5a5a">**Customer Profile**</span>: Discover the global audience visited us.
- <span style="color:#5a5a5a">**Forecast**</span>: Predict future review trends and pinpoint high-traffic days.

Use the sidebar to navigate pages and turn feedback into action!
""", unsafe_allow_html=True)
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
        * **`label`**: A simplified category (e.g., positive, negative, or neutral) assigned based on sentiment analysis.
        """)
else:
    st.error("Data is not available for display.")

st.info("The pre-processed dataset is ready in `st.session_state.processed_data` for all other dashboard pages.")
st.markdown("---")