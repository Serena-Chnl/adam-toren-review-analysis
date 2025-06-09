# pages/06_Staff_Performance.py

import streamlit as st
import pandas as pd
import plotly.express as px
from fuzzywuzzy import fuzz
import re
from PIL import Image
from pathlib import Path
from typing import List, Optional, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import os

from Home import download_nltk_resources

# --- Page Configuration ---
st.set_page_config(page_title="Staff Performance Analysis", layout="wide")

# Call centralized NLTK resource download
if not download_nltk_resources():
    st.error("Failed to download required NLTK resources. Some features may not work.")

# --- Configuration ---
STAFF_NAMES = [
    "Aimée", "Alhassan", "Bas", "Britt", "Camila", "Daaf", "Darren", "Yesni", "Dimitteo", "Dimitris",
    "Ektor", "Eleonore", "Georgios Moy", "Georgios Myro", "Vikram", "Jacky", "Janet", "Jason", "Jody",
    "Joran", "Jorge", "Julia Van Santen", "Konstantin", "LeÏse", "Leoncio", "Lucia", "Luenkan", "Malik",
    "Maryna", "Maud", "Megan", "Mery", "Misha", "Swalla", "Neal", "Nikita", "Saul", "Silven", "Stijn",
    "Tessa D.", "Tessa H.", "Tiago", "Thijmen", "Thys", "Vasko", "Xenia", "Ziga", "Stan", "Musa", "Maja",
    "Chloee Coolens", "Carlos"
]

# Define variants for commonly misspelled staff names
STAFF_NAME_VARIANTS = {
    "Nikita": ["Nikkita", "Nekita", "Nikta", "Nickita"],
    "Camila": ["Camilia", "Camilla", "Camela", "Kamila"],
}

# Extract first names and handle duplicates (e.g., Tessa D. and Tessa H. -> Tessa)
def get_first_names(staff_list: List[str]) -> List[str]:
    first_names = []
    seen = set()
    for name in staff_list:
        first_name = re.split(r'\s+|\.', name)[0].strip()
        if first_name.lower() == 'tessa':
            first_name = 'Tessa'
        if first_name.lower() not in seen:
            first_names.append(first_name)
            seen.add(first_name.lower())
    return sorted(first_names)

STAFF_FIRST_NAMES = get_first_names(STAFF_NAMES)

# --- Helper Functions ---
def create_styled_metric(label: str, value_str: str, background_color: str = "#510f30", text_color: str = "white") -> str:
    style = (
        f"background-color: {background_color}; "
        "padding: 1rem; "
        "border-radius: 8px; "
        "text-align: center; "
        "height: 100%; "
        "display: flex; "
        "flex-direction: column; "
        "justify-content: center; "
    )
    html = f"""
<div style="{style}">
    <div style="font-size: 0.875rem; color: {text_color}; margin-bottom: 0.25rem; line-height: 1.3;">{label}</div>
    <div style="font-size: 1.75rem; font-weight: 600; color: {text_color}; line-height: 1.3;">{value_str}</div>
</div>
"""
    return html

def extract_top_keywords(reviews: pd.Series, staff_name: str, num_keywords: int = 5) -> List[str]:  
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = []
    
    staff_name_lower = staff_name.lower()
    for review in reviews.dropna():
        tokens = word_tokenize(str(review).lower())
        filtered_words = [
            lemmatizer.lemmatize(word) for word in tokens 
            if word.isalpha() and word not in stop_words and word != staff_name_lower and len(word) > 3
        ]
        words.extend(filtered_words)
    
    word_counts = Counter(words)
    return [word for word, _ in word_counts.most_common(num_keywords)]

def extract_mentioned_languages(reviews: pd.Series, valid_languages: set) -> str:
    """
    Checks reviews for mentions of languages in valid_languages.
    Returns comma-separated string of mentioned languages or 'None'.
    """
    mentioned_languages = set()
    for review in reviews.dropna():
        review_lower = str(review).lower()
        for lang in valid_languages:
            if re.search(r'\b' + re.escape(lang.lower()) + r'\b', review_lower):
                mentioned_languages.add(lang)
    return ', '.join(sorted(mentioned_languages)) if mentioned_languages else 'None'

def analyze_staff_mentions(
    df: pd.DataFrame,
    staff_list: List[str],
    review_col: str = 'Review',
    rating_col: str = 'Rating',
    sentiment_col: str = 'compound'
) -> pd.DataFrame:
    staff_performance_data = []
    if df is None or df.empty or not staff_list:
        return pd.DataFrame(columns=['Staff Name', 'Number of Mentions', 'Average Rating of Mentions', 'Average Sentiment of Mentions', 'Mentioned Reviews'])

    for staff_name in staff_list:
        variants = STAFF_NAME_VARIANTS.get(staff_name, [])
        all_names = [staff_name] + variants
        pattern = r'\b(' + '|'.join(re.escape(name) for name in all_names) + r')\b'
        exact_matches = df[df[review_col].str.contains(pattern, case=False, na=False, regex=True)]
        
        fuzzy_matches = []
        for idx, review in df[review_col].dropna().items():
            words = re.findall(r'\b\w+\b', str(review).lower())
            for word in words:
                if len(word) >= len(staff_name) - 1 and len(word) <= len(staff_name) + 1:
                    if fuzz.ratio(word, staff_name.lower()) > 90:
                        if staff_name.lower() == 'stan' and word.lower() in ['stand', 'standing', 'stood']:
                            continue
                        fuzzy_matches.append(idx)
        fuzzy_matches_df = df.loc[fuzzy_matches].drop_duplicates()
        
        combined_reviews_df = pd.concat([exact_matches, fuzzy_matches_df]).drop_duplicates()
        
        num_mentions = len(combined_reviews_df)
        avg_rating = combined_reviews_df[rating_col].mean() if num_mentions > 0 and rating_col in combined_reviews_df.columns else None
        avg_sentiment = combined_reviews_df[sentiment_col].mean() if num_mentions > 0 and sentiment_col in combined_reviews_df.columns else None
        staff_performance_data.append({
            'Staff Name': staff_name,
            'Number of Mentions': num_mentions,
            'Average Rating of Mentions': avg_rating,
            'Average Sentiment of Mentions': avg_sentiment,
            'Mentioned Reviews': combined_reviews_df
        })
    return pd.DataFrame(staff_performance_data)

# --- Custom CSS for Centering Table Columns ---
st.markdown("""
    <style>
    div[data-testid="stDataFrame"] .ag-cell,
    div[data-testid="stDataFrame"] .ag-header-cell {
        text-align: center !important;
        justify-content: center !important;
    }
    div[data-testid="stDataFrame"] .ag-header-cell-label {
        display: flex !important;
        justify-content: center !important;
    }
    div[data-testid="stTable"] td,
    div[data-testid="stTable"] th {
        text-align: center !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Logo and Title Section ---
def display_header():
    try:
        logo_path = Path(__file__).resolve().parent.parent / "madam_logo_02.png"
        madam_logo_display = Image.open(logo_path)
        col_title, col_spacer, col_logo = st.columns([0.75, 0.05, 0.2])
        with col_title:
            st.title("Staff Performance Analysis")
            st.markdown("<br>", unsafe_allow_html=True)
        with col_logo:
            st.image(madam_logo_display, width=350)
    except FileNotFoundError:
        st.error(f"Logo image 'madam_logo_02.png' not found at expected path: {logo_path}.")
        st.title("Staff Performance Analysis - Madam")
    except Exception as e:
        st.error(f"An error occurred while loading the logo: {e}")
        st.title("Staff Performance Analysis - Madam")

display_header()

# --- Retrieve Processed Data ---
def load_data() -> Optional[pd.DataFrame]:
    if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
        st.error("Welkom! To begin, please click the '**Home**' page from the sidebar to load the dataset automatically. All pages will be available right after ☺︎")
        return None
    return st.session_state.processed_data.copy()

all_data = load_data()
if all_data is None:
    st.stop()

# --- Sidebar Filters ---
def apply_date_filters(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str], Optional[str]]:
    st.sidebar.header("Date Filters")
    filtered_df = df
    start_date = None
    end_date = None

    if 'Time' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Time']) and not df['Time'].isna().all():
        min_year = df['Time'].dt.year.min() if not df['Time'].empty else 2020
        min_date = pd.to_datetime(f"{min_year}-01-01").date()
        max_date = df['Time'].max().date()
        default_start = min_date
        default_end = max_date
        if default_start > default_end:
            default_start = default_end

        start_date = st.sidebar.date_input(
            "Start date",
            default_start,
            min_value=min_date,
            max_value=max_date,
            key="staff_start_date"
        )
        end_date = st.sidebar.date_input(
            "End date",
            default_end,
            min_value=min_date,
            max_value=max_date,
            key="staff_end_date"
        )

        if start_date > end_date:
            st.sidebar.error("Error: Start date cannot be after end date.")
            filtered_df = pd.DataFrame(columns=df.columns)
        else:
            start_datetime = pd.to_datetime(start_date)
            end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            filtered_df = df[
                (df['Time'] >= start_datetime) & (df['Time'] <= end_datetime)
            ]
        st.sidebar.markdown("❕If this page displays the Home page and errors, click '**Staff Performance**'again to fix.")

    else:
        st.sidebar.warning("Date filter cannot be applied: 'Time' column issue.")
    
    return filtered_df, start_date, end_date

filtered_data, start_date, end_date = apply_date_filters(all_data)

# --- Date Range Display ---
if start_date and end_date and start_date <= end_date:
    st.markdown(f"From **{start_date.strftime('%Y-%m-%d')}** to **{end_date.strftime('%Y-%m-%d')}**")
else:
    st.markdown("**Full Dataset**")

# --- Staff Performance Analysis ---
st.subheader("▸ Staff Performance Analysis")
st.markdown("Performance metrics for staff members based on review mentions, accounting for typos.")

@st.cache_data
def compute_staff_metrics(df: pd.DataFrame, staff_list: List[str]) -> pd.DataFrame:
    return analyze_staff_mentions(df, staff_list)

if filtered_data is None or filtered_data.empty:
    st.warning("No review data available in the selected period for staff analysis.")
else:
    staff_summary_df = compute_staff_metrics(filtered_data, STAFF_FIRST_NAMES)
    
    if staff_summary_df.empty:
        st.info("No data found for staff names in the selected period.")
    else:
        # --- KPI Display ---
        mentioned_staff = staff_summary_df[staff_summary_df['Number of Mentions'] > 0]
        kpi_top_1 = "N/A"
        kpi_top_2 = "N/A"
        kpi_top_3 = "N/A"
        
        if not mentioned_staff.empty:
            sorted_staff = mentioned_staff.sort_values('Number of Mentions', ascending=False)
            # Group by Number of Mentions to handle ties
            grouped = sorted_staff.groupby('Number of Mentions')['Staff Name'].apply(list).reset_index()
            grouped = grouped.sort_values('Number of Mentions', ascending=False)
            
            # Initialize lists for top 3 ranks
            top_1_names = []
            top_2_names = []
            top_3_names = []
            
            # Assign names to top 3 ranks, handling ties
            if len(grouped) >= 1:
                top_1_names = grouped.iloc[0]['Staff Name']
                kpi_top_1 = ', '.join(top_1_names)
            if len(grouped) >= 2:
                top_2_names = grouped.iloc[1]['Staff Name']
                kpi_top_2 = ', '.join(top_2_names)
            if len(grouped) >= 3:
                top_3_names = grouped.iloc[2]['Staff Name']
                kpi_top_3 = ', '.join(top_3_names)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(create_styled_metric("Top Mentioned Staff 1", kpi_top_1), unsafe_allow_html=True)
        with col2:
            st.markdown(create_styled_metric("Top Mentioned Staff 2", kpi_top_2), unsafe_allow_html=True)
        with col3:
            st.markdown(create_styled_metric("Top Mentioned Staff 3", kpi_top_3), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Summary Table ---
        st.markdown("##### Staff Performance Summary")
        display_summary_df = staff_summary_df[['Staff Name', 'Number of Mentions', 'Average Rating of Mentions', 'Average Sentiment of Mentions']].copy()
        display_summary_df['Number of Mentions'] = display_summary_df['Number of Mentions'].map(lambda x: f"{int(x)}" if pd.notnull(x) else "N/A")
        display_summary_df['Average Rating of Mentions'] = display_summary_df['Average Rating of Mentions'].map(lambda x: f"{x:.2f} ⭐" if pd.notnull(x) else "N/A")
        display_summary_df['Average Sentiment of Mentions'] = display_summary_df['Average Sentiment of Mentions'].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
        st.dataframe(
            display_summary_df.sort_values(by='Number of Mentions', ascending=False, key=lambda x: x.astype(int)),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Staff Name": st.column_config.TextColumn(
                    help="Name of the staff member",
                    width="medium"
                ),
                "Number of Mentions": st.column_config.TextColumn(
                    help="Total mentions in reviews",
                    width="small"
                ),
                "Average Rating of Mentions": st.column_config.TextColumn(
                    help="Average rating of reviews mentioning the staff",
                    width="small"
                ),
                "Average Sentiment of Mentions": st.column_config.TextColumn(
                    help="Average sentiment score of reviews mentioning the staff",
                    width="small"
                )
            },
            column_order=["Staff Name", "Number of Mentions", "Average Rating of Mentions", "Average Sentiment of Mentions"]
        )

        # --- Visualizations ---
        if not mentioned_staff.empty:
            st.markdown("##### Performance Visualizations")
            
            # Bar Chart for Mentions
            mentioned_staff_sorted = mentioned_staff.sort_values('Number of Mentions', ascending=False)
            fig_mentions = px.bar(
                mentioned_staff_sorted,
                x='Staff Name',
                y='Number of Mentions',
                title="Mentions per Staff Member",
                text='Number of Mentions',
                color='Number of Mentions',
                color_continuous_scale=['#d4a5b7', '#510f30']
            )
            fig_mentions.update_layout(
                xaxis_title="Staff Name",
                yaxis_title="Number of Mentions",
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_tickangle=45,
                font=dict(size=12),
                margin=dict(t=50, l=25, r=25, b=100),
                showlegend=False
            )
            fig_mentions.update_traces(
                textposition='outside',
                textfont=dict(size=10)
            )
            st.plotly_chart(fig_mentions, use_container_width=True)

            # --- Top Keywords ---
            st.markdown("##### Top Keywords in Staff Reviews")
            st.markdown("*Only top 3 most mentioned staff are included.*")
            keyword_data = []
            if not mentioned_staff.empty:
                valid_languages = set(all_data['Language'].dropna().unique())
                top_3_staff = mentioned_staff.sort_values('Number of Mentions', ascending=False).head(3)
                for _, row in top_3_staff.iterrows():
                    staff_name = row['Staff Name']
                    reviews_df = row['Mentioned Reviews']
                    top_keywords = extract_top_keywords(reviews_df['Review'], staff_name) if not reviews_df.empty else []
                    keywords_str = ', '.join(top_keywords + ['N/A'] * (5 - len(top_keywords))) if top_keywords else ', '.join(['N/A'] * 5)
                    mentioned_langs = extract_mentioned_languages(reviews_df['Review'], valid_languages) if not reviews_df.empty else 'None'
                    keyword_data.append({
                        'Staff Name': staff_name,
                        'Top Keywords': keywords_str,
                        'Mentioned Languages': mentioned_langs
                    })
            if keyword_data:
                keyword_df = pd.DataFrame(keyword_data)
                st.dataframe(
                    keyword_df.sort_values(by='Staff Name'),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Staff Name": st.column_config.TextColumn(
                            help="Name of the staff member",
                            width="medium"
                        ),
                        "Top Keywords": st.column_config.TextColumn(
                            help="Top 5 keywords associated with the staff, comma-separated",
                            width="large"
                        ),
                        "Mentioned Languages": st.column_config.TextColumn(
                            help="Languages mentioned in reviews associated with the staff, comma-separated",
                            width="medium"
                        )
                    },
                    column_order=["Staff Name", "Top Keywords", "Mentioned Languages"]
                )
            else:
                st.info("No staff with mentions found for keyword analysis.")
                
            # --- Review Explorer ---
            st.markdown("##### Explore Staff Reviews")
            selected_staff_review = st.selectbox(
                "Select staff to view reviews:",
                options=mentioned_staff['Staff Name'].tolist(),
                key="staff_review_select"
            )
            reviews_df = mentioned_staff[mentioned_staff['Staff Name'] == selected_staff_review]['Mentioned Reviews'].iloc[0]
            if not reviews_df.empty:
                for _, row in reviews_df.iterrows():
                    st.markdown(f"**Reviewer:** {row.get('Name', 'N/A')} | **Date:** {row['Time'].strftime('%Y-%m-%d %H:%M') if pd.notnull(row['Time']) else 'N/A'} | **Rating:** {row.get('Rating', 'N/A')} ⭐ | **Sentiment:** {row.get('label', 'N/A')} ({row.get('compound', 0.0):.2f})")
                    st.markdown(f"> _{row['Review']}_")
                    st.markdown("---")
            else:
                st.info(f"No specific review details found for {selected_staff_review}.")
        else:
            st.info("No mentions found for staff names in the selected period.")