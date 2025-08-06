# pages/06_Staff_Performance.py

import streamlit as st
import pandas as pd
import plotly.express as px
from fuzzywuzzy import fuzz # Keep this import, but handle gracefully if not available
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

# --- Define Base Directory for favicon ---
BASE_DIR = Path(__file__).resolve().parent.parent
LOGO_PATH = BASE_DIR / "madam_logo_01.png"

# --- START: Duplicated download_nltk_resources function (to avoid importing Home.py) ---
# This is a workaround due to constraints, ideally it would be in a shared utility file.
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
            ('tokenizers/punkt', 'punkt')
            # 'tokenizers/punkt_tab' is often not needed and can cause issues if not found
        ]:
            try:
                nltk.data.find(resource)
            except LookupError:
                nltk.download(download_name, quiet=True)
        return True
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")
        return False
# --- END: Duplicated download_nltk_resources function ---

# --- NLTK Resource Download ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Add this to download 'punkt_tab'
try:
    nltk.data.find('tokenizers/punkt_tab') # Corrected path for punkt_tab if it's a tokenizer
except LookupError:
    nltk.download('punkt_tab', quiet=True)


# --- Configuration ---
STAFF_NAMES = [
    "Aimée", "Alhassan", "Bas", "Britt", "Camila", "Daaf", "Darren", "Yesni", "Dimitteo", "Dimitris",
    "Ektor", "Eleonore", "Georgios Moy", "Georgios Myro", "Vikram", "Jacky", "Janet", "Jason", "Jody",
    "Joran", "Jorge", "Julia Van Santen", "Konstantin", "LeÏse", "Leoncio", "Lucia", "Luenkan", "Malik",
    "Maryna", "Maud", "Megan", "Mery", "Misha", "Swalla", "Neal", "Nikita", "Saul", "Silven", "Stijn",
    "Tessa D.", "Tessa H.", "Tiago", "Thijmen", "Thys", "Vasko", "Xenia", "Yesni", "Ziga", "Stan", "Musa", "Maja",
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
    # Default NLTK stopwords
    stop_words = set(stopwords.words('english'))
    # Custom stopwords that are common in reviews but might not be insightful as keywords
    # These are general positive/negative terms that might dilute more specific feedback.
    custom_stopwords = {'great', 'good', 'amazing', 'best', 'very', 'really', 'much',
                        'thank', 'thanks', 'nice', 'friendly', 'excellent', 'super',
                        'always', 'definitely', 'highly', 'recommend', 'perfect', 'awesome', 'wonderful',
                        'staff', 'service', 'madam', 'place', 'restaurant', 'food'} # Added more general review terms
    all_stop_words = stop_words.union(custom_stopwords)

    lemmatizer = WordNetLemmatizer()
    all_filtered_words = []
    staff_name_lower = staff_name.lower()
    staff_first_name_only_lower = staff_name_lower.split(' ')[0]

    # Pre-calculate all staff name variations in lowercase for robust filtering
    all_staff_name_variants_lower = {staff_name_lower, staff_first_name_only_lower}
    for full_name in STAFF_NAMES: # Use full STAFF_NAMES to get all possible forms
        all_staff_name_variants_lower.add(full_name.lower())
        all_staff_name_variants_lower.add(full_name.lower().split(' ')[0]) # Add just first name
    for variant_list in STAFF_NAME_VARIANTS.values(): # Add specific variants too
        all_staff_name_variants_lower.update(v.lower() for v in variant_list)


    for review in reviews.dropna():
        tokens = word_tokenize(str(review).lower())
        
        # Filter tokens: alphabetic, not in stopwords, not a staff name/variant, min length
        filtered_tokens_for_ngrams = [
            lemmatizer.lemmatize(word) for word in tokens
            if word.isalpha() and
               word not in all_stop_words and
               word not in all_staff_name_variants_lower and # Use the expanded set for filtering
               len(word) > 2 # Allow words like "bar", "tip"
        ]
        all_filtered_words.extend(filtered_tokens_for_ngrams)

    # Count unigrams (single words)
    unigram_counts = Counter(all_filtered_words)

    # Count bigrams (two-word phrases)
    bigrams = []
    if len(all_filtered_words) > 1:
        for i in range(len(all_filtered_words) - 1):
            bigram = f"{all_filtered_words[i]} {all_filtered_words[i+1]}"
            bigrams.append(bigram)
    bigram_counts = Counter(bigrams)

    # Combine unigram and bigram counts
    # Using 'update' allows adding counts from bigrams to unigrams,
    # treating bigrams as distinct items for commonality.
    combined_counts = Counter()
    combined_counts.update(unigram_counts)
    combined_counts.update(bigram_counts)

    # Filter out any keywords (unigram or bigram) that still contain staff names or their parts
    final_keywords = []
    for word, _ in combined_counts.most_common():
        is_staff_name_related = False
        # Check if the keyword (unigram or bigram) contains any part of a staff name
        for name_part in all_staff_name_variants_lower: # Use the comprehensive set
            if name_part in word: # Check if the staff name is part of the keyword
                is_staff_name_related = True
                break
        
        if not is_staff_name_related:
            final_keywords.append(word)
        
        if len(final_keywords) >= num_keywords:
            break

    return final_keywords


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
        
        # Pattern for exact matches
        pattern = r'\b(' + '|'.join(re.escape(name) for name in all_names) + r')\b'
        
        # Filter reviews for exact matches first
        exact_matches = df[df[review_col].str.contains(pattern, case=False, na=False, regex=True)].copy()

        # Fuzzy matching logic (this part might require fuzzywuzzy library)
        # If fuzzywuzzy is not installed, this part will be skipped or cause an error.
        # For robustness, we will try to use it but ensure the rest works without it.
        fuzzy_matches_df = pd.DataFrame()
        try:
            from fuzzywuzzy import fuzz # Re-import here to ensure it's checked at runtime
            fuzzy_matches_indices = []
            for idx, review in df[review_col].dropna().items():
                words = re.findall(r'\b\w+\b', str(review).lower())
                for word in words:
                    if len(word) >= len(staff_name) - 1 and len(word) <= len(staff_name) + 1:
                        if fuzz.ratio(word, staff_name.lower()) > 90:
                            # Specific exclusion for 'stan' to avoid common English words
                            if staff_name.lower() == 'stan' and word.lower() in ['stand', 'standing', 'stood']:
                                continue
                            fuzzy_matches_indices.append(idx)
            fuzzy_matches_df = df.loc[fuzzy_matches_indices].drop_duplicates().copy()
        except ImportError:
            # print("Warning: fuzzywuzzy not installed. Staff name matching will be exact only.")
            pass # Silently proceed without fuzzy matching if not available

        # Combine exact and fuzzy matches (if fuzzy_matches_df is not empty)
        if not fuzzy_matches_df.empty:
            combined_reviews_df = pd.concat([exact_matches, fuzzy_matches_df]).drop_duplicates()
        else:
            combined_reviews_df = exact_matches.copy() # If no fuzzy matches or fuzzywuzzy not found

        num_mentions = len(combined_reviews_df)
        avg_rating = combined_reviews_df[rating_col].mean() if num_mentions > 0 and rating_col in combined_reviews_df.columns else None
        avg_sentiment = combined_reviews_df[sentiment_col].mean() if num_mentions > 0 and sentiment_col in combined_reviews_df.columns else None
        
        staff_performance_data.append({
            'Staff Name': staff_name,
            'Number of Mentions': num_mentions,
            'Average Rating of Mentions': avg_rating,
            'Average Sentiment of Mentions': avg_sentiment,
            'Mentioned Reviews': combined_reviews_df # Keep the DataFrame for keyword/language extraction later
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
        # Note: This logo_path is for the image displayed within the page, not the favicon.
        logo_path = BASE_DIR / "madam_logo_02.png" # Assuming madam_logo_02.png is in the main directory
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
        # --- KPI Display (Improved Tie Handling for Display) ---
        mentioned_staff = staff_summary_df[staff_summary_df['Number of Mentions'] > 0].copy() # Ensure copy to avoid SettingWithCopyWarning
        
        kpi_top_1 = "N/A"
        kpi_top_2 = "N/A"
        kpi_top_3 = "N/A"
        
        if not mentioned_staff.empty:
            # Sort by mentions in descending order
            sorted_staff = mentioned_staff.sort_values('Number of Mentions', ascending=False)
            
            # Get unique mention counts in descending order
            unique_mention_counts = sorted_staff['Number of Mentions'].unique()

            # Assign names to top 3 ranks, ensuring all ties are included
            if len(unique_mention_counts) >= 1:
                top_1_count = unique_mention_counts[0]
                top_1_names = sorted_staff[sorted_staff['Number of Mentions'] == top_1_count]['Staff Name'].tolist()
                kpi_top_1 = ', '.join(sorted(top_1_names)) # Sort names alphabetically for consistency
            
            if len(unique_mention_counts) >= 2:
                top_2_count = unique_mention_counts[1]
                top_2_names = sorted_staff[sorted_staff['Number of Mentions'] == top_2_count]['Staff Name'].tolist()
                kpi_top_2 = ', '.join(sorted(top_2_names))
            
            if len(unique_mention_counts) >= 3:
                top_3_count = unique_mention_counts[2]
                top_3_names = sorted_staff[sorted_staff['Number of Mentions'] == top_3_count]['Staff Name'].tolist()
                kpi_top_3 = ', '.join(sorted(top_3_names))

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

            # --- Top Keywords (Modified to include all tied staff) ---
            st.markdown("##### Top Keywords in Staff Reviews")
            st.markdown("*Keywords are shown for the staff members in the top 3 tiers of mentions, including all ties.*") # Updated description
            keyword_data = []
            
            if not mentioned_staff.empty:
                # Re-sort to ensure consistency and get unique tiers
                sorted_staff_for_keywords_analysis = mentioned_staff.sort_values('Number of Mentions', ascending=False)
                unique_mention_counts_for_keywords = sorted_staff_for_keywords_analysis['Number of Mentions'].unique()
                
                staff_to_process_for_keywords = pd.DataFrame()
                
                # Collect staff from the top 3 tiers of mention counts
                if len(unique_mention_counts_for_keywords) >= 1:
                    tier1_count = unique_mention_counts_for_keywords[0]
                    staff_to_process_for_keywords = pd.concat([staff_to_process_for_keywords, sorted_staff_for_keywords_analysis[sorted_staff_for_keywords_analysis['Number of Mentions'] == tier1_count]])
                
                if len(unique_mention_counts_for_keywords) >= 2:
                    tier2_count = unique_mention_counts_for_keywords[1]
                    staff_to_process_for_keywords = pd.concat([staff_to_process_for_keywords, sorted_staff_for_keywords_analysis[sorted_staff_for_keywords_analysis['Number of Mentions'] == tier2_count]])
                
                if len(unique_mention_counts_for_keywords) >= 3:
                    tier3_count = unique_mention_counts_for_keywords[2]
                    staff_to_process_for_keywords = pd.concat([staff_to_process_for_keywords, sorted_staff_for_keywords_analysis[sorted_staff_for_keywords_analysis['Number of Mentions'] == tier3_count]])
                
                # Remove any duplicates that might arise from concat if a staff member was somehow in multiple tiers (unlikely but safe)
                staff_to_process_for_keywords = staff_to_process_for_keywords.drop_duplicates(subset=['Staff Name']).copy()


                for _, row in staff_to_process_for_keywords.iterrows(): # Iterate over the correctly selected staff
                    staff_name = row['Staff Name']
                    reviews_df = row['Mentioned Reviews']
                    
                    top_keywords = extract_top_keywords(reviews_df['Review'], staff_name) if not reviews_df.empty else []
                    
                    # Ensure top_keywords list has enough items to fill 5 slots, using 'N/A' for missing
                    keywords_str = ', '.join((top_keywords + ['N/A'] * 5)[:5]) if top_keywords else ', '.join(['N/A'] * 5)
                    
                    # Extract unique original languages from the 'Language' column of the mentioned reviews
                    mentioned_original_languages = reviews_df['Language'].dropna().unique().tolist()
                    mentioned_langs_str = ', '.join(sorted(mentioned_original_languages)) if mentioned_original_languages else 'None'

                    keyword_data.append({
                        'Staff Name': staff_name,
                        'Top Keywords': keywords_str,
                        'Mentioned Languages': mentioned_langs_str # This is now the original language of the review
                    })
            if keyword_data:
                keyword_df = pd.DataFrame(keyword_data)
                st.dataframe(
                    keyword_df.sort_values(by='Staff Name'), # Sort by staff name for consistent display
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Staff Name": st.column_config.TextColumn(
                            help="Name of the staff member",
                            width="medium"
                        ),
                        "Top Keywords": st.column_config.TextColumn(
                            help="Top 5 keywords (unigrams or bigrams) associated with the staff, comma-separated",
                            width="large"
                        ),
                        "Mentioned Languages": st.column_config.TextColumn(
                            help="Original languages of reviews mentioning the staff, comma-separated",
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
                    st.markdown(f"**Reviewer:** {row.get('Name', 'N/A')} | **Date:** {row['Time'].strftime('%Y-%m-%d %H:%M') if pd.notnull(row['Time']) else 'N/A'} | **Rating:** {row.get('Rating', 'N/A')} ⭐ | **Sentiment:** {row.get('label', 'N/A')} ({row.get('compound', 0.0):.2f}) | **Original Language:** {row.get('Language', 'N/A')}")
                    st.markdown(f"> _{row['Review']}_")
                    st.markdown("---")
            else:
                st.info(f"No specific review details found for {selected_staff_review}.")
        else:
            st.info("No mentions found for staff names in the selected period.")