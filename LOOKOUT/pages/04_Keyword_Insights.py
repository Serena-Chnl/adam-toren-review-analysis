# pages/04_Keyword_Analysis.py

import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
from collections import Counter
import nltk
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
from PIL import Image # Import Image
from pathlib import Path # Import Path
os.environ["PLOTLY_PANDAS_BACKEND"] = "pandas"

# --- Page Configuration ---
# Define the base directory for this script
BASE_DIR = Path(__file__).resolve().parent

# Path to lookout_logo_01.png (assuming it's in the parent directory of 'pages')
LOOKOUT_LOGO_PATH = BASE_DIR.parent / "lookout_logo_01.png"

page_icon_keyword_analysis = "💡" # Default emoji icon for this page
try:
    img_icon_keyword_analysis = Image.open(LOOKOUT_LOGO_PATH)
    page_icon_keyword_analysis = img_icon_keyword_analysis
except Exception as e:
    st.warning(f"Cannot use '{LOOKOUT_LOGO_PATH.name}' as page icon for Keyword Analysis: {e}. Using default emoji icon.")

st.set_page_config(page_title="Keyword Analysis - A'DAM LOOKOUT", page_icon=page_icon_keyword_analysis, layout="wide")

# --- NLTK Resource Download ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Helper function for styled metrics ---
def create_styled_metric(label, value_str, background_color="#5a5a5a", text_color="#ffffff"):
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

# --- Title and Logo Section ---
col_title, col_spacer, col_logo = st.columns([0.65, 0.05, 0.3])
with col_title:
    st.title("Keyword Analysis")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
with col_logo:
    try:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(current_script_dir, "..", "lookout_logo_02.png")
        st.image(logo_path, width=550)
    except FileNotFoundError:
        st.error(f"Logo image 'lookout_logo_02.png' not found. Please check the path: {logo_path}.")
    except Exception as e:
        st.error(f"An error occurred while loading the logo: {e}. Ensure the file is a valid image and the path is correct: {logo_path}")



# --- Retrieve Processed Data from Session State ---
if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
    st.error("Welkom! To begin, please click the '**Home**' page from the sidebar to load the dataset automatically. All pages will be available right after ☺︎")
    st.stop()

all_data = st.session_state.processed_data.copy()

# --- Sidebar for Date Range Filter ---
st.sidebar.header("Keyword Analysis Filters")
filtered_data_for_keywords = all_data

if 'Time' in all_data.columns and pd.api.types.is_datetime64_any_dtype(all_data['Time']) and not all_data['Time'].isna().all():
    min_year = all_data['Time'].dt.year.min() if not all_data['Time'].empty else 2020
    min_date_data = pd.to_datetime(f"{min_year}-01-01").date()
    max_date_data = all_data['Time'].max().date()
    default_start = min_date_data
    default_end = max_date_data
    if default_start > default_end:
        default_start = default_end
    start_date_kw = st.sidebar.date_input(
        "Start date",
        default_start,
        min_value=min_date_data,
        max_value=max_date_data,
        key="keywords_start_date"
    )
    end_date_kw = st.sidebar.date_input(
        "End date",
        default_end,
        min_value=min_date_data,
        max_value=max_date_data,
        key="keywords_end_date"
    )
    if start_date_kw > end_date_kw:
        st.sidebar.error("Error: Start date cannot be after end date.")
        filtered_data_for_keywords = pd.DataFrame(columns=all_data.columns)
    else:
        start_datetime_kw = pd.to_datetime(start_date_kw)
        end_datetime_kw = pd.to_datetime(end_date_kw) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        filtered_data_for_keywords = all_data[
            (all_data['Time'] >= start_datetime_kw) &
            (all_data['Time'] <= end_datetime_kw)
        ]
else:
    st.sidebar.warning("Date filter cannot be applied: 'Time' column is missing, not datetime, or all invalid.")

# --- Text Preprocessing and Keyword Extraction ---
lemmatizer = WordNetLemmatizer()
stop_words_english = set(stopwords.words('english'))
# Custom stopwords tailored for A'DAM LOOKOUT
custom_stopwords = {
    'lookout', 'adam', 'amsterdam', 'place', 'also', 'get', 'got', 'would', 'could', 'like', 'time', 'experience',
    'really', 'one', 'even', 'us', 'went', 'came', 'told', 'asked', 'staff', 'employee', 'guide'
}
stop_words_english.update(custom_stopwords)

# Whitelist for key attraction-related terms (expanded for new sections)
attraction_whitelist = {
    'view', 'views', 'panoramic', 'scenery', 'skyline', 'rooftop', 'vista',
    'swing', 'over the edge', 'thrill', 'ride',
    'vr', 'vr ride', 'virtual reality', 'simulation',
    'photo', 'photos', 'picture', 'pictures', 'watermark', 'snapshot', 'image',
    'ticket', 'combi', 'combo', 'entry', 'admission', 'pass',
    'madam', 'restaurant', 'burger', 'drink', 'meal', 'dining', 'food',
    'staff', 'employee', 'guide', 'crew', 'attendant', 'service',
    'queue', 'line', 'wait', 'waiting', 'crowd', 'delay',
    'price', 'cost', 'expensive', 'pricey', 'fee', 'charge',
    'experience', 'activity', 'attraction', 'fun', 'adventure',
    'weather', 'rain', 'wind', 'sun', 'sunny', 'cloudy', 'cold', 'hot' # Added for weather
}

def preprocess_text_for_keywords(text):
    if pd.isna(text):
        return []
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    lemmatized_words = [
        lemmatizer.lemmatize(word) for word in words
        if (word not in stop_words_english and len(word) > 2) or word in attraction_whitelist
    ]
    return lemmatized_words

def preprocess_text_for_wordcloud(text):
    if pd.isna(text):
        return []
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    lemmatized_words = [
        lemmatizer.lemmatize(word) for word in words
        if (word in attraction_whitelist or word in {'amazing', 'thrilling', 'bad', 'poor', 'expensive', 'pricey'})
    ]
    return lemmatized_words

# --- Calculate KPIs for Most Mentioned Words by Rating ---
if 'Review' in all_data.columns and 'Rating' in all_data.columns:
    keyword_groups = {
        'view': ['view', 'views', 'panoramic', 'scenery'],
        'swing': ['swing', 'over the edge'],
        'vr_ride': ['vr', 'ride', 'vr ride'],
        'photo': ['photo', 'photos', 'watermark'],
        'ticket': ['ticket', 'combi', 'combo']
    }
    kpi_words = {5: "N/A", 4: "N/A", 3: "N/A", 2: "N/A", 1: "N/A"}
    for rating in range(1, 6):
        rating_data = all_data[all_data['Rating'] == rating]
        all_words = []
        for review_text in rating_data['Review']:
            all_words.extend(preprocess_text_for_keywords(review_text))
        if all_words:
            word_counts = Counter(all_words)
            group_counts = {key: 0 for key in keyword_groups}
            for word, count in word_counts.items():
                for group, synonyms in keyword_groups.items():
                    if word in synonyms:
                        group_counts[group] += count
            top_group = max(group_counts, key=group_counts.get, default="N/A")
            if group_counts[top_group] > 0:
                kpi_words[rating] = top_group.replace('_', ' ').capitalize()
            else:
                kpi_words[rating] = "N/A"

# Function to create bar chart for keywords (reusable)
def create_keyword_chart(keywords, title, sentiment):
    if not keywords:
        return None
    df_keywords = pd.DataFrame(keywords, columns=['Keyword', 'Count'])
    bar_color = '#a3bbce' if sentiment == 'positive' else '#d8575d'
    fig = go.Figure(data=[
        go.Bar(
            x=df_keywords['Count'],
            y=df_keywords['Keyword'].str.capitalize(),
            orientation='h',
            marker_color=bar_color
        )
    ])
    fig.update_layout(
        title=title,
        xaxis_title="Frequency",
        yaxis_title="",
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='#f8f9fa',
        margin=dict(t=50, l=25, r=25, b=25),
        height=300
    )
    return fig

# --- Main Page Content ---
if filtered_data_for_keywords is None or filtered_data_for_keywords.empty:
    if 'start_date_kw' in locals():
        st.warning(f"No review data available for keyword analysis in the selected period: {start_date_kw.strftime('%Y-%m-%d')} to {end_date_kw.strftime('%Y-%m-%d')}.")
    else:
        st.warning("No data available for keyword analysis. This might be due to initial data loading issues or problems with the 'Time' column.")
else:
    if 'start_date_kw' in locals() and 'end_date_kw' in locals():
        st.write(f"From **{start_date_kw.strftime('%Y-%m-%d')}** to **{end_date_kw.strftime('%Y-%m-%d')}**")
    else:
        st.write("Overall Keyword Insights (Full Dataset)")

    # Display KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(create_styled_metric("5-Star Top Word", kpi_words[5]), unsafe_allow_html=True)
    with col2:
        st.markdown(create_styled_metric("4-Star Top Word", kpi_words[4]), unsafe_allow_html=True)
    with col3:
        st.markdown(create_styled_metric("3-Star Top Word", kpi_words[3]), unsafe_allow_html=True)
    with col4:
        st.markdown(create_styled_metric("2-Star Top Word", kpi_words[2]), unsafe_allow_html=True)
    with col5:
        st.markdown(create_styled_metric("1-Star Top Word", kpi_words[1]), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        .keyword-filters {
            font-size: 24px !important;
        }
        </style>
        <div class='keyword-filters'>Keyword Filters</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    sentiment_filter_kw = st.selectbox(
        "Select Sentiment:",
        options=['All Sentiments', 'Positive Reviews', 'Negative Reviews', 'Neutral Reviews'],
        index=0,
        key="sentiment_filter_keywords"
    )

    num_keywords = st.slider("Number of top keywords for visualization:", min_value=5, max_value=30, value=15, key="num_keywords_slider")
    st.markdown("<br>", unsafe_allow_html=True)

    if sentiment_filter_kw == 'Negative Reviews':
        st.info("""
        **Note on Keywords in Negative Reviews:**
        Words like "view" or "swing" may appear in negative reviews if mentioned in an overall negative context
        (e.g., "The view was great, but the photo prices were too high.").
        Check the "Review Explorer" page for full context.
        """)

    data_to_analyze_keywords = filtered_data_for_keywords
    if sentiment_filter_kw != 'All Sentiments' and 'label' in data_to_analyze_keywords.columns:
        sentiment_label_map = {
            'Positive Reviews': 'positive',
            'Negative Reviews': 'negative',
            'Neutral Reviews': 'neutral'
        }
        selected_sentiment = sentiment_label_map.get(sentiment_filter_kw)
        if selected_sentiment:
            data_to_analyze_keywords = data_to_analyze_keywords[data_to_analyze_keywords['label'] == selected_sentiment]

    if data_to_analyze_keywords.empty:
        st.warning(f"No reviews found for '{sentiment_filter_kw}' in the selected period.")
    elif 'Review' not in data_to_analyze_keywords.columns:
        st.error("'Review' column not found. Cannot perform keyword analysis.")
    else:
        all_words = []
        for review_text in data_to_analyze_keywords['Review']:
            all_words.extend(preprocess_text_for_keywords(review_text))
        
        if not all_words:
            st.info("No keywords found after preprocessing for the current selection.")
        else:
            word_counts = Counter(all_words)
            most_common_words = word_counts.most_common(num_keywords)
            
            if not most_common_words:
                st.info("No common keywords to display for the current selection.")
            else:
                df_most_common = pd.DataFrame(most_common_words, columns=['Keyword', 'Frequency'])
                
                # --- Keyword Visualization ---
                st.subheader(f"▸ Top Keywords Bubble Chart ({sentiment_filter_kw})")
                if not df_most_common.empty:
                    try:
                        df_most_common = df_most_common.dropna(subset=['Keyword', 'Frequency']).copy()
                        df_most_common['Keyword'] = df_most_common['Keyword'].astype(str)
                        df_most_common['Frequency'] = df_most_common['Frequency'].astype(int)
                        fig_bubble = px.scatter(
                            df_most_common,
                            x='Keyword',
                            y='Frequency',
                            size='Frequency',
                            color='Frequency',
                            color_continuous_scale=['#a3bbce', '#d8575d'],  # Positive to negative
                            title=f"Top {min(len(df_most_common), num_keywords)} Keywords ({sentiment_filter_kw})",
                            hover_data={'Keyword': True, 'Frequency': True}
                        )
                        fig_bubble.update_layout(
                            xaxis_title="Keyword",
                            yaxis_title="",
                            showlegend=True,
                            xaxis_tickangle=45,
                            margin=dict(t=50, l=25, r=25, b=25),
                            plot_bgcolor='#f8f9fa',
                            paper_bgcolor='#f8f9fa'
                        )
                        st.plotly_chart(fig_bubble, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating bubble chart: {e}")
                else:
                    st.info("No keywords available to display.")

            # --- Word Cloud ---
            st.subheader(f"▸ Keyword Cloud ({sentiment_filter_kw})")
            all_words_wc = []
            for review_text in data_to_analyze_keywords['Review']:
                all_words_wc.extend(preprocess_text_for_wordcloud(review_text))
            
            if not all_words_wc:
                st.info("No keywords found after preprocessing for the word cloud.")
            else:
                word_counts_wc = Counter(all_words_wc)
                if word_counts_wc:
                    try:
                        wc = WordCloud(
                            width=800,
                            height=400,
                            background_color='#f8f9fa',
                            colormap='viridis',
                            max_words=50,
                            contour_width=1,
                            contour_color='#5a5a5a'
                        ).generate_from_frequencies(dict(word_counts_wc))
                        fig_wc, ax = plt.subplots(figsize=(12, 6))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig_wc)
                    except Exception as e:
                        st.error(f"Could not generate word cloud: {e}")
                else:
                    st.info("Not enough data to generate a word cloud for the current selection.")
st.markdown("---")


# --- Swing, VR Ride, and Photo Feedback Analysis ---
st.markdown("<br>", unsafe_allow_html=True) # Space before the subheader

if 'Review' in filtered_data_for_keywords.columns and 'label' in filtered_data_for_keywords.columns:
    # Swing KPIs calculation
    swing_reviews = filtered_data_for_keywords[
        filtered_data_for_keywords['Review'].str.contains('swing|over the edge', case=False, na=False, regex=True) &
        filtered_data_for_keywords['label'].isin(['positive', 'negative'])
    ]
    total_swing_reviews = len(swing_reviews)
    swing_positive_count = len(swing_reviews[swing_reviews['label'] == 'positive'])
    swing_negative_count = len(swing_reviews[swing_reviews['label'] == 'negative'])
    swing_positive_pct = (swing_positive_count / total_swing_reviews * 100) if total_swing_reviews > 0 else 0
    swing_negative_pct = (swing_negative_count / total_swing_reviews * 100) if total_swing_reviews > 0 else 0

    # VR KPIs calculation
    vr_reviews = filtered_data_for_keywords[
        filtered_data_for_keywords['Review'].str.contains('vr|virtual reality|ride', case=False, na=False, regex=True) &
        filtered_data_for_keywords['label'].isin(['positive', 'negative'])
    ]
    total_vr_reviews = len(vr_reviews)
    vr_positive_count = len(vr_reviews[vr_reviews['label'] == 'positive'])
    vr_negative_count = len(vr_reviews[vr_reviews['label'] == 'negative'])
    vr_positive_pct = (vr_positive_count / total_vr_reviews * 100) if total_vr_reviews > 0 else 0
    vr_negative_pct = (vr_negative_count / total_vr_reviews * 100) if total_vr_reviews > 0 else 0

    # Photo KPIs calculation (Moved here)
    photo_keywords_overall = [r'\bphoto\b', r'\bphotos\b', r'\bpicture\b', r'\bpictures\b', r'\bwatermark\b', r'\bsnapshot\b', r'\bimage\b']
    photo_reviews_overall = filtered_data_for_keywords[
        filtered_data_for_keywords['Review'].str.contains('|'.join(photo_keywords_overall), case=False, regex=True, na=False) &
        filtered_data_for_keywords['label'].isin(['positive', 'negative'])
    ]
    total_photo_reviews_overall = len(photo_reviews_overall)
    photo_positive_count_overall = len(photo_reviews_overall[photo_reviews_overall['label'] == 'positive'])
    photo_negative_count_overall = len(photo_reviews_overall[photo_reviews_overall['label'] == 'negative'])
    photo_positive_pct_overall = (photo_positive_count_overall / total_photo_reviews_overall * 100) if total_photo_reviews_overall > 0 else 0
    photo_negative_pct_overall = (photo_negative_count_overall / total_photo_reviews_overall * 100) if total_photo_reviews_overall > 0 else 0


    st.subheader("▸ Insights on Add-on: Swing, VR Ride, Photo")
    st.markdown(
        """
        <br>
        <div style="font-size: 1.1em; line-height: 1.6;">
        Detailed sentiment breakdown for our popular add-on attractions:
        <ul>
            <li>For reviews that mention "Swing", <b><span style='color:#66c2ff'>{swing_positive_pct:.1f}%</span></b> are positive, and <b><span style='color:#ff6666'>{swing_negative_pct:.1f}%</span></b> are negative.</li>
            <li>For reviews that mention "VR Ride", <b><span style='color:#66c2ff'>{vr_positive_pct:.1f}%</span></b> are positive, and <b><span style='color:#ff6666'>{vr_negative_pct:.1f}%</span></b> are negative.</li>
            <li>For reviews mentioning "Photo", <b><span style='color:#66c2ff'>{photo_positive_pct_overall:.1f}%</span></b> are positive, and <b><span style='color:#ff6666'>{photo_negative_pct_overall:.1f}%</span></b> are negative.</li>
        </ul>
        </div>
        """.format(
            swing_positive_pct=swing_positive_pct,
            swing_negative_pct=swing_negative_pct,
            vr_positive_pct=vr_positive_pct,
            vr_negative_pct=vr_negative_pct,
            photo_positive_pct_overall=photo_positive_pct_overall, # Use the overall photo KPI here
            photo_negative_pct_overall=photo_negative_pct_overall  # Use the overall photo KPI here
        ),
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True) # Add some space after the text summary

    # Extended whitelist with more meaningful keywords for general use in this section
    attraction_whitelist.update({
        'duration', 'short', 'brief', 'quick', 'long', 'length', 'wait', 'queue', 
        'thrilling', 'exciting', 'scary', 'fun', 'enjoyable', 'boring', 'disappointing', 
        'expensive', 'pricey', 'costly', 'worth', 'value', 'cheap', 
        'quality', 'immersive', 'realistic', 'poor', 'bad', 
        'service', 'staff', 'friendly', 'helpful', 'slow', 'rude'
    })

    # Define keywords for Swing, VR, Photo with stricter matching
    swing_keywords = [r'\bover the edge\b', r'\bswing\b']
    vr_keywords = [r'\bvr ride\b', r'\bvirtual reality\b', r'\bvr\b']
    photo_keywords_detail = [r'\bphoto\b', r'\bphotos\b', r'\bpicture\b', r'\bpictures\b', r'\bwatermark\b', r'\bsnapshot\b', r'\bimage\b']


    # Function to extract top keywords for Swing/VR/Photo (made more generic to handle exclusion)
    def extract_top_addon_keywords(reviews, keywords_to_match, sentiment, other_addon_keywords_for_exclusion, max_keywords=5):
        pattern = '|'.join(keywords_to_match)
        # Exclude reviews that primarily focus on other add-ons within this section
        exclude_pattern = '|'.join(other_addon_keywords_for_exclusion)
        
        relevant_reviews = reviews[
            reviews['Review'].str.contains(pattern, case=False, regex=True, na=False) &
            (reviews['label'] == sentiment)
        ]
        if exclude_pattern: # Apply exclusion only if there's a pattern to exclude
             relevant_reviews = relevant_reviews[~relevant_reviews['Review'].str.contains(exclude_pattern, case=False, regex=True, na=False)]

        all_words = []
        
        # Determine specific whitelist based on the current context/sentiment
        if 'photo' in ' '.join(keywords_to_match): # Check if current analysis is for photos
            whitelist = photo_positive_whitelist if sentiment == 'positive' else photo_negative_whitelist
        else: # Default to general attraction_whitelist for Swing/VR
            whitelist = attraction_whitelist
        
        for review_text in relevant_reviews['Review']:
            words = preprocess_text_for_keywords(review_text)
            all_words.extend([
                word for word in words
                if word in whitelist and
                word not in {'swing', 'vr', 'ride', 'virtual', 'reality', 'over', 'edge', 'photo', 'photos', 'picture', 'pictures', 'watermark', 'snapshot', 'image'} # Exclude the main topic words of all add-ons
            ])
        word_counts = Counter(all_words)
        return word_counts.most_common(max_keywords)


    st.markdown("**Select a tab below!**") # New line

    st.markdown("""
    <style>
        /* This CSS targets the Streamlit tab buttons */
        .stTabs [data-baseweb="tab-list"] button {
            margin-right: 40px; /* Adjust this value to increase spacing */
            padding: 10px 15px; /* Optional: Adjust padding inside the buttons if needed */
        }
    </style>
    """, unsafe_allow_html=True)


    
    tab_swing, tab_vr, tab_photo = st.tabs(["**Swing**", "**VR Ride**", "**Photo**"])

    with tab_swing:
        swing_positive_keywords = extract_top_addon_keywords(
            filtered_data_for_keywords, swing_keywords, 'positive', vr_keywords + photo_keywords_detail # Exclude VR and Photo
        )
        swing_negative_keywords = extract_top_addon_keywords(
            filtered_data_for_keywords, swing_keywords, 'negative', vr_keywords + photo_keywords_detail # Exclude VR and Photo
        )


        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Positive Feedback**")
            fig_swing_pos = create_keyword_chart(
                swing_positive_keywords,
                "Top Positive Keywords for Swing",
                'positive'
            )
            if fig_swing_pos:
                st.plotly_chart(fig_swing_pos, use_container_width=True)
            else:
                st.info("No positive keywords available for Swing.")

        with col2:
            st.markdown("**Negative Feedback**")
            fig_swing_neg = create_keyword_chart(
                swing_negative_keywords,
                "Top Negative Keywords for Swing",
                'negative'
            )
            if fig_swing_neg:
                st.plotly_chart(fig_swing_neg, use_container_width=True)
            else:
                st.info("No negative keywords available for Swing.")

    with tab_vr:
        vr_positive_keywords = extract_top_addon_keywords(
            filtered_data_for_keywords, vr_keywords, 'positive', swing_keywords + photo_keywords_detail # Exclude Swing and Photo
        )
        vr_negative_keywords = extract_top_addon_keywords(
            filtered_data_for_keywords, vr_keywords, 'negative', swing_keywords + photo_keywords_detail # Exclude Swing and Photo
        )


        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Positive Feedback**")
            fig_vr_pos = create_keyword_chart(
                vr_positive_keywords,
                "Top Positive Keywords for VR Ride",
                'positive'
            )
            if fig_vr_pos:
                st.plotly_chart(fig_vr_pos, use_container_width=True)
            else:
                st.info("No positive keywords available for VR Ride during the selected period.")

        with col2:
            st.markdown("**Negative Feedback**")
            fig_vr_neg = create_keyword_chart(
                vr_negative_keywords,
                "Top Negative Keywords for VR Ride",
                'negative'
            )
            if fig_vr_neg:
                st.plotly_chart(fig_vr_neg, use_container_width=True)
            else:
                st.info("No negative keywords available for VR Ride.")
    
    with tab_photo: # New Photo Tab

        photo_positive_whitelist = {
            'beautiful', 'clear', 'stunning', 'amazing', 'great', 'lovely', 'nice',
            'quality', 'memorable', 'professional', 'awesome', 'fantastic', 'perfect',
            'free', 'download', 'easy', 'convenient', 'complimentary', 'worth', 'value' 
        }
        photo_negative_whitelist = {
            'expensive', 'overpriced', 'pricey', 'costly', 'blurry', 'poor', 'bad',
            'disappointing', 'annoying', 'unfair', 'low', 'terrible', 'unhappy',
            'scam', 'rip off', 'charge', 'hidden', 'extra', 'pay', 'watermark',
            'confusing', 'misleading', 'difficult', 'forced' 
        }
        # Re-using photo_keywords_overall from KPI calculation for consistency
        exclude_keywords_photo_detail = swing_keywords + vr_keywords # Exclude Swing and VR

        # Redefine extract_top_photo_keywords locally for this tab to ensure specific whitelist is used
        def extract_top_photo_keywords(reviews, keywords, sentiment, exclude_keywords, max_keywords=5):
            pattern = '|'.join(keywords)
            relevant_reviews = reviews[
                reviews['Review'].str.contains(pattern, case=False, regex=True, na=False) &
                (reviews['label'] == sentiment) &
                ~reviews['Review'].str.contains('|'.join(exclude_keywords), case=False, regex=True, na=False)
            ]
            all_words = []
            whitelist = photo_positive_whitelist if sentiment == 'positive' else photo_negative_whitelist
            for review_text in relevant_reviews['Review']:
                words = preprocess_text_for_keywords(review_text)
                all_words.extend([
                    word for word in words
                    if word in whitelist and
                    word not in {'photo', 'photos', 'picture', 'pictures', 'watermark', 'snapshot', 'image', 'swing', 'vr', 'ride', 'virtual', 'reality', 'over', 'edge'} 
                ])
            word_counts = Counter(all_words)
            return word_counts.most_common(max_keywords)

        photo_positive_keywords = extract_top_photo_keywords(
            filtered_data_for_keywords, photo_keywords_detail, 'positive', exclude_keywords_photo_detail
        )
        photo_negative_keywords = extract_top_photo_keywords(
            filtered_data_for_keywords, photo_keywords_detail, 'negative', exclude_keywords_photo_detail
        )


      
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Positive Feedback**")
            fig_photo_pos = create_keyword_chart(
                photo_positive_keywords,
                "Top Positive Keywords for Photos",
                'positive'
            )
            if fig_photo_pos:
                st.plotly_chart(fig_photo_pos, use_container_width=True)
            else:
                st.info("No positive keywords available for Photos.")

        with col2:
            st.markdown("**Negative Feedback**")
            fig_photo_neg = create_keyword_chart(
                photo_negative_keywords,
                "Top Negative Keywords for Photos",
                'negative'
            )
            if fig_photo_neg:
                st.plotly_chart(fig_photo_neg, use_container_width=True)
            else:
                st.info("No negative keywords available for Photos.")

else:
    st.info("Cannot analyze Swing, VR Ride, or Photo feedback due to missing 'Review' or 'label' columns.")

st.markdown("<br>", unsafe_allow_html=True)

# --- Food & Beverage, Staff, and Weather Feedback Analysis ---
if 'Review' in filtered_data_for_keywords.columns and 'label' in filtered_data_for_keywords.columns:
    # Food & Beverage KPIs calculation
    food_keywords = [r'\bfood\b', r'\bdrink\b', r'\bmeal\b', r'\brestaurant\b', r'\bmadam\b', r'\bdining\b', r'\bburger\b']
    food_reviews = filtered_data_for_keywords[
        filtered_data_for_keywords['Review'].str.contains('|'.join(food_keywords), case=False, na=False, regex=True) &
        filtered_data_for_keywords['label'].isin(['positive', 'negative'])
    ]
    total_food_reviews = len(food_reviews)
    food_positive_count = len(food_reviews[food_reviews['label'] == 'positive'])
    food_negative_count = len(food_reviews[food_reviews['label'] == 'negative'])
    food_positive_pct = (food_positive_count / total_food_reviews * 100) if total_food_reviews > 0 else 0
    food_negative_pct = (food_negative_count / total_food_reviews * 100) if total_food_reviews > 0 else 0

    # Staff KPIs calculation
    staff_keywords = [r'\bstaff\b', r'\bemployee\b', r'\bguide\b', r'\bcrew\b', r'\battendant\b', r'\bservice\b']
    staff_reviews = filtered_data_for_keywords[
        filtered_data_for_keywords['Review'].str.contains('|'.join(staff_keywords), case=False, na=False, regex=True) &
        filtered_data_for_keywords['label'].isin(['positive', 'negative'])
    ]
    total_staff_reviews = len(staff_reviews)
    staff_positive_count = len(staff_reviews[staff_reviews['label'] == 'positive'])
    staff_negative_count = len(staff_reviews[staff_reviews['label'] == 'negative'])
    staff_positive_pct = (staff_positive_count / total_staff_reviews * 100) if total_staff_reviews > 0 else 0
    staff_negative_pct = (staff_negative_count / total_staff_reviews * 100) if total_staff_reviews > 0 else 0

    # Weather KPIs calculation
    weather_keywords = [r'\bweather\b', r'\brain\b', r'\bwind\b', r'\bsun\b', r'\bsunny\b', r'\bcloudy\b', r'\bcold\b', r'\bhot\b']
    weather_reviews = filtered_data_for_keywords[
        filtered_data_for_keywords['Review'].str.contains('|'.join(weather_keywords), case=False, na=False, regex=True) &
        filtered_data_for_keywords['label'].isin(['positive', 'negative'])
    ]
    total_weather_reviews = len(weather_reviews)
    weather_positive_count = len(weather_reviews[weather_reviews['label'] == 'positive'])
    weather_negative_count = len(weather_reviews[weather_reviews['label'] == 'negative'])
    weather_positive_pct = (weather_positive_count / total_weather_reviews * 100) if total_weather_reviews > 0 else 0
    weather_negative_pct = (weather_negative_count / total_weather_reviews * 100) if total_weather_reviews > 0 else 0

    # Define whitelists for Food&Beverage, Staff, and Weather for more relevant keywords
    food_positive_whitelist = {
        'delicious', 'tasty', 'good', 'great', 'excellent', 'amazing', 'nice', 'fresh',
        'quality', 'worth', 'value', 'enjoyed', 'burger', 'drink', 'coffee', 'cocktail',
        'selection', 'menu', 'variety'
    }
    food_negative_whitelist = {
        'expensive', 'pricey', 'overpriced', 'poor', 'bad', 'disappointing', 'cold', 'stale',
        'small', 'limited', 'slow', 'wait', 'queue', 'bland', 'overcooked', 'undercooked'
    }
    staff_positive_whitelist = {
        'friendly', 'helpful', 'polite', 'kind', 'attentive', 'efficient', 'professional',
        'welcoming', 'great', 'excellent', 'amazing', 'nice', 'knowledgeable'
    }
    staff_negative_whitelist = {
        'rude', 'unhelpful', 'slow', 'disinterested', 'unprofessional', 'bad', 'poor',
        'ignoring', 'impolite', 'unfriendly'
    }
    weather_positive_whitelist = {
        'sunny', 'clear', 'beautiful', 'perfect', 'nice', 'warm', 'good', 'amazing', 'bright'
    }
    weather_negative_whitelist = {
        'rain', 'rainy', 'windy', 'cold', 'cloudy', 'bad', 'poor', 'mist', 'fog', 'grey', 'wet'
    }

    def extract_top_contextual_keywords(reviews_df, target_keywords_list, sentiment, specific_whitelist, max_keywords=5):
        pattern = '|'.join(target_keywords_list)
        relevant_reviews = reviews_df[
            reviews_df['Review'].str.contains(pattern, case=False, regex=True, na=False) &
            (reviews_df['label'] == sentiment)
        ]
        all_words = []
        for review_text in relevant_reviews['Review']:
            words = preprocess_text_for_keywords(review_text) # Re-use existing preprocess function
            all_words.extend([
                word for word in words
                if word in specific_whitelist and
                word not in [kw.replace(r'\b', '') for kw in target_keywords_list] # Exclude the main topic words
            ])
        word_counts = Counter(all_words)
        return word_counts.most_common(max_keywords)


    st.subheader("▸ Other Key Feedback Areas: Food & Beverage, Staff, and Weather")
    st.markdown("<br>", unsafe_allow_html=True) # Add some space before the text summary

    # Food & Beverage
    st.markdown("##### Food & Beverage")
    st.markdown(
        """
        <div style="font-size: 1.1em; line-height: 1.6;">
        <ul>
            <li>Among reviews that mention "Food & Beverage", <b><span style='color:#66c2ff'>{food_positive_pct:.1f}%</span></b> are positive, and <b><span style='color:#ff6666'>{food_negative_pct:.1f}%</span></b> are negative.</li>
            <li>
                Positive keywords: <span style='color:#66c2ff;'>{food_pos_keywords}</span>. Negative keywords: <span style='color:#ff6666;'>{food_neg_keywords}</span>.
            </li>
        </ul>
        </div>
        """.format(
            food_positive_pct=food_positive_pct,
            food_negative_pct=food_negative_pct,
            food_pos_keywords=", ".join([word for word, count in extract_top_contextual_keywords(filtered_data_for_keywords, food_keywords, 'positive', food_positive_whitelist)]),
            food_neg_keywords=", ".join([word for word, count in extract_top_contextual_keywords(filtered_data_for_keywords, food_keywords, 'negative', food_negative_whitelist)])
        ),
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Staff
    st.markdown("##### Staff")
    st.markdown(
        """
        <div style="font-size: 1.1em; line-height: 1.6;">
        <ul>
            <li>Among reviews that mention "Staff", <b><span style='color:#66c2ff'>{staff_positive_pct:.1f}%</span></b> are positive, and <b><span style='color:#ff6666'>{staff_negative_pct:.1f}%</span></b> are negative.</li>
            <li>
                Positive keywords: <span style='color:#66c2ff;'>{staff_pos_keywords}</span>. Negative keywords: <span style='color:#ff6666;'>{staff_neg_keywords}</span>.
            </li>
        </ul>
        </div>
        """.format(
            staff_positive_pct=staff_positive_pct,
            staff_negative_pct=staff_negative_pct,
            staff_pos_keywords=", ".join([word for word, count in extract_top_contextual_keywords(filtered_data_for_keywords, staff_keywords, 'positive', staff_positive_whitelist)]),
            staff_neg_keywords=", ".join([word for word, count in extract_top_contextual_keywords(filtered_data_for_keywords, staff_keywords, 'negative', staff_negative_whitelist)])
        ),
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Weather
    st.markdown("##### Weather")
    st.markdown(
        """
        <div style="font-size: 1.1em; line-height: 1.6;">
        <ul>
            <li>Among reviews that mention "Weather", <b><span style='color:#66c2ff'>{weather_positive_pct:.1f}%</span></b> are positive, and <b><span style='color:#ff6666'>{weather_negative_pct:.1f}%</span></b> are negative.</li>
            <li>
                Positive keywords: <span style='color:#66c2ff;'>{weather_pos_keywords}</span>. Negative keywords: <span style='color:#ff6666;'>{weather_neg_keywords}</span>.
            </li>
        </ul>
        </div>
        """.format(
            weather_positive_pct=weather_positive_pct,
            weather_negative_pct=weather_negative_pct,
            weather_pos_keywords=", ".join([word for word, count in extract_top_contextual_keywords(filtered_data_for_keywords, weather_keywords, 'positive', weather_positive_whitelist)]),
            weather_neg_keywords=", ".join([word for word, count in extract_top_contextual_keywords(filtered_data_for_keywords, weather_keywords, 'negative', weather_negative_whitelist)])
        ),
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

else:
    st.info("Cannot analyze Food & Beverage, Staff, or Weather feedback due to missing 'Review' or 'label' columns.")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---") # Retain this separator if desired between Add-ons and Other Feedback


