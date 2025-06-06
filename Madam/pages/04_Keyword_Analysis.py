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
os.environ["PLOTLY_PANDAS_BACKEND"] = "pandas"

# --- Page Title and Introduction ---
st.set_page_config(page_title="Keyword Analysis - Madam", layout="wide")

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
def create_styled_metric(label, value_str, background_color="#510f30"):
    """
    Creates an HTML string for a styled metric box.
    """
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
    <div style="font-size: 0.875rem; color: #FFFFFF; margin-bottom: 0.25rem; line-height: 1.3;">{label}</div>
    <div style="font-size: 1.75rem; font-weight: 600; color: #FFFFFF; line-height: 1.3;">{value_str}</div>
</div>
"""
    return html

# --- Title and Logo Section ---
col_title, col_spacer, col_logo = st.columns([0.75, 0.05, 0.2])
with col_title:
    st.title("Keyword Analysis")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
with col_logo:
    try:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(current_script_dir, "..", "madam_logo_02.png")
        st.image(logo_path, width=350)
    except FileNotFoundError:
        st.error(f"Logo image 'madam_logo_02.png' not found. Please check the path: {logo_path}.")
    except Exception as e:
        st.error(f"An error occurred while loading the logo: {e}. Ensure the file is a valid image and the path is correct: {logo_path}")

# --- Retrieve Processed Data from Session State ---
if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
    st.error("Welkom! To begin, please click the '**Home**' page from the sidebar to load the dataset automatically. All pages will be available right after ☺︎")
    st.stop()

all_data = st.session_state.processed_data

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
# Modified: Adjusted stopwords for word cloud to highlight dishes
custom_stopwords = {
    'madam', 'restaurant', 'place', 'also', 'get', 'got', 'would', 'could', 'like', 'time', 'experience', 'amsterdam',
    'really', 'one', 'even', 'us', 'went', 'came', 'told', 'asked', 'view', 'staff', 'server', 'waitress', 'waiter',
    'hostess', 'manager', 'served'
}
stop_words_english.update(custom_stopwords)

# Dish whitelist to ensure specific dishes are retained
dish_whitelist = {
    'burger', 'salmon', 'steak', 'fries', 'salad', 'cod', 'pasta', 'ravioli', 'schnitzel', 'cocktail', 'cocktails',
    'wine', 'water', 'coffee', 'dessert', 'cake', 'burrata', 'carpaccio', 'soup', 'tenderloin'
}

def preprocess_text_for_keywords(text):
    if pd.isna(text):
        return []
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words_english and len(word) > 2]
    return lemmatized_words

def preprocess_text_for_wordcloud(text):
    if pd.isna(text):
        return []
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    # Retain dish whitelist words and descriptive adjectives, skip stopwords
    lemmatized_words = [
        lemmatizer.lemmatize(word) for word in words
        if (word in dish_whitelist or word in {'delicious', 'tasty', 'amazing', 'bad', 'poor', 'bland', 'salty'})
    ]
    return lemmatized_words

# --- Main Page Content ---

# --- Calculate KPIs for Most Mentioned Words by Rating ---
if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
    all_data = st.session_state.processed_data
    lemmatizer = WordNetLemmatizer()
    stop_words_english = set(stopwords.words('english'))
    custom_stopwords = {'madam', 'restaurant', 'place', 'also', 'get', 'got', 'would', 'could', 'like', 'good', 'great', 'nice', 'time', 'experience', 'amsterdam', 'really', 'one', 'even', 'us', 'went', 'came', 'told', 'asked'}
    stop_words_english.update(custom_stopwords)

    def preprocess_text_for_kpis(text):
        if pd.isna(text):
            return []
        text = str(text).lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words_english and len(word) > 2]
        return lemmatized_words

    # Define keyword groups for KPIs
    keyword_groups = {
        'service': ['service', 'server', 'waitress', 'waiter', 'staff', 'hostess', 'manager', 'served'],
        'food': ['food', 'meal', 'dish', 'dishes', 'salmon', 'burger', 'steak', 'fries', 'salad', 'cod', 'menu'],
        'view': ['view', 'views', 'scenery'],
        'restaurant': ['restaurant', 'place', 'location'],
        'drinks': ['drinks', 'cocktail', 'cocktails', 'wine', 'water', 'coffee']
    }

    # Initialize KPI values
    kpi_words = {5: "N/A", 4: "N/A", 3: "N/A", 2: "N/A", 1: "N/A"}

    if 'Review' in all_data.columns and 'Rating' in all_data.columns:
        for rating in range(1, 6):
            rating_data = all_data[all_data['Rating'] == rating]
            all_words = []
            for review_text in rating_data['Review']:
                all_words.extend(preprocess_text_for_kpis(review_text))
            
            if all_words:
                word_counts = Counter(all_words)
                group_counts = {key: 0 for key in keyword_groups}
                for word, count in word_counts.items():
                    for group, synonyms in keyword_groups.items():
                        if word in synonyms:
                            group_counts[group] += count
                top_group = max(group_counts, key=group_counts.get, default="N/A")
                if group_counts[top_group] > 0:
                    kpi_words[rating] = top_group.capitalize()
                else:
                    kpi_words[rating] = "N/A"

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
        Sometimes, a word that seems positive (like "view" or a staff name) might appear here.
        This can happen if the word was mentioned in a review that was *overall* negative due to other factors
        (e.g., "The view was great, but the food was terrible.").
        To understand the full context, it's recommended to find these reviews in the "Review Explorer" page.
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
                st.subheader(f"▸ Top Keywords Bubble Chart ({sentiment_filter_kw}) ")
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
                            color_continuous_scale='Viridis',
                            title=f"Top {min(len(df_most_common), num_keywords)} Keywords ({sentiment_filter_kw})",
                            hover_data={'Keyword': True, 'Frequency': True}
                        )
                        fig_bubble.update_layout(
                            xaxis_title="Keyword",
                            yaxis_title="",
                            showlegend=True,
                            xaxis_tickangle=45,
                            margin=dict(t=50, l=25, r=25, b=25)
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
                            background_color='white',
                            colormap='viridis',
                            max_words=50,  # Reduced to focus on high-impact words
                            contour_width=1,
                            contour_color='steelblue'
                        ).generate_from_frequencies(dict(word_counts_wc))
                        fig_wc, ax = plt.subplots(figsize=(12, 6))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig_wc)
                    except Exception as e:
                        st.error(f"Could not generate word cloud: {e}")
                else:
                    st.info("Not enough data to generate a word cloud for the current selection.")

# --- Keyword Comparison: Weekday vs. Weekend ---
st.markdown("---")
st.subheader("▸ Keyword Comparison: Weekday vs. Weekend")

if 'Time' in filtered_data_for_keywords.columns and \
   pd.api.types.is_datetime64_any_dtype(filtered_data_for_keywords['Time']) and \
   'Review' in filtered_data_for_keywords.columns and \
   'label' in filtered_data_for_keywords.columns:
    
    comparison_data = filtered_data_for_keywords.copy()
    comparison_data['DayType'] = comparison_data['Time'].dt.dayofweek.apply(
        lambda x: 'Weekend' if x >= 5 else 'Weekday'
    )
    
    weekday_positive_data = comparison_data[(comparison_data['DayType'] == 'Weekday') & (comparison_data['label'] == 'positive')]
    weekend_positive_data = comparison_data[(comparison_data['DayType'] == 'Weekend') & (comparison_data['label'] == 'positive')]
    weekday_negative_data = comparison_data[(comparison_data['DayType'] == 'Weekday') & (comparison_data['label'] == 'negative')]
    weekend_negative_data = comparison_data[(comparison_data['DayType'] == 'Weekend') & (comparison_data['label'] == 'negative')]

    def get_keyword_df(df):
        if not df.empty:
            words = []
            for review_text in df['Review']:
                words.extend(preprocess_text_for_keywords(review_text))
            
            if words:
                word_counts = Counter(words)
                most_common = word_counts.most_common(15)
                return pd.DataFrame(most_common, columns=['Keyword', 'Frequency'])
        return pd.DataFrame(columns=['Keyword', 'Frequency'])

    col_positive, col_negative = st.columns(2)

    with col_positive:
        st.markdown("<h5 style='text-align: center; font-weight: bold;'>Positive Reviews</h5>", unsafe_allow_html=True)
        
        df_wp = get_keyword_df(weekday_positive_data)
        df_we_p = get_keyword_df(weekend_positive_data)
        
        if df_wp.empty and df_we_p.empty:
            st.info("No positive review data available to compare.")
        else:
            df_positive_combined = pd.concat([df_wp.reset_index(drop=True), df_we_p.reset_index(drop=True)], axis=1)
            df_positive_combined.columns = pd.MultiIndex.from_tuples([
                ('Weekday', 'Keyword'), ('Weekday', 'Frequency'),
                ('Weekend', 'Keyword'), ('Weekend', 'Frequency')
            ])
            df_positive_combined.index = df_positive_combined.index + 1
            st.dataframe(df_positive_combined.fillna(''), use_container_width=True)

    with col_negative:
        st.markdown("<h5 style='text-align: center; font-weight: bold;'>Negative Reviews</h5>", unsafe_allow_html=True)
        
        df_wn = get_keyword_df(weekday_negative_data)
        df_we_n = get_keyword_df(weekend_negative_data)
        
        if df_wn.empty and df_we_n.empty:
            st.info("No negative review data available to compare.")
        else:
            df_negative_combined = pd.concat([df_wn.reset_index(drop=True), df_we_n.reset_index(drop=True)], axis=1)
            df_negative_combined.columns = pd.MultiIndex.from_tuples([
                ('Weekday', 'Keyword'), ('Weekday', 'Frequency'),
                ('Weekend', 'Keyword'), ('Weekend', 'Frequency')
            ])
            df_negative_combined.index = df_negative_combined.index + 1
            st.dataframe(df_negative_combined.fillna(''), use_container_width=True)

else:
    st.warning("Cannot perform comparison. Required columns: 'Time', 'Review', 'label'.")

st.info("""
    **Note on Keywords in Negative Reviews:**
    Sometimes, a word that seems positive (like "view" or a staff name) might appear here.
    This can happen if the word was mentioned in a review that was *overall negative* due to other factors
    (e.g., "The view was great, but the food was terrible.").
    To understand the full context, it's recommended to find these reviews in the "Review Explorer" page.
    """)

st.markdown("---")