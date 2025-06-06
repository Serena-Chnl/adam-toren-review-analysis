# pages/05_Customer_Profile.py

import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path
import plotly.express as px
import pycountry
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize # This line might trigger download if not handled below
from collections import Counter
import re
import os # Import os

from Home import download_nltk_resources

# --- Page Configuration ---
st.set_page_config(page_title="Customer Profile Analysis", layout="wide")

# Call centralized NLTK resource download
if not download_nltk_resources():
    st.error("Failed to download required NLTK resources. Some features may not work.")


# --- Helper function for styled metrics ---
def create_styled_metric(label, value_str, background_color="#510f30", text_color="white"):
    style = (
        f"background-color: {background_color}; "
        "padding: 1rem; "
        "border-radius: 8px; "
        "text-align: center; "
        "height: 100%; "
        "display: flex; "
        "flex-direction: column; "
        "justify-content: center;"
    )
    html = f"""
    <div style="{style}">
        <div style='font-size: 0.875rem; color: {text_color}; margin-bottom: 0.25rem; line-height: 1.3;'>{label}</div>
        <div style='font-size: 1.75rem; font-weight: 600; color: {text_color}; line-height: 1.3;'>{value_str}</div>
    </div>
    """
    return html

# --- Helper function to map language to primary country ---
def get_language_country(language):
    if not language or pd.isna(language):
        return 'Unknown'
    language_lower = language.lower()
    try:
        lang = pycountry.languages.get(name=language)
        if lang and hasattr(lang, 'alpha_2'):
            # Map language codes to primary country codes (simplified)
            language_to_country = {
                'en': 'GB', 'nl': 'NL', 'de': 'DE', 'fr': 'FR', 'it': 'IT',
                'es': 'ES', 'pt': 'PT', 'pl': 'PL', 'ro': 'RO', 'no': 'NO',
                'da': 'DK', 'tr': 'TR'
            }
            return language_to_country.get(lang.alpha_2, 'Unknown')
    except Exception:
        pass
    return 'Unknown'

# --- Helper function to extract top keywords ---
def get_top_keywords(text_series, n=5):
    if text_series.empty or not text_series.str.strip().any():
        return ['No text']
    
    try:
        # Combine all reviews into a single string
        text = ' '.join(text_series.dropna().astype(str).str.lower())
        
        # Tokenize and clean
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english') + [
            'food', 'service', 'restaurant', 'place', 'madam', 'view', 'amsterdam',
            'table', 'staff', 'waitress', 'great', 'nice', 'good', 'bad'
        ])
        tokens = [word for word in tokens if word.isalnum() and word not in stop_words and len(word) > 3]
        
        # Count frequencies
        word_counts = Counter(tokens)
        
        # Get top n keywords
        top_keywords = [word for word, _ in word_counts.most_common(n)]
        return top_keywords if top_keywords else ['No keywords']
    except Exception as e:
        return [f'Error: {str(e)}']



# --- Logo and Title Section ---
try:
    logo_path = Path(__file__).resolve().parent.parent / "madam_logo_02.png"
    madam_logo_display = Image.open(logo_path)
    col_title, col_spacer, col_logo = st.columns([0.75, 0.05, 0.2])
    with col_title:
        st.title("Customer Profile Analysis")
    with col_logo:
        st.image(madam_logo_display, width=350)
except FileNotFoundError:
    st.error(f"Logo image 'madam_logo_02.png' not found at expected path: {logo_path}.")
    st.title("👤 Customer Profile Analysis - Madam")
except Exception as e:
    st.error(f"An error occurred while loading the logo: {e}")
    st.title("👤 Customer Profile Analysis - Madam")

# --- Retrieve Processed Data from Session State ---
if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
    st.error("Welkom! To begin, please click the '**Home**' page from the sidebar to load the dataset automatically. All pages will be available right after ☺︎")
    st.stop()

all_data = st.session_state.processed_data.copy()

# --- Sidebar Filters ---
st.sidebar.header("Customer Profile Filters")
filtered_data = all_data
start_date = None
end_date = None

if 'Time' in all_data.columns and pd.api.types.is_datetime64_any_dtype(all_data['Time']) and not all_data['Time'].isna().all():
    min_year = all_data['Time'].dt.year.min() if not all_data['Time'].empty else 2020
    min_date_data_time = pd.to_datetime(f"{min_year}-01-01").date()
    max_date_data_time = all_data['Time'].max().date()
    default_start_time = min_date_data_time
    default_end_time = max_date_data_time
    if default_start_time > default_end_time:
        default_start_time = default_end_time
    start_date = st.sidebar.date_input(
        "Start date (for KPIs)",
        default_start_time,
        min_value=min_date_data_time,
        max_value=max_date_data_time,
        key="customer_profile_start_date"
    )
    end_date = st.sidebar.date_input(
        "End date (for KPIs)",
        default_end_time,
        min_value=min_date_data_time,
        max_value=max_date_data_time,
        key="customer_profile_end_date"
    )
    if start_date and end_date:
        if start_date > end_date:
            st.sidebar.error("Error: Start date cannot be after end date for the KPIs.")
            filtered_data = pd.DataFrame(columns=all_data.columns)
        else:
            start_datetime_time = pd.to_datetime(start_date)
            end_datetime_time = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            filtered_data = all_data[
                (all_data['Time'] >= start_datetime_time) &
                (all_data['Time'] <= end_datetime_time)
            ]
    else:
        filtered_data = all_data.copy()

    st.sidebar.markdown("❕If this page displays the Home page and errors, click '**Customer Profile**'again to fix.")
else:
    st.sidebar.warning("Date filter for KPIs cannot be applied: 'Time' column issue.")
    filtered_data = pd.DataFrame(columns=all_data.columns)

# --- Date Range Display ---
st.markdown("<br>", unsafe_allow_html=True)
if start_date and end_date and start_date <= end_date:
    st.markdown(f"From **{start_date.strftime('%Y-%m-%d')}** to **{end_date.strftime('%Y-%m-%d')}**")
else:
    st.markdown("**Full Dataset**")

# --- KPIs for Language Usage ---
st.markdown("<br>", unsafe_allow_html=True)

kpi_unique_languages = "N/A"
kpi_most_common_language = "N/A"
kpi_positive_language = "N/A"
kpi_negative_language = "N/A"

if not filtered_data.empty and 'Language' in filtered_data.columns:
    # KPI 1: Number of Unique Languages
    unique_languages = filtered_data['Language'].nunique(dropna=True)
    kpi_unique_languages = f"{unique_languages}<br><span style='font-size:0.75em; color:#FFFFFF;'>(languages)</span>"
    
    # KPI 2: Most Common Language Overall
    language_counts = filtered_data['Language'].value_counts()
    if not language_counts.empty:
        most_common_lang = language_counts.index[0]
        count = language_counts.iloc[0]
        kpi_most_common_language = f"{most_common_lang}<br><span style='font-size:0.75em; color:#FFFFFF;'>({count} reviews)</span>"
    else:
        kpi_most_common_language = "No Language Data"
    
    # KPI 3: Most Common Language in Positive Reviews
    if 'label' in filtered_data.columns:
        positive_reviews = filtered_data[filtered_data['label'] == 'positive']
        positive_lang_counts = positive_reviews['Language'].value_counts()
        if not positive_lang_counts.empty:
            most_common_positive_lang = positive_lang_counts.index[0]
            count = positive_lang_counts.iloc[0]
            kpi_positive_language = f"<span style='color:#82ff95'>{most_common_positive_lang}</span><br><span style='font-size:0.75em; color:#FFFFFF;'>({count} reviews)</span>"
        else:
            kpi_positive_language = "No Positive Reviews"
    else:
        kpi_positive_language = "Label Column Missing"
    
    # KPI 4: Most Common Language in Negative Reviews
    if 'label' in filtered_data.columns:
        negative_reviews = filtered_data[filtered_data['label'] == 'negative']
        negative_lang_counts = negative_reviews['Language'].value_counts()
        if not negative_lang_counts.empty:
            most_common_negative_lang = negative_lang_counts.index[0]
            count = negative_lang_counts.iloc[0]
            kpi_negative_language = f"<span style='color:#ff384f'>{most_common_negative_lang}</span><br><span style='font-size:0.75em; color:#FFFFFF;'>({count} reviews)</span>"
        else:
            kpi_negative_language = "No Negative Reviews"
    else:
        kpi_negative_language = "Label Column Missing"
else:
    kpi_unique_languages = "No Data"
    kpi_most_common_language = "No Data"
    kpi_positive_language = "No Data"
    kpi_negative_language = "No Data"

# Display KPIs
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
with kpi_col1:
    st.markdown(create_styled_metric("Unique Languages", kpi_unique_languages), unsafe_allow_html=True)
with kpi_col2:
    st.markdown(create_styled_metric("Top Language", kpi_most_common_language), unsafe_allow_html=True)
with kpi_col3:
    st.markdown(create_styled_metric("Top Positive Language", kpi_positive_language), unsafe_allow_html=True)
with kpi_col4:
    st.markdown(create_styled_metric("Top Negative Language", kpi_negative_language), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Language Distribution Bar Chart ---
st.subheader("▸ Language Distribution")
st.markdown("<br>", unsafe_allow_html=True)

if not filtered_data.empty and 'Language' in filtered_data.columns:
    language_counts = filtered_data['Language'].value_counts().reset_index()
    language_counts.columns = ['Language', 'Count']
    language_counts = language_counts.copy()  # Ensure pandas DataFrame
    total_reviews = language_counts['Count'].sum()
    language_counts['Percentage'] = (language_counts['Count'] / total_reviews * 100).round(2)
    
    # Add country codes for flag emojis
    language_counts['Country_Code'] = language_counts['Language'].apply(get_language_country)
    language_counts['Display_Label'] = language_counts.apply(
        lambda row: f"{row['Language']} ({row['Percentage']}%)",
        axis=1
    )
    
    if not language_counts.empty:
        # Use rating_color_map for consistent color tone
        rating_colors = ['#e6e1e4', '#c7adc1', '#9e6791', '#823871', '#610c4d']
        
        fig_bar = px.bar(
            language_counts,
            y='Display_Label',
            x='Count',
            orientation='h',
            title="Languages Used in Reviews",
            text=language_counts['Percentage'].apply(lambda x: f'{x}%'),
            color='Count',
            color_continuous_scale=rating_colors
        )
        fig_bar.update_traces(
            textposition='auto',
            hovertemplate='%{y}<br>Count: %{x}<br>Percentage: %{text}',
            marker=dict(line=dict(color='#e8dfce', width=1))  # Neutral outline
        )
        fig_bar.update_layout(
            xaxis_title="Number of Reviews",
            yaxis_title=None,
            margin=dict(t=50, l=25, r=25, b=25),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            yaxis=dict(autorange="reversed"),  # Top-to-bottom order
            font=dict(color='#510f30')
        )
        fig_bar.update_xaxes(gridcolor='#f2f4f7')
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No language data available for visualization.")
else:
    st.info("No language data available for visualization.")

st.divider()

# --- Sentiment by Top Languages ---
st.subheader("▸ Sentiment by Top Languages")
st.markdown("<br>", unsafe_allow_html=True)

if not filtered_data.empty and 'Language' in filtered_data.columns and 'label' in filtered_data.columns:
    # Get top 6 languages by review count
    top_languages = filtered_data['Language'].value_counts().head(6).index.tolist()
    
    if top_languages:
        # Filter data for top 6 languages
        top_lang_data = filtered_data[filtered_data['Language'].isin(top_languages)]
        
        # Calculate sentiment percentages
        sentiment_pivot = pd.pivot_table(
            top_lang_data,
            values='Name',  # Any column to count occurrences
            index='Language',
            columns='label',
            aggfunc='count',
            fill_value=0
        )
        
        # Ensure all sentiment categories are present
        for label in ['positive', 'neutral', 'negative']:
            if label not in sentiment_pivot.columns:
                sentiment_pivot[label] = 0
        
        # Calculate percentages
        sentiment_pivot['Total'] = sentiment_pivot.sum(axis=1)
        sentiment_pivot['Positive (%)'] = (sentiment_pivot['positive'] / sentiment_pivot['Total'] * 100).round(2)
        sentiment_pivot['Neutral (%)'] = (sentiment_pivot['neutral'] / sentiment_pivot['Total'] * 100).round(2)
        sentiment_pivot['Negative (%)'] = (sentiment_pivot['negative'] / sentiment_pivot['Total'] * 100).round(2)
        
        # Reset index and sort by Positive (%) desc, Negative (%) asc
        sentiment_data = sentiment_pivot[['Positive (%)', 'Neutral (%)', 'Negative (%)']].reset_index()
        sentiment_data = sentiment_data.sort_values(
            by=['Positive (%)', 'Negative (%)'], ascending=[False, True]
        )
        
        # Identify happiest and grumpiest languages
        happiest_lang = sentiment_data.loc[sentiment_data['Positive (%)'].idxmax(), 'Language']
        grumpiest_lang = sentiment_data.loc[sentiment_data['Negative (%)'].idxmax(), 'Language']
        
        # Display KPIs for happiest and grumpiest
        kpi_sentiment_col1, kpi_sentiment_col2 = st.columns(2)
        with kpi_sentiment_col1:
            st.markdown(create_styled_metric("Happiest Customer", happiest_lang, background_color="#510f30", text_color="#82ff95"), unsafe_allow_html=True)
        with kpi_sentiment_col2:
            st.markdown(create_styled_metric("Grumpiest Customer", grumpiest_lang, background_color="#510f30", text_color="#ff384f"), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Melt data for stacked bar chart
        sentiment_melted = sentiment_data.melt(
            id_vars='Language',
            value_vars=['Positive (%)', 'Neutral (%)', 'Negative (%)'],
            var_name='Sentiment',
            value_name='Percentage'
        )
        
        # Use language as display label (no flags)
        sentiment_melted['Display_Label'] = sentiment_melted['Language']
        
        # Create stacked bar chart
        fig_sentiment = px.bar(
            sentiment_melted,
            x='Percentage',
            y='Display_Label',
            color='Sentiment',
            orientation='h',
            title="Sentiment Distribution for Top 6 Languages",
            text=sentiment_melted['Percentage'].apply(lambda x: f'{x}%'),
            color_discrete_map={
                'Positive (%)': '#7e8e65',  # Positive: Green
                'Neutral (%)': '#e8dfce',   # Neutral: Light beige
                'Negative (%)': '#b65149'   # Negative: Red
            }
        )
        fig_sentiment.update_traces(
            textposition='inside',
            hovertemplate='%{y}<br>%{data.color}: %{x}%',
            marker=dict(line=dict(color='#e8dfce', width=1))
        )
        fig_sentiment.update_layout(
            xaxis_title="Percentage of Reviews",
            yaxis_title=None,
            margin=dict(t=50, l=25, r=25, b=25),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            barmode='stack',
            yaxis=dict(
                autorange="reversed",
                categoryorder='array',
                categoryarray=sentiment_data['Language'].tolist()  # Preserve happiest-to-grumpiest order
            ),
            font=dict(color='#510f30'),
            legend_title_text='Sentiment',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        fig_sentiment.update_xaxes(gridcolor='#f2f4f7', range=[0, 100])
        st.plotly_chart(fig_sentiment, use_container_width=True)
    else:
        st.info("Not enough language data to analyze sentiment for top 6 languages.")
else:
    st.info("No language or sentiment data available for visualization. Ensure 'Language' and 'label' columns exist.")

st.divider()

# --- Top Keywords by Language ---
st.subheader("▸ Average Rating, Top Keywords and Rating Distribution by Top Language")
st.markdown("<br>", unsafe_allow_html=True)

# Add filter for number of keywords
num_keywords = st.slider("Select number of top keywords to display", min_value=1, max_value=10, value=5, key="num_keywords")

if not filtered_data.empty and 'Language' in filtered_data.columns and 'Review' in filtered_data.columns and 'Rating' in filtered_data.columns:
    # Get top 6 languages by review count
    top_languages = filtered_data['Language'].value_counts().head(6).index.tolist()
    
    if top_languages:
        # Define unique gradient backgrounds for each card
        gradients = [
            'linear-gradient(135deg, #faefcf, #f7dcc7)',  # Intermediate Beige to Intermediate Peach
            'linear-gradient(135deg, #f7dcc7, #eadfe0)',  # Intermediate Peach to Intermediate Light Pink
            'linear-gradient(135deg, #eadfe0, #faefcf)',  # Intermediate Light Pink to Intermediate Beige
            'linear-gradient(135deg, #faefcf, #eadfe0)',  # Intermediate Beige to Intermediate Light Pink
            'linear-gradient(135deg, #f7dcc7, #faefcf)',  # Intermediate Peach to Intermediate Beige
            'linear-gradient(135deg, #faefcf, #eadfe0)'   # Intermediate Beige to Intermediate Light Pink
        ]
        
        # Define color map for rating distribution
        rating_color_map = {
            '1': '#e6e1e4',
            '2': '#c7adc1',
            '3': '#9e6791',
            '4': '#823871',
            '5': '#610c4d'
        }
        
        # Create first row of 3 cards
        cols = st.columns(3)
        for i, lang in enumerate(top_languages[:3]):
            lang_reviews = filtered_data[filtered_data['Language'] == lang]['Review']
            top_keywords = get_top_keywords(lang_reviews, n=num_keywords)
            
            # Bold keywords
            bold_keywords = [f"<b>{word}</b>" for word in top_keywords]
            
            # Calculate average rating
            avg_rating = filtered_data[filtered_data['Language'] == lang]['Rating'].mean()
            avg_rating_str = f"<b>{avg_rating:.2f}</b> ⭐️" if not pd.isna(avg_rating) else "N/A ⭐️"
            
            # Create card with HTML
            card_html = f"""
            <div style='background:{gradients[i]}; border-radius:8px; padding:15px; text-align:left; height:230px;'>
                <h3 style='color:#510f30; margin:0; font-weight:bold; font-size:1.8em;'>{lang}</h3>
                <p style='color:#510f30; font-size:1.2em; margin:5px 0;'>Average Rating: {avg_rating_str}</p>
                <p style='color:#510f30; font-size:1.2em; margin:5px 0; line-height:1.4;'>{', '.join(bold_keywords)}</p>
            </div>
            """
            
            # Create rating distribution bar chart
            rating_counts = filtered_data[filtered_data['Language'] == lang]['Rating'].astype(str).value_counts().sort_index().reset_index()
            rating_counts.columns = ['Rating', 'Count']
            # Update x-axis labels to "1-star", "2-star", etc.
            rating_counts['Rating'] = rating_counts['Rating'].apply(lambda x: f"{x}-star")
            
            fig_rating_dist = px.bar(
                rating_counts,
                x='Rating',
                y='Count',
                text_auto=True,
                color='Rating',
                color_discrete_map={f"{k}-star": v for k, v in rating_color_map.items()},
                height=150
            )
            fig_rating_dist.update_xaxes(type='category', title=None)
            fig_rating_dist.update_yaxes(title=None)
            fig_rating_dist.update_layout(
                margin=dict(t=10, l=30, r=10, b=10),
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=10)
            )
            
            # Display card and chart
            with cols[i]:
                st.markdown(card_html, unsafe_allow_html=True)
                st.plotly_chart(fig_rating_dist, use_container_width=True)
        
        # Add vertical spacing
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        
        # Create second row of 3 cards
        cols = st.columns(3)
        for i, lang in enumerate(top_languages[3:]):
            lang_reviews = filtered_data[filtered_data['Language'] == lang]['Review']
            top_keywords = get_top_keywords(lang_reviews, n=num_keywords)
            
            # Bold keywords
            bold_keywords = [f"<b>{word}</b>" for word in top_keywords]
            
            # Calculate average rating
            avg_rating = filtered_data[filtered_data['Language'] == lang]['Rating'].mean()
            avg_rating_str = f"<b>{avg_rating:.2f}</b> ⭐️" if not pd.isna(avg_rating) else "N/A ⭐️"
            
            # Create card with HTML
            card_html = f"""
            <div style='background:{gradients[i+3]}; border-radius:8px; padding:15px; text-align:left; height:230px;'>
                <h3 style='color:#510f30; margin:0; font-weight:bold; font-size:1.8em;'>{lang}</h3>
                <p style='color:#510f30; font-size:1.2em; margin:5px 0;'>Average Rating: {avg_rating_str}</p>
                <p style='color:#510f30; font-size:1.2em; margin:5px 0; line-height:1.4;'>{', '.join(bold_keywords)}</p>
            </div>
            """
            
            # Create rating distribution bar chart
            rating_counts = filtered_data[filtered_data['Language'] == lang]['Rating'].astype(str).value_counts().sort_index().reset_index()
            rating_counts.columns = ['Rating', 'Count']
            # Update x-axis labels to "1-star", "2-star", etc.
            rating_counts['Rating'] = rating_counts['Rating'].apply(lambda x: f"{x}-star")
            
            fig_rating_dist = px.bar(
                rating_counts,
                x='Rating',
                y='Count',
                text_auto=True,
                color='Rating',
                color_discrete_map={f"{k}-star": v for k, v in rating_color_map.items()},
                height=150
            )
            fig_rating_dist.update_xaxes(type='category', title=None)
            fig_rating_dist.update_yaxes(title=None)
            fig_rating_dist.update_layout(
                margin=dict(t=10, l=30, r=10, b=10),
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=10)
            )
            
            # Display card and chart
            with cols[i]:
                st.markdown(card_html, unsafe_allow_html=True)
                st.plotly_chart(fig_rating_dist, use_container_width=True)
    else:
        st.info("Not enough language data to extract keywords for top 6 languages.")
else:
    st.info("No language, review, or rating data available for keyword analysis. Ensure 'Language', 'Review', and 'Rating' columns exist.")

st.markdown("---")