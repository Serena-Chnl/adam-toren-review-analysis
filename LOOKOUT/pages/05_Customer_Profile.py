# pages/05_Customer_Profile.py

import streamlit as st
import pandas as pd
import plotly.express as px
import pycountry
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import os
from PIL import Image # Import Image
from pathlib import Path # Import Path

# --- Page Configuration ---
# Define the base directory for this script
BASE_DIR = Path(__file__).resolve().parent

# Path to lookout_logo_01.png (assuming it's in the parent directory of 'pages')
LOOKOUT_LOGO_PATH = BASE_DIR.parent / "lookout_logo_01.png"

page_icon_customer_profile = "👤" # Default emoji icon for this page
try:
    img_icon_customer_profile = Image.open(LOOKOUT_LOGO_PATH)
    page_icon_customer_profile = img_icon_customer_profile
except Exception as e:
    st.warning(f"Cannot use '{LOOKOUT_LOGO_PATH.name}' as page icon for Customer Profile: {e}. Using default emoji icon.")

st.set_page_config(page_title="Customer Profile Analysis - A'DAM LOOKOUT", page_icon=page_icon_customer_profile, layout="wide")

# --- NLTK Resource Download ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

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
        "justify-content: center;"
    )
    html = f"""
<div style="{style}">
    <div style="font-size: 0.875rem; color: {text_color}; margin-bottom: 0.25rem; line-height: 1.3;">{label}</div>
    <div style="font-size: 1.75rem; font-weight: 600; color: {text_color}; line-height: 1.3;">{value_str}</div>
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
            language_to_country = {
                'en': 'GB', 'nl': 'NL', 'de': 'DE', 'fr': 'FR', 'it': 'IT',
                'es': 'ES', 'pt': 'PT', 'pl': 'PL', 'ro': 'RO', 'no': 'NO',
                'da': 'DK', 'tr': 'TR'
            }
            return language_to_country.get(lang.alpha_2, 'Unknown')
    except Exception:
        return 'Unknown'

# --- Helper function to extract top keywords ---
def get_top_keywords(text_series, n=5):
    if text_series.empty or not text_series.str.strip().any():
        return ['No text']
    try:
        text = ' '.join(text_series.dropna().astype(str).str.lower())
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english') + [
            'lookout', 'adam', 'amsterdam', 'view', 'views', 'swing', 'vr', 'ride',
            'photo', 'photos', 'ticket', 'place', 'experience', 'staff'
        ])
        tokens = [word for word in tokens if word.isalnum() and word not in stop_words and len(word) > 3]
        word_counts = Counter(tokens)
        top_keywords = [word for word, _ in word_counts.most_common(n)]
        return top_keywords if top_keywords else ['No keywords']
    except Exception as e:
        return [f'Error: {str(e)}']

# --- Title and Logo Section ---
col_title, col_spacer, col_logo = st.columns([0.65, 0.05, 0.3])
with col_title:
    st.title("Customer Profile Analysis")
with col_logo:
    try:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(current_script_dir, "..", "lookout_logo_02.png")
        st.image(logo_path, width=550)
    except FileNotFoundError:
        st.error(f"Logo image not found. Please check the path: {logo_path}.")
    except Exception as e:
        st.error(f"An error occurred while loading the logo: {e}.")


# --- Retrieve Processed Data ---
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
        "Start date",
        default_start_time,
        min_value=min_date_data_time,
        max_value=max_date_data_time,
        key="customer_profile_start_date"
    )
    end_date = st.sidebar.date_input(
        "End date",
        default_end_time,
        min_value=min_date_data_time,
        max_value=max_date_data_time,
        key="customer_profile_end_date"
    )
    if start_date and end_date:
        if start_date > end_date:
            st.sidebar.error("Error: Start date cannot be after end date.")
            filtered_data = pd.DataFrame(columns=all_data.columns)
        else:
            start_datetime_time = pd.to_datetime(start_date)
            end_datetime_time = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            filtered_data = all_data[
                (all_data['Time'] >= start_datetime_time) &
                (all_data['Time'] <= end_datetime_time)
            ]
else:
    st.sidebar.warning("Date filter cannot be applied: 'Time' column issue.")

# --- Date Range Display ---
st.markdown("<br>", unsafe_allow_html=True)
if start_date and end_date and start_date <= end_date:
    st.markdown(f"From **{start_date.strftime('%Y-%m-%d')}** to **{end_date.strftime('%Y-%m-%d')}**")
else:
    st.markdown("**Full Dataset**")

# --- KPIs for Language Usage ---
kpi_unique_languages = "N/A"
kpi_most_common_language = "N/A"
kpi_positive_language = "N/A"
kpi_negative_language = "N/A"

if not filtered_data.empty and 'Language' in filtered_data.columns:
    unique_languages = filtered_data['Language'].nunique(dropna=True)
    kpi_unique_languages = f"{unique_languages}<br><span style='font-size:0.75em; color:#ffffff;'>(languages)</span>"
    
    language_counts = filtered_data['Language'].value_counts()
    if not language_counts.empty:
        most_common_lang = language_counts.index[0]
        count = language_counts.iloc[0]
        kpi_most_common_language = f"{most_common_lang}<br><span style='font-size:0.75em; color:#ffffff;'>({count} reviews)</span>"
    
    if 'label' in filtered_data.columns:
        positive_reviews = filtered_data[filtered_data['label'] == 'positive']
        positive_lang_counts = positive_reviews['Language'].value_counts()
        if not positive_lang_counts.empty:
            most_common_positive_lang = positive_lang_counts.index[0]
            count = positive_lang_counts.iloc[0]
            kpi_positive_language = f"<span style='color:#66c2ff'>{most_common_positive_lang}</span><br><span style='font-size:0.75em; color:#ffffff;'>({count} reviews)</span>"
        
        negative_reviews = filtered_data[filtered_data['label'] == 'negative']
        negative_lang_counts = negative_reviews['Language'].value_counts()
        if not negative_lang_counts.empty:
            most_common_negative_lang = negative_lang_counts.index[0]
            count = negative_lang_counts.iloc[0]
            kpi_negative_language = f"<span style='color:#ff6666'>{most_common_negative_lang}</span><br><span style='font-size:0.75em; color:#ffffff;'>({count} reviews)</span>"
else:
    kpi_unique_languages = "No Data"
    kpi_most_common_language = "No Data"
    kpi_positive_language = "No Data"
    kpi_negative_language = "No Data"

# --- Display KPIs ---
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
with kpi_col1:
    st.markdown(create_styled_metric("Unique Languages", kpi_unique_languages), unsafe_allow_html=True)
    if 'show_all_languages' not in st.session_state:
        st.session_state.show_all_languages = False
    if st.button("Toggle All Languages", key="toggle_all_languages"):
        st.session_state.show_all_languages = not st.session_state.show_all_languages
    if st.session_state.show_all_languages:
        if not filtered_data.empty and 'Language' in filtered_data.columns:
            all_languages = sorted(filtered_data['Language'].dropna().unique())
            if all_languages:
                st.markdown(f"**All Languages:** {', '.join(all_languages)}")
            else:
                st.info("No languages found in the data.")
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
    language_counts = filtered_data['Language'].value_counts().head(15).reset_index()
    language_counts.columns = ['Language', 'Count']
    total_reviews = language_counts['Count'].sum()
    language_counts['Percentage'] = (language_counts['Count'] / total_reviews * 100).round(2)
    
    language_counts['Country_Code'] = language_counts['Language'].apply(get_language_country)
    language_counts['Display_Label'] = language_counts.apply(
        lambda row: f"{row['Language']} ({row['Percentage']}%)",
        axis=1
    )
    
    if not language_counts.empty:
        fig_bar = px.bar(
            language_counts,
            y='Display_Label',
            x='Count',
            orientation='h',
            title="Top 15 Languages Used in Reviews",
            text=language_counts['Percentage'].apply(lambda x: f'{x}%'),
            color='Count',
            color_continuous_scale=['#f0e6e6', '#e0cccc', '#d0b3b3', '#c09999', '#a85454']
        )
        fig_bar.update_traces(
            textposition='auto',
            hovertemplate='%{y}<br>Count: %{x}<br>Percentage: %{text}',
            marker=dict(line=dict(color='#e8dfce', width=1))
        )
        fig_bar.update_layout(
            xaxis_title="Number of Reviews",
            yaxis_title=None,
            margin=dict(t=50, l=25, r=25, b=25),
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='#f8f9fa',
            showlegend=False,
            yaxis=dict(autorange="reversed"),
            font=dict(color='#5a5a5a')
        )
        fig_bar.update_xaxes(gridcolor='#f2f4f7')
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No language data available for visualization.")
else:
    st.info("No language data available for visualization.")

st.markdown("---")



# --- Sentiment by Top Languages ---

st.subheader("▸ Sentiment by Top Languages")
st.markdown("<br>", unsafe_allow_html=True)

if not filtered_data.empty and 'Language' in filtered_data.columns and 'label' in filtered_data.columns:
    top_languages = filtered_data['Language'].value_counts().head(10).index.tolist()
    
    if top_languages:
        top_lang_data = filtered_data[filtered_data['Language'].isin(top_languages)]
        
        sentiment_pivot = pd.pivot_table(
            top_lang_data,
            values='Review',
            index='Language',
            columns='label',
            aggfunc='count',
            fill_value=0
        )
        
        for label in ['positive', 'neutral', 'negative']:
            if label not in sentiment_pivot.columns:
                sentiment_pivot[label] = 0
        
        sentiment_pivot['Total'] = sentiment_pivot.sum(axis=1)
        sentiment_pivot['Positive (%)'] = (sentiment_pivot['positive'] / sentiment_pivot['Total'] * 100).round(1)
        sentiment_pivot['Neutral (%)'] = (sentiment_pivot['neutral'] / sentiment_pivot['Total'] * 100).round(1)
        sentiment_pivot['Negative (%)'] = (sentiment_pivot['negative'] / sentiment_pivot['Total'] * 100).round(1)
        
        sentiment_data = sentiment_pivot.reset_index()
        sentiment_data = sentiment_data.sort_values(
            by=['Positive (%)', 'Negative (%)'], ascending=[False, True]
        )
        
        # Identify happiest and grumpiest languages
        max_positive_pct = sentiment_data['Positive (%)'].max()
        happiest_langs = sentiment_data[sentiment_data['Positive (%)'] == max_positive_pct]['Language'].tolist()
        max_negative_pct = sentiment_data['Negative (%)'].max()
        grumpiest_langs = sentiment_data[sentiment_data['Negative (%)'] == max_negative_pct]['Language'].tolist()
        
        kpi_sentiment_col1, kpi_sentiment_col2 = st.columns(2)
        with kpi_sentiment_col1:
            if len(happiest_langs) == 1:
                st.markdown(create_styled_metric("Happiest Visitors", happiest_langs[0], background_color="#5a5a5a", text_color="#66c2ff"), unsafe_allow_html=True)
            else:
                st.markdown(create_styled_metric("Happiest Visitors", f"Multiple ({len(happiest_langs)})", background_color="#5a5a5a", text_color="#66c2ff"), unsafe_allow_html=True)
                with st.expander("See all happiest languages"):
                    for lang in happiest_langs:
                        st.markdown(f"- {lang}")
        with kpi_sentiment_col2:
            if len(grumpiest_langs) == 1:
                st.markdown(create_styled_metric("Grumpiest Visitors", grumpiest_langs[0], background_color="#5a5a5a", text_color="#ff6666"), unsafe_allow_html=True)
            else:
                st.markdown(create_styled_metric("Grumpiest Visitors", f"Multiple ({len(grumpiest_langs)})", background_color="#5a5a5a", text_color="#ff6666"), unsafe_allow_html=True)
                with st.expander("See all grumpiest languages"):
                    for lang in grumpiest_langs:
                        st.markdown(f"- {lang}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        sentiment_melted = sentiment_data.melt(
            id_vars=['Language'],
            value_vars=['Positive (%)', 'Neutral (%)', 'Negative (%)'],
            var_name='Sentiment',
            value_name='Percentage'
        )
        
        sentiment_melted['Display_Label'] = sentiment_melted['Language']
        
        fig_sentiment = px.bar(
            sentiment_melted,
            x='Percentage',
            y='Display_Label',
            color='Sentiment',
            orientation='h',
            title="Sentiment Distribution for Top 10 Languages",
            text=sentiment_melted['Percentage'].apply(lambda x: f'{x}%'),
            color_discrete_map={
                'Positive (%)': '#a3bbce',
                'Neutral (%)': '#e8dfce',
                'Negative (%)': '#d8575d'
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
                categoryarray=sentiment_data['Language'].tolist()
            ),
            font=dict(color='#5a5a5a'),
            legend_title_text='Sentiment',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        fig_sentiment.update_xaxes(gridcolor='#f2f4f7', range=[0, 100])
        st.plotly_chart(fig_sentiment, use_container_width=True)
    else:
        st.info("Not enough language data to analyze sentiment for top 10 languages.")
else:
    st.info("No language or sentiment data available for visualization.")

st.markdown("---")

# --- KPIs for Star Rating Percentages by Language ---
st.subheader("▸ Languages with Highest Star Rating Percentages")
st.markdown("<br>", unsafe_allow_html=True)

if not filtered_data.empty and 'Language' in filtered_data.columns and 'Rating' in filtered_data.columns:
    rating_pivot = pd.pivot_table(
        filtered_data,
        values='Review',
        index='Language',
        columns='Rating',
        aggfunc='count',
        fill_value=0
    )
    
    for rating in range(1, 6):
        if rating not in rating_pivot.columns:
            rating_pivot[rating] = 0
    
    rating_pivot['Total'] = rating_pivot.sum(axis=1)
    # Filter out languages with zero reviews to avoid division by zero
    rating_pivot = rating_pivot[rating_pivot['Total'] > 0]
    
    for rating in range(1, 6):
        rating_pivot[f'{rating} Star (%)'] = (rating_pivot[rating] / rating_pivot['Total'] * 100).round(1)
    
    kpi_1_star = "N/A"
    kpi_2_star = "N/A"
    kpi_3_star = "N/A"
    kpi_4_star = "N/A"
    kpi_5_star = "N/A"
    expander_1_star = None
    expander_2_star = None
    expander_3_star = None
    expander_4_star = None
    expander_5_star = None
    
    if not rating_pivot.empty:
        for rating in range(1, 6):
            max_pct = rating_pivot[f'{rating} Star (%)'].max()
            top_langs = rating_pivot[rating_pivot[f'{rating} Star (%)'] == max_pct].index.tolist()
            if top_langs:
                if len(top_langs) == 1:
                    vars()[f'kpi_{rating}_star'] = f"{top_langs[0]}<br><span style='font-size:0.75em; color:#ffffff;'>({max_pct:.1f}%)</span>"
                else:
                    vars()[f'kpi_{rating}_star'] = f"Multiple ({len(top_langs)})<br><span style='font-size:0.75em; color:#ffffff;'>({max_pct:.1f}%)</span>"
                    vars()[f'expander_{rating}_star'] = top_langs
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(create_styled_metric("Highest 1-Star %", kpi_1_star), unsafe_allow_html=True)
        if expander_1_star:
            with st.expander("See all languages"):
                for lang in expander_1_star:
                    st.markdown(f"- {lang}")
    with col2:
        st.markdown(create_styled_metric("Highest 2-Star %", kpi_2_star), unsafe_allow_html=True)
        if expander_2_star:
            with st.expander("See all languages"):
                for lang in expander_2_star:
                    st.markdown(f"- {lang}")
    with col3:
        st.markdown(create_styled_metric("Highest 3-Star %", kpi_3_star), unsafe_allow_html=True)
        if expander_3_star:
            with st.expander("See all languages"):
                for lang in expander_3_star:
                    st.markdown(f"- {lang}")
    with col4:
        st.markdown(create_styled_metric("Highest 4-Star %", kpi_4_star), unsafe_allow_html=True)
        if expander_4_star:
            with st.expander("See all languages"):
                for lang in expander_4_star:
                    st.markdown(f"- {lang}")
    with col5:
        st.markdown(create_styled_metric("Highest 5-Star %", kpi_5_star), unsafe_allow_html=True)
        if expander_5_star:
            with st.expander("See all languages"):
                for lang in expander_5_star:
                    st.markdown(f"- {lang}")
else:
    st.info("No language or rating data available to calculate star rating percentages.")

st.markdown("""
**Note**: Star rating percentage here may be misleading due to the speically small sample sizes (e.g., a single review skewing percentages). Please refer to the Appendix page for detailed language-based rating distributions to verify data reliability.
""")
st.markdown("<br>", unsafe_allow_html=True)

# --- Top Keywords by Language ---
st.subheader("▸ Average Rating, Top Keywords, and Rating Distribution by Top Language")
st.markdown("<br>", unsafe_allow_html=True)

num_keywords = st.slider("Select number of top keywords to display", min_value=1, max_value=10, value=5, key="num_keywords")

if not filtered_data.empty and 'Language' in filtered_data.columns and 'Review' in filtered_data.columns and 'Rating' in filtered_data.columns:
    top_languages = filtered_data['Language'].value_counts().head(9).index.tolist()
    
    if top_languages:
        gradients = [
            'linear-gradient(135deg, #f7f4e9, #e8e8e8)',  # Soft beige to light grey
            'linear-gradient(135deg, #e8e8e8, #f0e6e6)',  # Light grey to pale pink
            'linear-gradient(135deg, #f7f4e9, #f0e1e1)',  # Soft beige to muted red
            'linear-gradient(135deg, #e8e8e8, #f7f4e9)',  # Light grey to soft beige
            'linear-gradient(135deg, #f0e6e6, #e8e8e8)',  # Pale pink to light grey
            'linear-gradient(135deg, #f7f4e9, #f5dcdc)',  # Soft beige to soft red (muted #ea1b28)
            'linear-gradient(135deg, #e8e8e8, #f0e6e6)',  # Light grey to pale pink
            'linear-gradient(135deg, #f7f4e9, #e8e8e8)',  # Soft beige to light grey
            'linear-gradient(135deg, #f0e1e1, #f7f4e9)'   # Muted red to soft beige
        ]
        
        rating_color_map = {
            '1': '#f0e6e6',
            '2': '#e0cccc',
            '3': '#d0b3b3',
            '4': '#c09999',
            '5': '#a85454'
        }
        
        cols = st.columns(3)
        for i, lang in enumerate(top_languages[:3]):
            lang_reviews = filtered_data[filtered_data['Language'] == lang]['Review']
            top_keywords = get_top_keywords(lang_reviews, n=num_keywords)
            bold_keywords = [f"<b>{word}</b>" for word in top_keywords]
            
            avg_rating = filtered_data[filtered_data['Language'] == lang]['Rating'].mean()
            avg_rating_str = f"<b>{avg_rating:.2f}</b> ⭐" if not pd.isna(avg_rating) else "N/A ⭐"
            
            card_html = f"""
<div style='background:{gradients[i]}; border-radius:8px; padding:15px; text-align:left; height:230px;'>
    <h3 style='color:#5a5a5a; margin:0; font-weight:bold; font-size:1.8em;'>{lang}</h3>
    <p style='color:#5a5a5a; font-size:1.2em; margin:5px 0;'>Average Rating: {avg_rating_str}</p>
    <p style='color:#5a5a5a; font-size:1.2em; margin:5px 0; line-height:1.4;'>{', '.join(bold_keywords)}</p>
</div>
"""
            
            rating_counts = filtered_data[filtered_data['Language'] == lang]['Rating'].astype(str).value_counts().sort_index().reset_index()
            rating_counts.columns = ['Rating', 'Count']
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
            
            with cols[i]:
                st.markdown(card_html, unsafe_allow_html=True)
                st.plotly_chart(fig_rating_dist, use_container_width=True)
        
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        
        cols = st.columns(3)
        for i, lang in enumerate(top_languages[3:6]):
            lang_reviews = filtered_data[filtered_data['Language'] == lang]['Review']
            top_keywords = get_top_keywords(lang_reviews, n=num_keywords)
            bold_keywords = [f"<b>{word}</b>" for word in top_keywords]
            
            avg_rating = filtered_data[filtered_data['Language'] == lang]['Rating'].mean()
            avg_rating_str = f"<b>{avg_rating:.2f}</b> ⭐" if not pd.isna(avg_rating) else "N/A ⭐"
            
            card_html = f"""
<div style='background:{gradients[i+3]}; border-radius:8px; padding:15px; text-align:left; height:230px;'>
    <h3 style='color:#5a5a5a; margin:0; font-weight:bold; font-size:1.8em;'>{lang}</h3>
    <p style='color:#5a5a5a; font-size:1.2em; margin:5px 0;'>Average Rating: {avg_rating_str}</p>
    <p style='color:#5a5a5a; font-size:1.2em; margin:5px 0; line-height:1.4;'>{', '.join(bold_keywords)}</p>
</div>
"""
            
            rating_counts = filtered_data[filtered_data['Language'] == lang]['Rating'].astype(str).value_counts().sort_index().reset_index()
            rating_counts.columns = ['Rating', 'Count']
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
            
            with cols[i]:
                st.markdown(card_html, unsafe_allow_html=True)
                st.plotly_chart(fig_rating_dist, use_container_width=True)
        
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        
        cols = st.columns(3)
        for i, lang in enumerate(top_languages[6:9]):
            lang_reviews = filtered_data[filtered_data['Language'] == lang]['Review']
            top_keywords = get_top_keywords(lang_reviews, n=num_keywords)
            bold_keywords = [f"<b>{word}</b>" for word in top_keywords]
            
            avg_rating = filtered_data[filtered_data['Language'] == lang]['Rating'].mean()
            avg_rating_str = f"<b>{avg_rating:.2f}</b> ⭐" if not pd.isna(avg_rating) else "N/A ⭐"
            
            card_html = f"""
<div style='background:{gradients[i+6]}; border-radius:8px; padding:15px; text-align:left; height:230px;'>
    <h3 style='color:#5a5a5a; margin:0; font-weight:bold; font-size:1.8em;'>{lang}</h3>
    <p style='color:#5a5a5a; font-size:1.2em; margin:5px 0;'>Average Rating: {avg_rating_str}</p>
    <p style='color:#5a5a5a; font-size:1.2em; margin:5px 0; line-height:1.4;'>{', '.join(bold_keywords)}</p>
</div>
"""
            
            rating_counts = filtered_data[filtered_data['Language'] == lang]['Rating'].astype(str).value_counts().sort_index().reset_index()
            rating_counts.columns = ['Rating', 'Count']
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
            
            with cols[i]:
                st.markdown(card_html, unsafe_allow_html=True)
                st.plotly_chart(fig_rating_dist, use_container_width=True)
    else:
        st.info("Not enough language data to extract keywords for top 9 languages.")
else:
    st.info("No language, review, or rating data available for keyword analysis.")



st.markdown("---")
