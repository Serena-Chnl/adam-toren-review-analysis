# pages/02_Dashboard_Overview.py

import streamlit as st
import pandas as pd
import plotly.express as px
import datetime # For date operations
import os # Added for robust path construction
from PIL import Image # Ensure PIL is imported
from pathlib import Path # Ensure Pathlib is imported

# --- Define Base Directory for favicon ---
# This path goes up one level from 'pages' directory to find 'madam_logo_01.png'
BASE_DIR = Path(__file__).resolve().parent.parent
LOGO_PATH = BASE_DIR / "madam_logo_01.png"

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
try:
    # Use madam_logo_01.png as the page icon (favicon)
    img_logo_icon = Image.open(LOGO_PATH)
    st.set_page_config(
        page_title="Dashboard Overview - Madam",
        page_icon=img_logo_icon, # 设置 favicon 为 madam_logo_01.png
        layout="wide"
    )
except FileNotFoundError:
    # Fallback if madam_logo_01.png is not found
    st.set_page_config(
        page_title="Dashboard Overview - Madam",
        page_icon="🍽️", # 备用 emoji 图标
        layout="wide"
    )

# --- Helper function for styled metrics ---
def create_styled_metric(label, value_str, background_color="#510f30", label_color="#FFFFFF", value_color="#FFFFFF"):
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
    <div style="font-size: 0.875rem; color: {label_color}; margin-bottom: 0.25rem; line-height: 1.3;">{label}</div>
    <div style="font-size: 1.75rem; font-weight: 600; color: {value_color}; line-height: 1.3;">{value_str}</div>
</div>
"""
    return html

# --- MODIFIED SECTION: Title and Logo ---
# Create columns: one for the title, a small spacer, and one for the logo.
# Adjust the ratios in the list (e.g., [0.75, 0.05, 0.2]) to change the relative widths.
col_title, col_spacer, col_logo = st.columns([0.75, 0.05, 0.2])

with col_title:
    st.title("Overview")

with col_logo:
    # Define the path to your logo using a more robust method.
    # This constructs an absolute path to the logo relative to this script file.
    try:
        # Get the directory of the current script (02_Dashboard_Overview.py)
        # Note: This path is for the LOGO displayed within the page, not the favicon.
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go one level up to the main project directory and then specify the logo name
        logo_path = os.path.join(current_script_dir, "..", "madam_logo_02.png") # Assuming madam_logo_02.png is in the main directory

        # Display the image. Adjust 'width' as needed.
        st.image(logo_path, width=350) # You can change the width
    except FileNotFoundError:
        # This specific error is caught if the path is definitely wrong and file doesn't exist
        st.error(f"Logo image not found. Please check the path: {logo_path}. Ensure 'madam_logo_02.png' is in the main project directory.")
    except Exception as e:
        # This catches other errors, including "Error opening" if the file is found but can't be read/processed
        st.error(f"An error occurred while loading the logo: {e}. Ensure the file is a valid image and the path is correct: {logo_path}")
# --- END MODIFIED SECTION ---

# --- Retrieve Processed Data from Session State ---
if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
    st.error("Welkom! To begin, please click the '**Home**' page from the sidebar to load the dataset automatically. All pages will be available right after ☺︎")
    st.stop()

all_data = st.session_state.processed_data

# --- Sidebar for Date Range Filter ---
st.sidebar.header("Dashboard Filters")
filtered_data_for_overview = all_data

if 'Time' in all_data.columns and pd.api.types.is_datetime64_any_dtype(all_data['Time']) and not all_data['Time'].isna().all():
    # Set min_date_data to January 1st of the earliest year in the dataset
    earliest_year = all_data['Time'].min().year if not all_data['Time'].empty else 2020  # Fallback to 2020 if no data
    min_date_data = pd.to_datetime(f"{earliest_year}-01-01").date()
    max_date_data = all_data['Time'].max().date()

    # Set default_start to January 1st of the earliest year
    default_start = min_date_data
    default_end = max_date_data
    if default_start > default_end:
        default_start = default_end

    start_date_ov = st.sidebar.date_input(
        "Start date",
        default_start,
        min_value=min_date_data,
        max_value=max_date_data,
        key="overview_start_date"
    )
    end_date_ov = st.sidebar.date_input(
        "End date",
        default_end,
        min_value=min_date_data,
        max_value=max_date_data,
        key="overview_end_date"
    )

    # Apply date filter to the data
    filtered_data_for_overview = filtered_data_for_overview[
        (filtered_data_for_overview['Time'].dt.date >= start_date_ov) &
        (filtered_data_for_overview['Time'].dt.date <= end_date_ov)
    ]

# --- Main Page Content ---
st.markdown("<br>", unsafe_allow_html=True)
if filtered_data_for_overview is None or filtered_data_for_overview.empty:
    if 'start_date_ov' in locals() and 'end_date_ov' in locals():
        st.warning(f"No review data available for the selected period: {start_date_ov.strftime('%Y-%m-%d')} to {end_date_ov.strftime('%Y-%m-%d')}.")
    else:
        st.warning("No data available to display. This might be due to initial data loading issues or problems with the 'Time' column.")
else:
    if 'start_date_ov' in locals() and 'end_date_ov' in locals():
        st.markdown(f"From **{start_date_ov.strftime('%Y-%m-%d')}** to **{end_date_ov.strftime('%Y-%m-%d')}**")
    else:
        st.subheader("Overall Insights (Full Dataset)")

    # --- Key Performance Indicators (KPIs) ---
    total_reviews_period = len(filtered_data_for_overview)

    avg_rating_period_val = filtered_data_for_overview['Rating'].mean() if 'Rating' in filtered_data_for_overview.columns and not filtered_data_for_overview['Rating'].empty else None
    avg_sentiment_period_val = filtered_data_for_overview['compound'].mean() if 'compound' in filtered_data_for_overview.columns and not filtered_data_for_overview['compound'].empty else None
    total_languages_used_val = filtered_data_for_overview['Language'].nunique() if 'Language' in filtered_data_for_overview.columns and not filtered_data_for_overview['Language'].empty else None

    kpi_total_reviews_str = f"{total_reviews_period}"
    kpi_avg_rating_str = f"{avg_rating_period_val:.2f} ⭐" if isinstance(avg_rating_period_val, float) else "N/A"
    kpi_avg_sentiment_str = f"{avg_sentiment_period_val:.2f}" if isinstance(avg_sentiment_period_val, float) else "N/A"
    kpi_total_languages_str = f"{total_languages_used_val}" if isinstance(total_languages_used_val, int) else "N/A"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(create_styled_metric("Total Reviews", kpi_total_reviews_str), unsafe_allow_html=True)
    with col2:
        st.markdown(create_styled_metric("Average Rating", kpi_avg_rating_str), unsafe_allow_html=True)
    with col3:
        st.markdown(create_styled_metric("Average Sentiment Score", kpi_avg_sentiment_str), unsafe_allow_html=True)
    with col4:
        st.markdown(create_styled_metric("Total Languages Used", kpi_total_languages_str), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Determine Resampling Frequency ---
    resample_freq = 'ME'
    freq_label = "Monthly"
    time_series_data = None
    if 'Time' in filtered_data_for_overview.columns and \
       pd.api.types.is_datetime64_any_dtype(filtered_data_for_overview['Time']) and \
       not filtered_data_for_overview['Time'].isna().all():

        time_series_data = filtered_data_for_overview.set_index('Time')

        date_range_days = 0
        if 'start_date_ov' in locals() and 'end_date_ov' in locals() and \
           isinstance(start_date_ov, datetime.date) and isinstance(end_date_ov, datetime.date):
            date_range_days = (end_date_ov - start_date_ov).days

        if date_range_days <= 1:
            resample_freq = 'D'
            freq_label = "Daily"
        elif date_range_days <= 35:
            resample_freq = 'D'
            freq_label = "Daily"
        elif date_range_days <= 180:
            resample_freq = 'W-Mon'
            freq_label = "Weekly"
        # else, it remains 'ME' (Monthly) as set by default

    # --- Rating Trends and Distribution ---
    # st.markdown("---")
    st.subheader("▸ Rating Trends and Distribution")

    if time_series_data is not None:
        # ----------- 1. Average Rating Trend -----------
        if 'Rating' in time_series_data.columns:
            avg_rating_trend = time_series_data['Rating'].resample(resample_freq).mean().dropna().reset_index()
            if not avg_rating_trend.empty:
                fig_rating_trend = px.line(avg_rating_trend, x='Time', y='Rating',
                                           title=f'{freq_label} Average Rating Trend',
                                           labels={'Rating': 'Average Rating (Stars)', 'Time': 'Date'})
                fig_rating_trend.update_traces(mode='lines+markers', line_color='#610c4d')
                st.plotly_chart(fig_rating_trend, use_container_width=True)
            else:
                st.write(f"No data to display for {freq_label} average rating trend.")
        else:
            st.write("Rating column not available for trend analysis.")

        # --------- 2. Review Counts by Star Ratings (Distribution) ----------
        if 'Rating' in filtered_data_for_overview.columns:
            rating_counts = filtered_data_for_overview['Rating'].astype(str).value_counts().sort_index().reset_index()
            rating_counts.columns = ['Rating', 'Count']

            rating_color_map = {
                '1': '#e6e1e4',
                '2': '#c7adc1',
                '3': '#9e6791',
                '4': '#823871',
                '5': '#610c4d',
            }

            fig_rating_dist = px.bar(rating_counts, x='Rating', y='Count',
                                     title="Review Counts by Star Rating",
                                     labels={'Rating': 'Star Rating', 'Count': 'Number of Reviews'},
                                     text_auto=True,
                                     color='Rating',
                                     color_discrete_map=rating_color_map
                                     )
            fig_rating_dist.update_xaxes(type='category')
            st.plotly_chart(fig_rating_dist, use_container_width=True)
        else:
            st.write("Rating column not available for distribution analysis.")
    else:
        st.write("'Time' column not available or in unsuitable format. Cannot display rating trends or distribution.")

    # --- Sentiment Trends and Distribution ---
    st.markdown("---")
    st.subheader("▸ Sentiment Trends and Distribution")
    st.markdown("<br>", unsafe_allow_html=True)

    if time_series_data is not None:
        # ------------ 1. Overall Sentiment Categories (Textual Percentages) ------------
        if 'label' in filtered_data_for_overview.columns:
            st.markdown("###### Overall Sentiment Categories (for selected period)")
            st.markdown("<br>", unsafe_allow_html=True)
            if total_reviews_period > 0:
                sentiment_counts = filtered_data_for_overview['label'].value_counts()

                positive_count = sentiment_counts.get('positive', 0)
                negative_count = sentiment_counts.get('negative', 0)
                neutral_count = sentiment_counts.get('neutral', 0)

                positive_percentage = (positive_count / total_reviews_period) * 100
                negative_percentage = (negative_count / total_reviews_period) * 100
                neutral_percentage = (neutral_count / total_reviews_period) * 100

                col_pct1, col_pct2, col_pct3 = st.columns(3)
                with col_pct1:
                    st.markdown(create_styled_metric("Awesome!", f"<span style='color:#82ff95'>{positive_percentage:.1f}%</span><br><span style='font-size:0.75em; color:#FFFFFF'>({positive_count} reviews)</span>", label_color="#FFFFFF", value_color="#FFFFFF"), unsafe_allow_html=True)
                with col_pct2:
                    st.markdown(create_styled_metric("It'ok.", f"<span style='color:#FFFFFF'>{neutral_percentage:.1f}%</span><br><span style='font-size:0.75em; color:#FFFFFF'>({neutral_count} reviews)</span>", label_color="#FFFFFF", value_color="#FFFFFF"), unsafe_allow_html=True)
                with col_pct3:
                    st.markdown(create_styled_metric("Not good!", f"<span style='color:#ff384f'>{negative_percentage:.1f}%</span><br><span style='font-size:0.75em; color:#FFFFFF'>({negative_count} reviews)</span>", label_color="#FFFFFF", value_color="#FFFFFF"), unsafe_allow_html=True)

            elif total_reviews_period == 0:
                st.info("No reviews in the selected period to show sentiment category distribution.")
        else:
            st.write("Sentiment 'label' column not available for category distribution.")

        # ------------ 2. Average Sentiment Trend --------------
        if 'compound' in time_series_data.columns:
            avg_sentiment_trend = time_series_data['compound'].resample(resample_freq).mean().dropna().reset_index()
            if not avg_sentiment_trend.empty:
                fig_sentiment_trend = px.line(avg_sentiment_trend, x='Time', y='compound',
                                              title=f'{freq_label} Average Sentiment Trend',
                                              labels={'compound': 'Average Compound Score', 'Time': 'Date'})
                fig_sentiment_trend.update_traces(mode='lines+markers', line_color='#610c4d')
                st.plotly_chart(fig_sentiment_trend, use_container_width=True)
            else:
                st.write(f"No data to display for {freq_label} average sentiment trend.")
        else:
            st.write("Sentiment 'compound' column not available for trend analysis.")

        # ---------- 3. Heatmap for Review Counts by Sentiment ----------
        if 'label' in time_series_data.columns:
            heatmap_data = time_series_data.groupby([pd.Grouper(freq=resample_freq), 'label']).size().unstack(fill_value=0)

            sentiment_order = [s for s in ['negative', 'neutral', 'positive'] if s in heatmap_data.columns]
            heatmap_data = heatmap_data.reindex(columns=sentiment_order)

            if not heatmap_data.empty:
                heatmap_data.index = heatmap_data.index.strftime('%Y-%m-%d')
                custom_heatmap_scale = ['#f7edf6', '#d4b9d9', '#af85ad', '#8a5081', '#610c4d']

                fig_heatmap = px.imshow(heatmap_data,
                                        labels=dict(x="Sentiment Category", y=f"{freq_label} Period", color="Review Count"),
                                        title=f"{freq_label} Heatmap of Review Counts by Sentiment",
                                        aspect="auto",
                                        color_continuous_scale=custom_heatmap_scale
                                       )
                fig_heatmap.update_xaxes(side="bottom")
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.write(f"No data to display as a heatmap for {freq_label} review counts by sentiment for the selected period.")
        else:
            st.write("Sentiment 'label' column not available for review count heatmap analysis.")
    else:
        st.write("'Time' column not available or in unsuitable format. Cannot display sentiment trends or distribution.")

    st.markdown("---")