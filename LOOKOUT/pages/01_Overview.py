# pages/01_Overview.py

import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
import os
from PIL import Image # Import Image from PIL
from pathlib import Path # Import Path for robust path handling

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

# --- Page Configuration ---
# Define the base directory for this script
BASE_DIR = Path(__file__).resolve().parent

# Path to lookout_logo_01.png (assuming it's in the parent directory of 'pages')
LOOKOUT_LOGO_PATH = BASE_DIR.parent / "lookout_logo_01.png"

page_icon_overview = "🌆" # Default emoji icon for this page
try:
    img_icon_overview = Image.open(LOOKOUT_LOGO_PATH)
    page_icon_overview = img_icon_overview
except Exception as e:
    st.warning(f"Cannot use '{LOOKOUT_LOGO_PATH.name}' as page icon for Overview: {e}. Using default emoji icon.")

st.set_page_config(page_title="Dashboard Overview - A'DAM LOOKOUT", page_icon=page_icon_overview, layout="wide") # Apply the custom page icon

# --- Title and Logo ---
col_title, col_spacer, col_logo = st.columns([0.65, 0.05, 0.3])
with col_title:
    st.title("Overview")
with col_logo:
    try:
        # Original logo path for the larger image within the page content
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

all_data = st.session_state.processed_data

# --- Sidebar for Date Range Filter ---
st.sidebar.header("Dashboard Filters")
filtered_data = all_data

if 'Time' in all_data.columns and pd.api.types.is_datetime64_any_dtype(all_data['Time']) and not all_data['Time'].isna().all():
    earliest_year = all_data['Time'].min().year if not all_data['Time'].empty else 2020
    min_date_data = pd.to_datetime(f"{earliest_year}-01-01").date()
    max_date_data = all_data['Time'].max().date()
    default_start = min_date_data
    default_end = max_date_data
    if default_start > default_end:
        default_start = default_end
    start_date = st.sidebar.date_input(
        "Start date",
        default_start,
        min_value=min_date_data,
        max_value=max_date_data,
        key="overview_start_date"
    )
    end_date = st.sidebar.date_input(
        "End date",
        default_end,
        min_value=min_date_data,
        max_value=max_date_data,
        key="overview_end_date"
    )
    if start_date > end_date:
        st.sidebar.error("Error: Start date cannot be after end date.")
        filtered_data = pd.DataFrame(columns=all_data.columns)
    else:
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        filtered_data = all_data[
            (all_data['Time'] >= start_datetime) &
            (all_data['Time'] <= end_datetime)
        ]

# --- Main Page Content ---
st.markdown("<br>", unsafe_allow_html=True)
if filtered_data.empty:
    st.warning(f"No review data available for the selected period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")
else:
    st.markdown(f"From **{start_date.strftime('%Y-%m-%d')}** to **{end_date.strftime('%Y-%m-%d')}**")

    # --- Key Performance Indicators (KPIs) ---
    total_reviews = len(filtered_data)
    avg_rating = filtered_data['Rating'].mean() if 'Rating' in filtered_data.columns else None
    avg_sentiment = filtered_data['compound'].mean() if 'compound' in filtered_data.columns else None
    total_languages = filtered_data['Language'].nunique() if 'Language' in filtered_data.columns and not filtered_data['Language'].empty else None

    kpi_total_reviews = f"{total_reviews}"
    kpi_avg_rating = f"{avg_rating:.2f} ⭐" if isinstance(avg_rating, float) else "N/A"
    kpi_avg_sentiment = f"{avg_sentiment:.2f}" if isinstance(avg_sentiment, float) else "N/A"
    kpi_total_languages = f"{total_languages}" if isinstance(total_languages, int) else "N/A"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(create_styled_metric("Total Reviews", kpi_total_reviews), unsafe_allow_html=True)
    with col2:
        st.markdown(create_styled_metric("Average Rating", kpi_avg_rating), unsafe_allow_html=True)
    with col3:
        st.markdown(create_styled_metric("Average Sentiment Score", kpi_avg_sentiment), unsafe_allow_html=True)
    with col4:
        st.markdown(create_styled_metric("Total Languages Used", kpi_total_languages), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Determine Resampling Frequency ---
    resample_freq = 'ME'
    freq_label = "Monthly"
    date_range_days = (end_date - start_date).days
    if date_range_days <= 35:
        resample_freq = 'D'
        freq_label = "Daily"
    elif date_range_days <= 180:
        resample_freq = 'W-Mon'
        freq_label = "Weekly"

    # --- Exclude incomplete last month from time-series charts ---
    if resample_freq == 'ME' and not filtered_data.empty:
        last_data_date = filtered_data['Time'].max()
        # Check if the last data point's date is NOT the last day of its month
        if last_data_date.date() != (last_data_date + pd.offsets.MonthEnd(0)).date():
            # If it's not, filter out all data from that incomplete month
            start_of_last_month = last_data_date.to_period('M').to_timestamp()
            filtered_data = filtered_data[filtered_data['Time'] < start_of_last_month]

    time_series_data = None
    if 'Time' in filtered_data.columns and pd.api.types.is_datetime64_any_dtype(filtered_data['Time']):
        time_series_data = filtered_data.set_index('Time')

    # --- Rating Trends and Distribution ---
    st.subheader("▸ Rating Trends and Distribution")
    if time_series_data is not None and not time_series_data.empty:
        # Average Rating Trend
        if 'Rating' in time_series_data.columns:
            avg_rating_trend = time_series_data['Rating'].resample(resample_freq).mean().dropna().reset_index()
            if not avg_rating_trend.empty:
                fig_rating_trend = px.line(avg_rating_trend, x='Time', y='Rating',
                                           title=f'{freq_label} Average Rating Trend',
                                           labels={'Rating': 'Average Rating (Stars)', 'Time': 'Date'})
                fig_rating_trend.update_traces(mode='lines+markers', line_color='#5a5a5a')
                fig_rating_trend.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='#f8f9fa',
                    yaxis=dict(
                        range=[3.0, 5.0],
                        autorange=False,
                        tickmode='array',
                        tickvals=[3.0, 3.5, 4.0, 4.5, 5.0],
                        ticktext=['3.0', '3.5', '4.0', '4.5', '5.0']
                    )
                )
                st.plotly_chart(fig_rating_trend, use_container_width=True)
            else:
                st.write(f"No data to display for {freq_label} average rating trend.")
        else:
            st.write("Rating column not available for trend analysis.")

        # Review Counts by Star Ratings
        if 'Rating' in filtered_data.columns:
            rating_counts = filtered_data['Rating'].astype(str).value_counts().sort_index().reset_index()
            rating_counts.columns = ['Rating', 'Count']
            rating_color_map = {
                '1': '#f0e6e6',
                '2': '#e0cccc',
                '3': '#d0b3b3',
                '4': '#c09999',
                '5': '#a85454'
            }
            fig_rating_dist = px.bar(rating_counts, x='Rating', y='Count',
                                     title="Review Counts by Star Rating",
                                     labels={'Rating': 'Star Rating', 'Count': 'Number of Reviews'},
                                     text_auto=True,
                                     color='Rating',
                                     color_discrete_map=rating_color_map)
            fig_rating_dist.update_xaxes(type='category')
            fig_rating_dist.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='#f8f9fa')
            st.plotly_chart(fig_rating_dist, use_container_width=True)
        else:
            st.write("Rating column not available for distribution analysis.")
    else:
        st.write("'Time' column not available or in unsuitable format.")

    # --- Sentiment Trends and Distribution ---
    st.markdown("---")
    st.subheader("▸ Sentiment Trends and Distribution")
    st.markdown("<br>", unsafe_allow_html=True)

    if time_series_data is not None and not time_series_data.empty:
        # Sentiment Categories
        if 'label' in filtered_data.columns and total_reviews > 0:
            st.markdown("###### Overall Sentiment Categories")
            st.markdown("<br>", unsafe_allow_html=True)
            sentiment_counts = filtered_data['label'].value_counts()
            positive_count = sentiment_counts.get('positive', 0)
            negative_count = sentiment_counts.get('negative', 0)
            neutral_count = sentiment_counts.get('neutral', 0)
            positive_percentage = (positive_count / total_reviews) * 100
            negative_percentage = (negative_count / total_reviews) * 100
            neutral_percentage = (neutral_count / total_reviews) * 100

            col_pct1, col_pct2, col_pct3 = st.columns(3)
            with col_pct1:
                # "Awesome!" KPI: Percentage color changed to green
                st.markdown(create_styled_metric("Awesome!", f"<span style='color:#66c2ff'>{positive_percentage:.1f}%</span><br><span style='font-size:0.75em; color:#ffffff'>({positive_count} reviews)</span>"), unsafe_allow_html=True)
            with col_pct2:
                st.markdown(create_styled_metric("It's ok.", f"<span style='color:#ffffff'>{neutral_percentage:.1f}%</span><br><span style='font-size:0.75em; color:#ffffff'>({neutral_count} reviews)</span>"), unsafe_allow_html=True)
            with col_pct3:
                # "Not good!" KPI: Percentage color changed to red
                st.markdown(create_styled_metric("Not good!", f"<span style='color:#ff6666'>{negative_percentage:.1f}%</span><br><span style='font-size:0.75em; color:#ffffff'>({negative_count} reviews)</span>"), unsafe_allow_html=True)

        # Average Sentiment Trend
        if 'compound' in time_series_data.columns:
            avg_sentiment_trend = time_series_data['compound'].resample(resample_freq).mean().dropna().reset_index()
            if not avg_sentiment_trend.empty:
                fig_sentiment_trend = px.line(avg_sentiment_trend, x='Time', y='compound',
                                        title=f'{freq_label} Average Sentiment Trend',
                                        labels={'compound': 'Average Sentiment Score', 'Time': 'Date'})
                fig_sentiment_trend.update_traces(mode='lines+markers', line_color='#5a5a5a')
                fig_sentiment_trend.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='#f8f9fa',
                    yaxis=dict(
                        range=[0.0, 1.0],
                        autorange=False,
                        tickmode='array',
                        tickvals=[0.0, 0.5, 1.0],
                        ticktext=['0.0', '0.5', '1.0'],
                        title='Average Sentiment Score'
                    )
                )
                st.plotly_chart(fig_sentiment_trend, use_container_width=True)
            else:
                st.write(f"No data to display for {freq_label} average sentiment trend.")
        else:
            st.write("Sentiment 'compound' column not available for trend analysis.")

        # Heatmap for Review Counts by Sentiment
        if 'label' in time_series_data.columns:
            heatmap_data = time_series_data.groupby([pd.Grouper(freq=resample_freq), 'label']).size().unstack(fill_value=0)
            sentiment_order = [s for s in ['negative', 'neutral', 'positive'] if s in heatmap_data.columns]
            heatmap_data = heatmap_data.reindex(columns=sentiment_order)
            if not heatmap_data.empty:
                heatmap_data.index = heatmap_data.index.strftime('%Y-%m-%d')
                # New continuous color scale from light pink to a new, deeper red
                # In pages/01_Overview.py, replace the custom_heatmap_scale list with:

                custom_heatmap_scale = ['#f0e6e6', '#e0cccc', '#d0b3b3', '#c09999', '#a85454']
                fig_heatmap = px.imshow(heatmap_data,
                                        labels=dict(x="Sentiment Category", y=f"{freq_label} Period", color="Review Count"),
                                        title=f"{freq_label} Heatmap of Review Counts by Sentiment",
                                        aspect="auto",
                                        color_continuous_scale=custom_heatmap_scale)
                fig_heatmap.update_xaxes(side="bottom")
                fig_heatmap.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='#f8f9fa')
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.write(f"No data to display as a heatmap for {freq_label} review counts by sentiment.")
        else:
            st.write("Sentiment 'label' column not available.")
    else:
        st.write("'Time' column not available or in unsuitable format.")

    st.markdown("---")