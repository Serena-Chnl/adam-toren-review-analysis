# pages/02_Feedback_Trend_Analysis.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import datetime
import calendar
import numpy as np
from PIL import Image # Import Image
from pathlib import Path # Import Path
import os

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

page_icon_feedback_trending = "📈" # Default emoji icon for this page
try:
    img_icon_feedback_trending = Image.open(LOOKOUT_LOGO_PATH)
    page_icon_feedback_trending = img_icon_feedback_trending
except Exception as e:
    st.warning(f"Cannot use '{LOOKOUT_LOGO_PATH.name}' as page icon for Feedback Trending: {e}. Using default emoji icon.")

st.set_page_config(page_title="Feedback Trend Analysis - A'DAM LOOKOUT", page_icon=page_icon_feedback_trending, layout="wide") # Apply the custom page icon


# --- Logo and Title Section ---
col_title, col_spacer, col_logo = st.columns([0.65, 0.05, 0.3])
with col_title:
    st.title("Feedback Trend Analysis")
with col_logo:
    try:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(current_script_dir, "..", "lookout_logo_02.png")
        st.image(logo_path, width=550)
    except FileNotFoundError:
        st.error(f"Logo image not found. Please check the path: {logo_path}.")
    except Exception as e:
        st.error(f"An error occurred while loading the logo: {e}.")


# --- Retrieve Processed Data from Session State ---
if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
    st.error("Welkom! To begin, please click the '**Home**' page from the sidebar to load the dataset automatically. All pages will be available right after ☺︎")
    st.stop()

all_data = st.session_state.processed_data.copy()

# --- Sidebar Filters ---
st.sidebar.header("Time and Trend Filters")
filtered_data_for_main_trend = all_data
start_date_main_trend = None
end_date_main_trend = None

if 'Time' in all_data.columns and pd.api.types.is_datetime64_any_dtype(all_data['Time']) and not all_data['Time'].isna().all():
    earliest_year = all_data['Time'].min().year if not all_data['Time'].empty else 2020
    min_date_data_time = pd.to_datetime(f"{earliest_year}-01-01").date()
    max_date_data_time = all_data['Time'].max().date()
    default_start_time = min_date_data_time
    default_end_time = max_date_data_time
    if default_start_time > default_end_time:
        default_start_time = default_end_time

    start_date_main_trend = st.sidebar.date_input(
        "Start date (for main trend chart & KPIs)",
        default_start_time,
        min_value=min_date_data_time,
        max_value=max_date_data_time,
        key="time_trend_analysis_start_date"
    )
    end_date_main_trend = st.sidebar.date_input(
        "End date (for main trend chart & KPIs)",
        default_end_time,
        min_value=min_date_data_time,
        max_value=max_date_data_time,
        key="time_trend_analysis_end_date"
    )
    
    if start_date_main_trend and end_date_main_trend:
        if start_date_main_trend > end_date_main_trend:
            st.sidebar.error("Error: Start date cannot be after end date for the main trend chart and KPIs.")
            filtered_data_for_main_trend = pd.DataFrame(columns=all_data.columns)
        else:
            start_datetime_time = pd.to_datetime(start_date_main_trend)
            end_datetime_time = pd.to_datetime(end_date_main_trend) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            filtered_data_for_main_trend = all_data[
                (all_data['Time'] >= start_datetime_time) &
                (all_data['Time'] <= end_datetime_time)
            ]
    else:
        filtered_data_for_main_trend = all_data.copy()
else:
    st.sidebar.warning("Date filter for main trend chart and KPIs cannot be applied: 'Time' column issue.")
    filtered_data_for_main_trend = pd.DataFrame(columns=all_data.columns)

frequency_options = {"Daily": "D", "Weekly": "W-Mon", "Monthly": "ME"}
selected_freq_label = st.sidebar.selectbox(
    "Select Trend Frequency (for main chart)",
    options=list(frequency_options.keys()),
    index=2,
    key="time_trend_analysis_frequency"
)
resample_freq = frequency_options[selected_freq_label]

# --- Color Customization ---
total_reviews_bar_color = '#deddd9'
positive_line_color = '#66c2ff'
neutral_line_color = '#808080'
negative_line_color = '#ff6666'
calendar_background_color = "#f2f4f7"
positive_calendar_colors = ['#e6f0fa', '#c2dffa', '#9ecff5', '#7bbfef', '#5a94c6']
neutral_calendar_colors = ['#f8f9fa', '#e9ecef', '#dee2e6', '#ced4da', '#adb5bd', '#6c757d']
negative_calendar_colors = ['#f0e6e6', '#e0cccc', '#d0b3b3', '#c09999', '#a85454']

# --- Monthly KPI Section ---
kpi_most_total_reviews_str = "N/A"
kpi_most_positive_reviews_str = "N/A"
kpi_most_negative_reviews_str = "N/A"

if not filtered_data_for_main_trend.empty and \
   'Time' in filtered_data_for_main_trend.columns and \
   pd.api.types.is_datetime64_any_dtype(filtered_data_for_main_trend['Time']) and \
   not filtered_data_for_main_trend['Time'].isna().all():
    try:
        monthly_total_reviews = filtered_data_for_main_trend.set_index('Time').resample('ME').size()
        if not monthly_total_reviews.empty and monthly_total_reviews.max() > 0:
            max_total_reviews_count = monthly_total_reviews.max()
            max_total_reviews_month_ts = monthly_total_reviews.idxmax()
            month_year_str = max_total_reviews_month_ts.strftime('%B %Y')
            kpi_most_total_reviews_str = f"{month_year_str}<br><span style='font-size:0.75em; color:#ffffff;'>({max_total_reviews_count} reviews)</span>"
        elif not monthly_total_reviews.empty:
            kpi_most_total_reviews_str = "No reviews in period"
        else:
            kpi_most_total_reviews_str = "No monthly data"
    except Exception:
        kpi_most_total_reviews_str = "Error calculating"

    if 'label' in filtered_data_for_main_trend.columns:
        try:
            positive_reviews_df = filtered_data_for_main_trend[filtered_data_for_main_trend['label'] == 'positive']
            if not positive_reviews_df.empty:
                monthly_positive_reviews = positive_reviews_df.set_index('Time').resample('ME').size()
                if not monthly_positive_reviews.empty and monthly_positive_reviews.max() > 0:
                    max_positive_reviews_count = monthly_positive_reviews.max()
                    max_positive_reviews_month_ts = monthly_positive_reviews.idxmax()
                    month_year_str = max_positive_reviews_month_ts.strftime('%B %Y')
                    kpi_most_positive_reviews_str = f"<span style='color:#66c2ff'>{month_year_str}</span><br><span style='font-size:0.75em; color:#ffffff;'>({max_positive_reviews_count} reviews)</span>"
                elif not monthly_positive_reviews.empty:
                    kpi_most_positive_reviews_str = "No positive reviews"
                else:
                    kpi_most_positive_reviews_str = "No monthly positive data"
            else:
                kpi_most_positive_reviews_str = "No positive reviews in period"
        except Exception:
            kpi_most_positive_reviews_str = "Error calculating"

        try:
            negative_reviews_df = filtered_data_for_main_trend[filtered_data_for_main_trend['label'] == 'negative']
            if not negative_reviews_df.empty:
                monthly_negative_reviews = negative_reviews_df.set_index('Time').resample('ME').size()
                if not monthly_negative_reviews.empty and monthly_negative_reviews.max() > 0:
                    max_negative_reviews_count = monthly_negative_reviews.max()
                    max_negative_reviews_month_ts = monthly_negative_reviews.idxmax()
                    month_year_str = max_negative_reviews_month_ts.strftime('%B %Y')
                    kpi_most_negative_reviews_str = f"<span style='color:#ff6666'>{month_year_str}</span><br><span style='font-size:0.75em; color:#ffffff;'>({max_negative_reviews_count} reviews)</span>"
                elif not monthly_negative_reviews.empty:
                    kpi_most_negative_reviews_str = "No negative reviews"
                else:
                    kpi_most_negative_reviews_str = "No monthly negative data"
            else:
                kpi_most_negative_reviews_str = "No negative reviews in period"
        except Exception:
            kpi_most_negative_reviews_str = "Error calculating"
    else:
        kpi_most_positive_reviews_str = "Label column missing"
        kpi_most_negative_reviews_str = "Label column missing"
elif filtered_data_for_main_trend.empty:
    kpi_most_total_reviews_str = "No data in selected period"
    kpi_most_positive_reviews_str = "No data in selected period"
    kpi_most_negative_reviews_str = "No data in selected period"
else:
    kpi_most_total_reviews_str = "Time data invalid"
    kpi_most_positive_reviews_str = "Time data invalid"
    kpi_most_negative_reviews_str = "Time data invalid"

st.markdown("<br>", unsafe_allow_html=True)

# --- Date Range Display ---
if start_date_main_trend and end_date_main_trend and start_date_main_trend <= end_date_main_trend:
    st.markdown(f"From **{start_date_main_trend.strftime('%Y-%m-%d')}** to **{end_date_main_trend.strftime('%Y-%m-%d')}**")
else:
    st.markdown("**Full Dataset**")

# --- Main Content Area ---
st.subheader(f"▸ {selected_freq_label} Review Volume and Sentiment Trends")
st.markdown("<br>", unsafe_allow_html=True)
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
with kpi_col1:
    st.markdown(create_styled_metric("Peak Review Volume Month", kpi_most_total_reviews_str), unsafe_allow_html=True)
with kpi_col2:
    st.markdown(create_styled_metric("Peak Positive Review Month", kpi_most_positive_reviews_str), unsafe_allow_html=True)
with kpi_col3:
    st.markdown(create_styled_metric("Peak Negative Review Month", kpi_most_negative_reviews_str), unsafe_allow_html=True)

# --- Main Trend Chart Logic ---
if filtered_data_for_main_trend.empty:
    st.warning("No review data available for the main trend chart based on current filter selection.")
else:
    if 'Time' not in filtered_data_for_main_trend.columns or \
       not pd.api.types.is_datetime64_any_dtype(filtered_data_for_main_trend['Time']) or \
       filtered_data_for_main_trend['Time'].isna().all():
        st.error("Valid 'Time' column is required for the main trend chart.")
    else:
        try:
            time_indexed_data = filtered_data_for_main_trend.set_index('Time')
            total_reviews_trend = time_indexed_data.resample(resample_freq).size().reset_index(name='Total Reviews')
            total_reviews_trend['Time'] = pd.to_datetime(total_reviews_trend['Time'])
            if 'label' in time_indexed_data.columns:
                sentiment_counts_trend = time_indexed_data.groupby([pd.Grouper(freq=resample_freq), 'label']).size().unstack(fill_value=0)
                for sentiment in ['positive', 'neutral', 'negative']:
                    if sentiment not in sentiment_counts_trend.columns:
                        sentiment_counts_trend[sentiment] = 0
                sentiment_counts_trend = sentiment_counts_trend.reindex(columns=['positive', 'neutral', 'negative'], fill_value=0)
                sentiment_counts_trend = sentiment_counts_trend.reset_index()
                sentiment_counts_trend['Time'] = pd.to_datetime(sentiment_counts_trend['Time'])
            else:
                st.warning("Sentiment 'label' column not found. Cannot display sentiment trends on main chart.")
                sentiment_counts_trend = pd.DataFrame(columns=['Time', 'positive', 'neutral', 'negative'])
                sentiment_counts_trend['Time'] = pd.to_datetime(sentiment_counts_trend['Time'])

            # Remove incomplete last period
            if end_date_main_trend and not filtered_data_for_main_trend.empty:
                end_date_ts = pd.to_datetime(end_date_main_trend)
                is_incomplete_period = False
                if selected_freq_label == "Weekly":
                    if end_date_ts.dayofweek != 0:
                        is_incomplete_period = True
                elif selected_freq_label == "Monthly":
                    if not end_date_ts.is_month_end:
                        is_incomplete_period = True
                if is_incomplete_period:
                    if not total_reviews_trend.empty:
                        total_reviews_trend = total_reviews_trend.iloc[:-1]
                    if not sentiment_counts_trend.empty:
                        sentiment_counts_trend = sentiment_counts_trend.iloc[:-1]

            fig = go.Figure()
            if not total_reviews_trend.empty:
                fig.add_trace(go.Bar(
                    x=total_reviews_trend['Time'],
                    y=total_reviews_trend['Total Reviews'],
                    name='Total Reviews',
                    marker_color=total_reviews_bar_color,
                    opacity=0.6
                ))
            if not sentiment_counts_trend.empty and 'Time' in sentiment_counts_trend.columns:
                sentiment_lines = {
                    'positive': {'name': '🤩', 'color': positive_line_color},
                    'neutral': {'name': '🙂', 'color': neutral_line_color},
                    'negative': {'name': '😒', 'color': negative_line_color}
                }
                for sentiment, props in sentiment_lines.items():
                    if sentiment in sentiment_counts_trend.columns:
                        fig.add_trace(go.Scatter(
                            x=sentiment_counts_trend['Time'],
                            y=sentiment_counts_trend[sentiment],
                            name=props['name'],
                            mode='lines+markers',
                            line=dict(color=props['color'], width=2)
                        ))
            fig.update_layout(
                title_text=f"{selected_freq_label} Trend of Review Volume and Sentiment Counts",
                xaxis_title="Time Period",
                yaxis_title="Number of Reviews",
                barmode='overlay',
                legend_title_text='Legend',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='#f8f9fa'
            )
            fig.update_yaxes(rangemode="tozero")
            if not total_reviews_trend.empty or \
               (not sentiment_counts_trend.empty and any(col in sentiment_counts_trend.columns for col in ['positive', 'neutral', 'negative'])):
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No data available to plot main trends for the selected period and {selected_freq_label.lower()} frequency.")
        except Exception as e:
            st.error(f"An error occurred while generating the main trend chart: {e}")

st.markdown("---")

# --- Calendar Heatmap Section ---
st.subheader("▸ Daily Review Sentiment Calendar")
st.markdown("##### Peak Review Days (Filtered Period)")
st.markdown("<br>", unsafe_allow_html=True)

# --- Modified Peak Review Days (Filtered Period) KPIs ---
kpi_peak_day_filtered_total_str = "N/A"
kpi_peak_day_filtered_positive_str = "N/A"
kpi_peak_day_filtered_negative_str = "N/A"
expander_total_dates = None
expander_positive_dates = None
expander_negative_dates = None

if not filtered_data_for_main_trend.empty and \
   'Time' in filtered_data_for_main_trend.columns and \
   pd.api.types.is_datetime64_any_dtype(filtered_data_for_main_trend['Time']) and \
   not filtered_data_for_main_trend['Time'].isna().all():
    try:
        daily_total_reviews_filtered = filtered_data_for_main_trend.groupby(filtered_data_for_main_trend['Time'].dt.date).size()
        if not daily_total_reviews_filtered.empty and daily_total_reviews_filtered.max() > 0:
            max_count = daily_total_reviews_filtered.max()
            peak_dates = daily_total_reviews_filtered[daily_total_reviews_filtered == max_count].index
            peak_date_strs = [date.strftime('%B %d, %Y') for date in peak_dates]
            if len(peak_date_strs) > 1:
                kpi_peak_day_filtered_total_str = f"Multiple<br><span style='font-size:0.75em; color:#ffffff;'>({max_count} reviews each)</span>"
                expander_total_dates = peak_date_strs
            else:
                kpi_peak_day_filtered_total_str = f"{peak_date_strs[0]}<br><span style='font-size:0.75em; color:#ffffff;'>({max_count} reviews)</span>"
        elif not daily_total_reviews_filtered.empty:
            kpi_peak_day_filtered_total_str = "No reviews in period"
        else:
            kpi_peak_day_filtered_total_str = "No daily data"
    except Exception:
        kpi_peak_day_filtered_total_str = "Error calculating"

    if 'label' in filtered_data_for_main_trend.columns:
        try:
            positive_filtered_data = filtered_data_for_main_trend[filtered_data_for_main_trend['label'] == 'positive']
            if not positive_filtered_data.empty:
                daily_positive_reviews_filtered = positive_filtered_data.groupby(positive_filtered_data['Time'].dt.date).size()
                if not daily_positive_reviews_filtered.empty and daily_positive_reviews_filtered.max() > 0:
                    max_count = daily_positive_reviews_filtered.max()
                    peak_dates = daily_positive_reviews_filtered[daily_positive_reviews_filtered == max_count].index
                    peak_date_strs = [date.strftime('%B %d, %Y') for date in peak_dates]
                    if len(peak_date_strs) > 1:
                        kpi_peak_day_filtered_positive_str = f"<span style='color:#66c2ff'>Multiple</span><br><span style='font-size:0.75em; color:#ffffff;'>({max_count} reviews each)</span>"
                        expander_positive_dates = peak_date_strs
                    else:
                        kpi_peak_day_filtered_positive_str = f"<span style='color:#66c2ff'>{peak_date_strs[0]}</span><br><span style='font-size:0.75em; color:#ffffff;'>({max_count} reviews)</span>"
                elif not daily_positive_reviews_filtered.empty:
                    kpi_peak_day_filtered_positive_str = "No positive reviews"
                else:
                    kpi_peak_day_filtered_positive_str = "No daily positive data"
            else:
                kpi_peak_day_filtered_positive_str = "No positive reviews in period"
        except Exception:
            kpi_peak_day_filtered_positive_str = "Error calculating"
        
        try:
            negative_filtered_data = filtered_data_for_main_trend[filtered_data_for_main_trend['label'] == 'negative']
            if not negative_filtered_data.empty:
                daily_negative_reviews_filtered = negative_filtered_data.groupby(negative_filtered_data['Time'].dt.date).size()
                if not daily_negative_reviews_filtered.empty and daily_negative_reviews_filtered.max() > 0:
                    max_count = daily_negative_reviews_filtered.max()
                    peak_dates = daily_negative_reviews_filtered[daily_negative_reviews_filtered == max_count].index
                    peak_date_strs = [date.strftime('%B %d, %Y') for date in peak_dates]
                    if len(peak_date_strs) > 1:
                        kpi_peak_day_filtered_negative_str = f"<span style='color:#ff6666'>Multiple</span><br><span style='font-size:0.75em; color:#ffffff;'>({max_count} reviews each)</span>"
                        expander_negative_dates = peak_date_strs
                    else:
                        kpi_peak_day_filtered_negative_str = f"<span style='color:#ff6666'>{peak_date_strs[0]}</span><br><span style='font-size:0.75em; color:#ffffff;'>({max_count} reviews)</span>"
                elif not daily_negative_reviews_filtered.empty:
                    kpi_peak_day_filtered_negative_str = "No negative reviews"
                else:
                    kpi_peak_day_filtered_negative_str = "No daily negative data"
            else:
                kpi_peak_day_filtered_negative_str = "No negative reviews in period"
        except Exception:
            kpi_peak_day_filtered_negative_str = "Error calculating"
    else:
        kpi_peak_day_filtered_positive_str = "Label column missing"
        kpi_peak_day_filtered_negative_str = "Label column missing"
else:
    kpi_peak_day_filtered_total_str = "No data for period"
    kpi_peak_day_filtered_positive_str = "No data for period"
    kpi_peak_day_filtered_negative_str = "No data for period"

# Display KPIs in a single row
peak_day_filt_col1, peak_day_filt_col2, peak_day_filt_col3 = st.columns(3)
with peak_day_filt_col1:
    st.markdown(create_styled_metric("Busiest Day (Filtered Period)", kpi_peak_day_filtered_total_str), unsafe_allow_html=True)
    if expander_total_dates:
        with st.expander("See all peak days"):
            for date in expander_total_dates:
                st.markdown(f"- {date}")
with peak_day_filt_col2:
    st.markdown(create_styled_metric("Most Positive Day (Filtered Period)", kpi_peak_day_filtered_positive_str), unsafe_allow_html=True)
    if expander_positive_dates:
        with st.expander("See all peak positive days"):
            for date in expander_positive_dates:
                st.markdown(f"- {date}")
with peak_day_filt_col3:
    st.markdown(create_styled_metric("Most Negative Day (Filtered Period)", kpi_peak_day_filtered_negative_str), unsafe_allow_html=True)
    if expander_negative_dates:
        with st.expander("See all peak negative days"):
            for date in expander_negative_dates:
                st.markdown(f"- {date}")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if 'Time' in all_data.columns and pd.api.types.is_datetime64_any_dtype(all_data['Time']) and not all_data['Time'].isna().all():
    all_data_calendar = all_data.copy()
    all_data_calendar['Time'] = pd.to_datetime(all_data_calendar['Time'])
    all_data_calendar['YearMonth'] = all_data_calendar['Time'].dt.to_period('M')
    available_months = sorted(all_data_calendar['YearMonth'].unique().astype(str))
    
    if not available_months:
        st.warning("No month data available to select for calendar view.")
    else:
        month_options_map = {ym_str: pd.Period(ym_str).strftime('%Y-%m (%B %Y)') for ym_str in available_months}
        st.markdown("##### Calendar View (Filtered Month)")
        selected_month_display = st.selectbox(
            "Select Month for Calendar View",
            options=list(month_options_map.values()),
            index=len(month_options_map) - 1,
            key="calendar_month_select"
        )
        selected_month_str = None
        for ym_str_orig, display_str in month_options_map.items():
            if display_str == selected_month_display:
                selected_month_str = ym_str_orig
                break
        
        if selected_month_str:
            selected_period = pd.Period(selected_month_str, freq='M')
            month_data = all_data_calendar[all_data_calendar['Time'].dt.to_period('M') == selected_period]
            bullet_busiest_day_this_month = "N/A"
            bullet_most_positive_day_this_month = "N/A"
            bullet_most_negative_day_this_month = "N/A"

            if not month_data.empty:
                try:
                    daily_total_reviews = month_data.groupby(month_data['Time'].dt.date).size()
                    if not daily_total_reviews.empty and daily_total_reviews.max() > 0:
                        max_count = daily_total_reviews.max()
                        peak_date = daily_total_reviews.idxmax()
                        peak_date_str = peak_date.strftime('%B %d, %Y')
                        bullet_busiest_day_this_month = f"{peak_date_str} ({max_count} reviews)"
                    elif not daily_total_reviews.empty:
                        bullet_busiest_day_this_month = "No reviews this month"
                    else:
                        bullet_busiest_day_this_month = "No daily data"
                except Exception:
                    bullet_busiest_day_this_month = "Error calculating"

                if 'label' in month_data.columns:
                    try:
                        positive_month_data = month_data[month_data['label'] == 'positive']
                        if not positive_month_data.empty:
                            daily_positive_reviews = positive_month_data.groupby(positive_month_data['Time'].dt.date).size()
                            if not daily_positive_reviews.empty and daily_positive_reviews.max() > 0:
                                max_count = daily_positive_reviews.max()
                                peak_date = daily_positive_reviews.idxmax()
                                peak_date_str = peak_date.strftime('%B %d, %Y')
                                bullet_most_positive_day_this_month = f"{peak_date_str} ({max_count} reviews)"
                            elif not daily_positive_reviews.empty:
                                bullet_most_positive_day_this_month = "No positive reviews this month"
                            else:
                                bullet_most_positive_day_this_month = "No daily positive data"
                        else:
                            bullet_most_positive_day_this_month = "No positive reviews this month"
                    except Exception:
                        bullet_most_positive_day_this_month = "Error calculating"
                    
                    try:
                        negative_month_data = month_data[month_data['label'] == 'negative']
                        if not negative_month_data.empty:
                            daily_negative_reviews = negative_month_data.groupby(negative_month_data['Time'].dt.date).size()
                            if not daily_negative_reviews.empty and daily_negative_reviews.max() > 0:
                                max_count = daily_negative_reviews.max()
                                peak_date = daily_negative_reviews.idxmax()
                                peak_date_str = peak_date.strftime('%B %d, %Y')
                                bullet_most_negative_day_this_month = f"{peak_date_str} ({max_count} reviews)"
                            elif not daily_negative_reviews.empty:
                                bullet_most_negative_day_this_month = "No negative reviews this month"
                            else:
                                bullet_most_negative_day_this_month = "No daily negative data"
                        else:
                            bullet_most_negative_day_this_month = "No negative reviews this month"
                    except Exception:
                        bullet_most_negative_day_this_month = "Error calculating"
                else:
                    bullet_most_positive_day_this_month = "Label column missing"
                    bullet_most_negative_day_this_month = "Label column missing"
            else:
                bullet_busiest_day_this_month = "No data for this month"
                bullet_most_positive_day_this_month = "No data for this month"
                bullet_most_negative_day_this_month = "No data for this month"
            
            st.markdown(f"- Busiest Day: **{bullet_busiest_day_this_month}**")
            st.markdown(f"- Most Positive Day: **{bullet_most_positive_day_this_month}**")
            st.markdown(f"- Most Negative Day: **{bullet_most_negative_day_this_month}**")
            st.markdown("<br>", unsafe_allow_html=True)
            
            def get_calendar_data(df, current_selected_month_period, sentiment_label=None):
                df_copy = df.copy()
                df_copy['Time'] = pd.to_datetime(df_copy['Time'])
                current_sentiment_label_for_hover = "Reviews"
                if sentiment_label:
                    if 'label' not in df_copy.columns: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
                    df_sentiment_filtered = df_copy[df_copy['label'] == sentiment_label]
                    current_sentiment_label_for_hover = sentiment_label.capitalize() + " Reviews"
                else:
                    df_sentiment_filtered = df_copy
                daily_counts = df_sentiment_filtered.groupby(df_sentiment_filtered['Time'].dt.date).size()
                start_of_month = current_selected_month_period.start_time.date()
                end_of_month = current_selected_month_period.end_time.date()
                all_days_in_month_index = pd.date_range(start_of_month, end_of_month, freq='D')
                calendar_df = pd.DataFrame({'Date': all_days_in_month_index})
                calendar_df['Date_date'] = calendar_df['Date'].dt.date
                daily_counts_df = daily_counts.reset_index()
                daily_counts_df.columns = ['Date_date', 'Count']
                calendar_df = pd.merge(calendar_df, daily_counts_df, on='Date_date', how='left').fillna(0)
                calendar_df['Count'] = calendar_df['Count'].astype(int)
                calendar_df['DayOfMonth'] = calendar_df['Date'].dt.day
                calendar_df['DayOfWeek'] = calendar_df['Date'].dt.dayofweek
                first_day_offset = calendar_df['Date'].iloc[0].dayofweek
                calendar_df['WeekOfMonth'] = (calendar_df['DayOfMonth'] + first_day_offset - 1) // 7
                try:
                    heatmap_matrix = calendar_df.pivot_table(index='WeekOfMonth', columns='DayOfWeek', values='Count', fill_value=0)
                    day_text_matrix = calendar_df.pivot_table(index='WeekOfMonth', columns='DayOfWeek', values='DayOfMonth', fill_value=np.nan)
                    hover_text_matrix = heatmap_matrix.copy()
                    for r_idx in hover_text_matrix.index:
                        for c_idx in hover_text_matrix.columns:
                            count_val = hover_text_matrix.loc[r_idx, c_idx]
                            day_num_obj = day_text_matrix.loc[r_idx, c_idx]
                            if pd.notna(day_num_obj):
                                if count_val > 0:
                                    hover_text_matrix.loc[r_idx, c_idx] = f"{current_sentiment_label_for_hover}: {int(count_val)}"
                                else:
                                    hover_text_matrix.loc[r_idx, c_idx] = f"No {current_sentiment_label_for_hover.lower()}"
                            else:
                                hover_text_matrix.loc[r_idx, c_idx] = ""
                    for i in range(7):
                        if i not in heatmap_matrix.columns: heatmap_matrix[i] = 0
                        if i not in day_text_matrix.columns: day_text_matrix[i] = np.nan
                        if i not in hover_text_matrix.columns: hover_text_matrix[i] = ""
                    heatmap_matrix = heatmap_matrix.reindex(columns=list(range(7)), fill_value=0)
                    day_text_matrix = day_text_matrix.reindex(columns=list(range(7)), fill_value=np.nan)
                    hover_text_matrix = hover_text_matrix.reindex(columns=list(range(7)), fill_value="")
                    day_text_matrix = day_text_matrix.applymap(lambda x: str(int(x)) if pd.notna(x) else '')
                except Exception as e:
                    st.error(f"Error pivoting calendar data for {current_sentiment_label_for_hover}: {e}")
                    heatmap_matrix = pd.DataFrame(index=range(6), columns=range(7)).fillna(0)
                    day_text_matrix = pd.DataFrame(index=range(6), columns=range(7)).fillna('')
                    hover_text_matrix = pd.DataFrame(index=range(6), columns=range(7)).fillna('')
                return heatmap_matrix, day_text_matrix, hover_text_matrix
            
            def plot_calendar_heatmap(heatmap_matrix, day_text_matrix, hover_text_matrix, title, sentiment_colors_list, base_bg_color):
                days_of_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                if heatmap_matrix.empty:
                    st.info(f"No data to display for {title.lower()}.")
                    return
                z_min = 0
                z_max = heatmap_matrix.values.max()
                plotly_colorscale = [[0, base_bg_color]]
                if z_max > 0:
                    if len(sentiment_colors_list) == 1:
                        plotly_colorscale.append([0.00001, sentiment_colors_list[0]])
                        plotly_colorscale.append([1.0, sentiment_colors_list[0]])
                    elif len(sentiment_colors_list) > 1:
                        plotly_colorscale.append([0.00001, sentiment_colors_list[0]])
                        for i_color, color_hex in enumerate(sentiment_colors_list):
                            point = (i_color + 1) / len(sentiment_colors_list)
                            plotly_colorscale.append([point, color_hex])
                else:
                    plotly_colorscale.append([1.0, base_bg_color])
                fig_cal = go.Figure(data=go.Heatmap(
                    z=heatmap_matrix.values,
                    x=days_of_week,
                    y=heatmap_matrix.index,
                    colorscale=plotly_colorscale,
                    zmin=z_min,
                    zmax=z_max if z_max > 0 else 1,
                    text=day_text_matrix.values,
                    texttemplate="%{text}",
                    hovertext=hover_text_matrix.values,
                    hoverinfo='text',
                    showscale=False,
                    xgap=1, ygap=1,
                    hoverongaps=False
                ))
                fig_cal.update_layout(
                    title=title,
                    xaxis_title=None,
                    yaxis_title=None,
                    yaxis_autorange='reversed',
                    plot_bgcolor='rgba(0,0,0,0)',
                    # The line below has been changed for a transparent background
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                fig_cal.update_xaxes(
                    side="top",
                    constrain="domain"
                )
                fig_cal.update_yaxes(
                    showticklabels=False,
                    ticks="",
                    scaleanchor="x",
                    scaleratio=1,
                    constrain="domain"
                )
                st.plotly_chart(fig_cal, use_container_width=True)

            sentiments_to_plot_calendar = [
                ('positive', "Positive 😃", positive_calendar_colors),
                ('neutral', "Neutral 🙂", neutral_calendar_colors),
                ('negative', "Negative 😒", negative_calendar_colors)
            ]
            cal_col1, cal_col2, cal_col3 = st.columns(3)
            columns_map_calendar = {0: cal_col1, 1: cal_col2, 2: cal_col3}
            if not month_data.empty and 'label' in month_data.columns:
                for i, (s_label, s_title, s_colors) in enumerate(sentiments_to_plot_calendar):
                    with columns_map_calendar[i]:
                        hm_df, dt_df, ht_df = get_calendar_data(month_data, selected_period, sentiment_label=s_label)
                        plot_calendar_heatmap(hm_df, dt_df, ht_df, s_title, s_colors, calendar_background_color)
            elif not month_data.empty:
                with st.container():
                    st.warning("Sentiment 'label' column not found in data for the selected month. Cannot display sentiment-specific calendars.")
            else:
                st.info(f"No review data found for {selected_month_display} to display in calendars.")
        else:
            st.error("Month selection failed. Cannot display calendars.")
else:
    st.warning("Cannot display calendar view: 'Time' column is missing, invalid, or contains no valid dates in the overall dataset.")

st.markdown("---")