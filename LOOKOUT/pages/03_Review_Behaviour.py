import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from PIL import Image # Import Image
from pathlib import Path # Import Path

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

page_icon_review_behaviour = "👥" # Default emoji icon for this page
try:
    img_icon_review_behaviour = Image.open(LOOKOUT_LOGO_PATH)
    page_icon_review_behaviour = img_icon_review_behaviour
except Exception as e:
    st.warning(f"Cannot use '{LOOKOUT_LOGO_PATH.name}' as page icon for Review Behaviour: {e}. Using default emoji icon.")

st.set_page_config(page_title="Review Behaviour Analysis - A'DAM LOOKOUT", page_icon=page_icon_review_behaviour, layout="wide")


# --- Title and Logo ---
col_title, col_spacer, col_logo = st.columns([0.65, 0.05, 0.3])
with col_title:
    st.title("Review Behaviour Analysis")
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
st.sidebar.header("Time Filters")
filtered_data_for_main_trend = all_data
start_date_main_trend = None
end_date_main_trend = None

if 'Time' in all_data.columns and pd.api.types.is_datetime64_any_dtype(all_data['Time']) and not all_data['Time'].isna().all():
    min_year = all_data['Time'].dt.year.min() if not all_data['Time'].empty else 2020
    min_date_data_time = pd.to_datetime(f"{min_year}-01-01").date()
    max_date_data_time = all_data['Time'].max().date()
    default_start_time = min_date_data_time
    default_end_time = max_date_data_time
    if default_start_time > default_end_time:
        default_start_time = default_end_time
    start_date_main_trend = st.sidebar.date_input(
        "Start date (for charts)",
        default_start_time,
        min_value=min_date_data_time,
        max_value=max_date_data_time,
        key="review_behaviour_start_date"
    )
    end_date_main_trend = st.sidebar.date_input(
        "End date (for charts)",
        default_end_time,
        min_value=min_date_data_time,
        max_value=max_date_data_time,
        key="review_behaviour_end_date"
    )
    if start_date_main_trend and end_date_main_trend:
        if start_date_main_trend > end_date_main_trend:
            st.sidebar.error("Error: Start date cannot be after end date for the charts.")
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
    st.sidebar.warning("Date filter for charts cannot be applied: 'Time' column issue.")
    filtered_data_for_main_trend = pd.DataFrame(columns=all_data.columns)

# --- Hourly Sentiment Distribution ---
st.subheader("▸ Hourly Sentiment Distribution")
st.markdown("<br>", unsafe_allow_html=True)

if not filtered_data_for_main_trend.empty and \
   'Time' in filtered_data_for_main_trend.columns and \
   pd.api.types.is_datetime64_any_dtype(filtered_data_for_main_trend['Time']) and \
   not filtered_data_for_main_trend['Time'].isna().all() and \
   'label' in filtered_data_for_main_trend.columns:
    try:
        hourly_data = filtered_data_for_main_trend.copy()
        hourly_data['Hour'] = hourly_data['Time'].dt.hour
        hourly_sentiment_counts = hourly_data.groupby(['Hour', 'label']).size().unstack(fill_value=0)
        for sentiment in ['positive', 'neutral', 'negative']:
            if sentiment not in hourly_sentiment_counts.columns:
                hourly_sentiment_counts[sentiment] = 0
        hourly_sentiment_counts = hourly_sentiment_counts.reindex(columns=['positive', 'neutral', 'negative'], fill_value=0)
        hourly_sentiment_counts = hourly_sentiment_counts.reset_index()
        hourly_sentiment_counts['Total'] = hourly_sentiment_counts[['positive', 'neutral', 'negative']].sum(axis=1)
        hourly_sentiment_counts['Positive_Percentage'] = (
            hourly_sentiment_counts['positive'] / hourly_sentiment_counts['Total'] * 100
        ).fillna(0)

        # --- KPI Calculations ---
        kpi_peak_duration_total_str = "N/A"
        kpi_peak_duration_positive_str = "N/A"
        kpi_peak_duration_negative_str = "N/A"
        expander_positive_hours = None
        expander_negative_hours = None

        def get_time_duration(hour):
            start_hour = f"{hour:02d}:00"
            end_hour = f"{(hour + 1) % 24:02d}:00"
            return f"{start_hour} - {end_hour}"

        if not hourly_sentiment_counts.empty and hourly_sentiment_counts['Total'].sum() > 0:
            max_total = hourly_sentiment_counts['Total'].max()
            peak_total_hours = hourly_sentiment_counts[hourly_sentiment_counts['Total'] == max_total]['Hour'].tolist()
            if len(peak_total_hours) > 1:
                kpi_peak_duration_total_str = f"Multiple<br><span style='font-size:0.75em; color:#ffffff;'>({max_total} reviews each)</span>"
            else:
                peak_total_hour = peak_total_hours[0]
                kpi_peak_duration_total_str = get_time_duration(peak_total_hour)

            if hourly_sentiment_counts['positive'].max() > 0:
                max_positive = hourly_sentiment_counts['positive'].max()
                peak_positive_hours = hourly_sentiment_counts[hourly_sentiment_counts['positive'] == max_positive]['Hour'].tolist()
                if len(peak_positive_hours) > 1:
                    kpi_peak_duration_positive_str = f"<span style='color:#66c2ff'>Multiple</span>"
                    expander_positive_hours = [get_time_duration(hour) for hour in peak_positive_hours]
                else:
                    peak_positive_hour = peak_positive_hours[0]
                    kpi_peak_duration_positive_str = f"<span style='color:#66c2ff'>{get_time_duration(peak_positive_hour)}</span>"
            else:
                kpi_peak_duration_positive_str = "No Positive Reviews"

            if hourly_sentiment_counts['negative'].max() > 0:
                max_negative = hourly_sentiment_counts['negative'].max()
                peak_negative_hours = hourly_sentiment_counts[hourly_sentiment_counts['negative'] == max_negative]['Hour'].tolist()
                if len(peak_negative_hours) > 1:
                    kpi_peak_duration_negative_str = f"<span style='color:#ff6666'>Multiple</span>"
                    expander_negative_hours = [get_time_duration(hour) for hour in peak_negative_hours]
                else:
                    peak_negative_hour = peak_negative_hours[0]
                    kpi_peak_duration_negative_str = f"<span style='color:#ff6666'>{get_time_duration(peak_negative_hour)}</span>"
            else:
                kpi_peak_duration_negative_str = "No Negative Reviews"

        # Display KPIs
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        with kpi_col1:
            st.markdown(create_styled_metric("Peak Review Duration", kpi_peak_duration_total_str), unsafe_allow_html=True)
            if len(peak_total_hours) > 1:
                with st.expander("See all peak review durations"):
                    for hour in peak_total_hours:
                        st.markdown(f"- {get_time_duration(hour)}")
        with kpi_col2:
            st.markdown(create_styled_metric("Peak Positive Review Duration", kpi_peak_duration_positive_str), unsafe_allow_html=True)
            if expander_positive_hours:
                with st.expander("See all peak positive durations"):
                    for duration in expander_positive_hours:
                        st.markdown(f"- {duration}")
        with kpi_col3:
            st.markdown(create_styled_metric("Peak Negative Review Duration", kpi_peak_duration_negative_str), unsafe_allow_html=True)
            if expander_negative_hours:
                with st.expander("See all peak negative durations"):
                    for duration in expander_negative_hours:
                        st.markdown(f"- {duration}")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Stacked Bar Chart ---
        fig_hourly = go.Figure()
        sentiment_colors = {
            'positive': '#a3bbce',
            'neutral': '#d6d0bc',
            'negative': '#d8575d'
        }
        for sentiment in ['positive', 'neutral', 'negative']:
            fig_hourly.add_trace(go.Bar(
                x=hourly_sentiment_counts['Hour'],
                y=hourly_sentiment_counts[sentiment],
                name=sentiment.capitalize(),
                marker_color=sentiment_colors[sentiment]
            ))
        fig_hourly.update_layout(
            title="Sentiment Distribution by Hour of Review Submission",
            xaxis_title="Hour of Day (24-hour format)",
            yaxis_title="Number of Reviews",
            barmode='stack',
            legend_title_text='Sentiment',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='#f8f9fa',
            xaxis=dict(tickmode='linear', tick0=0, dtick=1)
        )
        fig_hourly.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig_hourly, use_container_width=True)

        # --- Clock-Shaped Radial Bar Charts ---
        st.markdown("**Clock-view radial chart**")
        # Prepare data for radial charts
        hours = list(range(24))
        radial_data = hourly_sentiment_counts.set_index('Hour')
        radial_data = radial_data.reindex(hours, fill_value=0).reset_index()

        # Create three columns for the radial charts
        col_rad1, col_rad2, col_rad3 = st.columns(3)

        # Function to create a minimalistic radial bar chart for a sentiment
        def create_radial_chart(sentiment, color, column):
            fig = go.Figure()
            fig.add_trace(go.Barpolar(
                r=radial_data[sentiment],
                theta=[(h / 24) * 360 for h in radial_data['Hour']],
                marker_color=color,
                marker_line_width=0,
                opacity=0.7,
                name=sentiment.capitalize()
            ))
            fig.update_layout(
                font=dict(family="Arial", size=12),
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, radial_data[sentiment].max() * 1.2] if radial_data[sentiment].max() > 0 else [0, 1],
                        showticklabels=True,
                        ticks='',
                        showgrid=False,
                        showline=False
                    ),
                    angularaxis=dict(
                        tickmode='array',
                        tickvals=[(h / 24) * 360 for h in range(24)],
                        ticktext=[f"{h:02d}" for h in range(24)],
                        rotation=90,
                        direction="clockwise",
                        showgrid=False,
                        showline=False
                    ),
                    bgcolor='rgba(0,0,0,0)'
                ),
                showlegend=False,
                height=350,
                margin=dict(l=30, r=30, t=80, b=30),
                paper_bgcolor='#f8f9fa',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            with column:
                st.plotly_chart(fig, use_container_width=True)

        # Create radial charts for each sentiment
        create_radial_chart('positive', '#a3bbce', col_rad1)
        create_radial_chart('neutral', '#d6d0bc', col_rad2)
        create_radial_chart('negative', '#d8575d', col_rad3)

        if not hourly_sentiment_counts.empty and hourly_sentiment_counts['Total'].sum() > 0:
            valid_hours = hourly_sentiment_counts[hourly_sentiment_counts['Total'] >= 5]
            if not valid_hours.empty:
                peak_positive_hour = valid_hours.loc[
                    valid_hours['Positive_Percentage'].idxmax(), 'Hour']
                positive_percentage = valid_hours['Positive_Percentage'].max()
                peak_negative_hour = valid_hours.loc[
                    valid_hours['negative'].idxmax(), 'Hour'] if valid_hours['negative'].max() > 0 else None
                negative_count = valid_hours['negative'].max() if peak_negative_hour is not None else 0
            else:
                st.markdown(
                    "**Insight:** Not enough data (no hours with 5+ reviews) to recommend optimal email timing. "
                    "Try expanding the date range to include more reviews."
                )
        else:
            st.markdown("**Insight:** No review data available for the selected period.")
    except Exception as e:
        st.error(f"Error generating hourly sentiment distribution: {e}")
else:
    st.info("No data available for hourly sentiment distribution due to missing 'Time' or 'label' column or empty filtered data.")

st.markdown("---")



# --- Weekday vs. Weekend Trends ---
st.subheader("▸ Weekday vs. Weekend Trends")
st.markdown("<br>", unsafe_allow_html=True)

if not filtered_data_for_main_trend.empty and \
   'Time' in filtered_data_for_main_trend.columns and \
   pd.api.types.is_datetime64_any_dtype(filtered_data_for_main_trend['Time']) and \
   not filtered_data_for_main_trend['Time'].isna().all() and \
   'label' in filtered_data_for_main_trend.columns and \
   'Rating' in filtered_data_for_main_trend.columns:
    try:
        daytype_data = filtered_data_for_main_trend.copy()
        daytype_data['DayType'] = daytype_data['Time'].dt.dayofweek.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

        # --- Sentiment ---
        st.markdown("##### Sentiment Distribution")
        st.markdown("<br>", unsafe_allow_html=True)

        if not filtered_data_for_main_trend.empty and \
        'Time' in filtered_data_for_main_trend.columns and \
        pd.api.types.is_datetime64_any_dtype(filtered_data_for_main_trend['Time']) and \
        not filtered_data_for_main_trend['Time'].isna().all() and \
        'label' in filtered_data_for_main_trend.columns:
            # Create daytype_sentiment_counts
            daytype_data = filtered_data_for_main_trend.copy()
            daytype_data['DayType'] = daytype_data['Time'].dt.dayofweek.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
            daytype_sentiment_counts = daytype_data.groupby(['DayType', 'label']).size().unstack(fill_value=0)
            for sentiment in ['positive', 'neutral', 'negative']:
                if sentiment not in daytype_sentiment_counts.columns:
                    daytype_sentiment_counts[sentiment] = 0
            daytype_sentiment_counts = daytype_sentiment_counts.reindex(columns=['positive', 'neutral', 'negative'], fill_value=0)
            daytype_sentiment_counts['Total'] = daytype_sentiment_counts[['positive', 'neutral', 'negative']].sum(axis=1)
            daytype_sentiment_counts['Positive_Percentage'] = (daytype_sentiment_counts['positive'] / daytype_sentiment_counts['Total'] * 100).fillna(0)
            daytype_sentiment_counts['Neutral_Percentage'] = (daytype_sentiment_counts['neutral'] / daytype_sentiment_counts['Total'] * 100).fillna(0)
            daytype_sentiment_counts['Negative_Percentage'] = (daytype_sentiment_counts['negative'] / daytype_sentiment_counts['Total'] * 100).fillna(0)
            daytype_sentiment_counts = daytype_sentiment_counts.reset_index()

            # Prepare table data
            table_data = daytype_sentiment_counts[['DayType', 'Total', 'Positive_Percentage', 'Neutral_Percentage', 'Negative_Percentage']].copy()
            table_data['Positive_Percentage'] = table_data['Positive_Percentage'].apply(lambda x: f"{x:.1f}%")
            table_data['Neutral_Percentage'] = table_data['Neutral_Percentage'].apply(lambda x: f"{x:.1f}%")
            table_data['Negative_Percentage'] = table_data['Negative_Percentage'].apply(lambda x: f"{x:.1f}%")
            table_data['Total'] = table_data['Total'].astype(int)
            table_data = table_data.rename(columns={
                'DayType': 'Day Type',
                'Total': 'Total Reviews',
                'Positive_Percentage': 'Positive (%)',
                'Neutral_Percentage': 'Neutral (%)',
                'Negative_Percentage': 'Negative (%)'
            })

            # Display styled table
            st.dataframe(
                table_data,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Day Type': st.column_config.TextColumn(width='medium', help='Weekday or Weekend'),
                    'Positive (%)': st.column_config.TextColumn(width='small'),
                    'Neutral (%)': st.column_config.TextColumn(width='small'),
                    'Negative (%)': st.column_config.TextColumn(width='small'),
                    'Total Reviews': st.column_config.TextColumn(width='small')
                },
                column_order=['Day Type', 'Positive (%)', 'Neutral (%)', 'Negative (%)', 'Total Reviews']
            )

            # Center-align table content using custom CSS
            st.markdown(
                """
                <style>
                .stDataFrame [data-testid="stTable"] th, 
                .stDataFrame [data-testid="stTable"] td {
                    text-align: center !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("No sentiment data available for the selected period.")

        # --- Star Rating Bar Chart ---
        st.markdown("##### Star Rating Distribution")
        daytype_data['Rating'] = daytype_data['Rating'].astype(str)
        daytype_rating_counts = daytype_data.groupby(['DayType', 'Rating']).size().unstack(fill_value=0)
        for rating in ['1', '2', '3', '4', '5']:
            if rating not in daytype_rating_counts.columns:
                daytype_rating_counts[rating] = 0
        daytype_rating_counts = daytype_rating_counts.reindex(columns=['1', '2', '3', '4', '5'], fill_value=0)
        daytype_rating_counts = daytype_rating_counts.reset_index()

        daytype_rating_counts['Total'] = daytype_rating_counts[['1', '2', '3', '4', '5']].sum(axis=1)
        for rating in ['1', '2', '3', '4', '5']:
            daytype_rating_counts[f'{rating}_Percentage'] = (
                daytype_rating_counts[rating] / daytype_rating_counts['Total'] * 100
            ).fillna(0)

        fig_daytype = go.Figure()
        rating_color_map = {
            '1': '#e8e8e8',  # Very light gray
            '2': '#cccccc',  # Light gray
            '3': '#b0b0b0',  # Medium-light gray
            '4': '#989898',  # Slightly lighter than #808080
            '5': '#808080'
        }
        for rating in ['1', '2', '3', '4', '5']:
            fig_daytype.add_trace(go.Bar(
                x=daytype_rating_counts['DayType'],
                y=daytype_rating_counts[f'{rating}_Percentage'],
                name=f'{rating} Star',
                marker_color=rating_color_map[rating],
                text=[f"{val:.1f}%" for val in daytype_rating_counts[f'{rating}_Percentage']],
                textposition='auto'
            ))
        fig_daytype.update_layout(
            title="Percentage of Reviews by Star Rating: Weekday vs. Weekend",
            xaxis_title="Day Type",
            yaxis_title="Percentage of Reviews (%)",
            barmode='group',
            legend_title_text='Rating',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='#f8f9fa',
            yaxis=dict(range=[0, 100])
        )
        st.plotly_chart(fig_daytype, use_container_width=True)

        
    
        

    except Exception as e:
        st.error(f"Error generating weekday vs. weekend trends: {e}")
else:
    st.info("No data available for weekday vs. weekend trends due to missing 'Time', 'label', or 'Rating' column or empty filtered data.")

st.markdown("---")

