import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from pathlib import Path

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
st.set_page_config(page_title="Review Behaviour Analysis - Madam", layout="wide")

# --- Logo and Title Section ---
try:
    logo_path = Path(__file__).resolve().parent.parent / "madam_logo_02.png"
    madam_logo_display = Image.open(logo_path)
    col_title, col_spacer, col_logo = st.columns([0.75, 0.05, 0.2])
    with col_title:
        st.title("Review Behaviour Analysis")
    with col_logo:
        st.image(madam_logo_display, width=350)
except FileNotFoundError:
    st.error(f"Logo image 'madam_logo_02.png' not found at expected path: {logo_path}.")
    st.title("🗓️ Review Behaviour Analysis - Madam")
except Exception as e:
    st.error(f"An error occurred while loading the logo: {e}")
    st.title("🗓️ Review Behaviour Analysis - Madam")

# --- Retrieve Processed Data from Session State ---
if 'processed_data' not in st.session_state or st.session_state.processed_data is None:
    st.error("Data not loaded. Please ensure data is available from the main page (Home.py).")
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

        def get_time_duration(hour):
            start_hour = f"{hour:02d}:00"
            end_hour = f"{(hour + 1) % 24:02d}:00"
            return f"{start_hour} - {end_hour}"

        if not hourly_sentiment_counts.empty and hourly_sentiment_counts['Total'].sum() > 0:
            # Peak Duration for Total Reviews
            peak_total_hour = hourly_sentiment_counts.loc[
                hourly_sentiment_counts['Total'].idxmax(), 'Hour']
            peak_duration = get_time_duration(peak_total_hour)
            kpi_peak_duration_total_str = f"{peak_duration}"

            # Peak Duration for Positive Reviews
            if hourly_sentiment_counts['positive'].max() > 0:
                peak_positive_hour = hourly_sentiment_counts.loc[
                    hourly_sentiment_counts['positive'].idxmax(), 'Hour']
                peak_positive_duration = get_time_duration(peak_positive_hour)
                kpi_peak_duration_positive_str = f"<span style='color:#82ff95'>{peak_positive_duration}</span>"
            else:
                kpi_peak_duration_positive_str = "No Positive Reviews"

            # Peak Duration for Negative Reviews
            if hourly_sentiment_counts['negative'].max() > 0:
                peak_negative_hour = hourly_sentiment_counts.loc[
                    hourly_sentiment_counts['negative'].idxmax(), 'Hour']
                peak_negative_duration = get_time_duration(peak_negative_hour)
                kpi_peak_duration_negative_str = f"<span style='color:#ff384f'>{peak_negative_duration}</span>"
            else:
                kpi_peak_duration_negative_str = "No Negative Reviews"
        else:
            kpi_peak_duration_total_str = "No Data"
            kpi_peak_duration_positive_str = "No Data"
            kpi_peak_duration_negative_str = "No Data"

        # Display KPIs
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        with kpi_col1:
            st.markdown(create_styled_metric("Peak Review Duration", kpi_peak_duration_total_str, background_color="#510f30", text_color="white"), unsafe_allow_html=True)
        with kpi_col2:
            st.markdown(create_styled_metric("Peak Positive Review Duration", kpi_peak_duration_positive_str, background_color="#510f30", text_color="white"), unsafe_allow_html=True)
        with kpi_col3:
            st.markdown(create_styled_metric("Peak Negative Review Duration", kpi_peak_duration_negative_str, background_color="#510f30", text_color="white"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Stacked Bar Chart ---
        fig_hourly = go.Figure()
        sentiment_colors = {
            'positive': '#7e8e65',
            'neutral': '#e8dfce',
            'negative': '#b65149'
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
            xaxis=dict(tickmode='linear', tick0=0, dtick=1)
        )
        fig_hourly.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig_hourly, use_container_width=True)

        # --- Clock-Shaped Radial Bar Charts ---
        st.markdown("**Clock-view radial chart**")
        # st.markdown("Visualizing review counts per hour for each sentiment in a clock-like radial chart.")

        # Prepare data for radial charts
        hours = list(range(24))
        radial_data = hourly_sentiment_counts.set_index('Hour')
        radial_data = radial_data.reindex(hours, fill_value=0).reset_index()

        # Create three columns for the radial charts
        col_rad1, col_rad2, col_rad3 = st.columns(3)

        # Function to create a minimalistic radial bar chart for a sentiment
        # def create_radial_chart(sentiment, color, title, column):
        def create_radial_chart(sentiment, color, column):
            fig = go.Figure()
            fig.add_trace(go.Barpolar(
                r=radial_data[sentiment],
                theta=[(h / 24) * 360 for h in radial_data['Hour']],
                marker_color=color,
                marker_line_width=0,  # Remove black outline
                opacity=0.7,  # Softer look
                name=sentiment.capitalize()
            ))
            fig.update_layout(
                # title=title,
                font=dict(family="Arial", size=12),
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, radial_data[sentiment].max() * 1.2] if radial_data[sentiment].max() > 0 else [0, 1],
                        showticklabels=True,
                        ticks='',
                        showgrid=False,  # Remove radial grid lines
                        showline=False
                    ),
                    angularaxis=dict(
                        tickmode='array',
                        tickvals=[(h / 24) * 360 for h in range(24)],
                        ticktext=[f"{h:02d}" for h in range(24)],  # Minimal hour labels
                        rotation=90,
                        direction="clockwise",
                        showgrid=False,
                        showline=False
                    ),
                    bgcolor='rgba(0,0,0,0)'  # Transparent background
                ),
                showlegend=False,
                height=350,  # Slightly smaller for minimalism
                margin=dict(l=30, r=30, t=80, b=30),  # Reduced margins
                paper_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
                plot_bgcolor='rgba(0,0,0,0)'
            )
            with column:
                st.plotly_chart(fig, use_container_width=True)

        # Create radial charts for each sentiment
        create_radial_chart('positive', '#7e8e65', col_rad1)
        create_radial_chart('neutral', '#e8dfce', col_rad2)
        create_radial_chart('negative', '#b65149', col_rad3)
        # create_radial_chart('positive', '#7e8e65', 'Positive Reviews', col_rad1)
        # create_radial_chart('neutral', '#e8dfce', 'Neutral Reviews', col_rad2)
        # create_radial_chart('negative', '#b65149', 'Negative Reviews', col_rad3)

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
            '1': '#e6e1e4',
            '2': '#c7adc3',
            '3': '#9e6791',
            '4': '#823871',
            '5': '#610c4d'
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
            yaxis=dict(range=[0, 100])
        )
        st.plotly_chart(fig_daytype, use_container_width=True)

        # --- Sentiment Pie Charts ---
        st.markdown("##### Sentiment Distribution")
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

        sentiment_colors = {
            'positive': '#7e8e65',
            'neutral': '#e8dfce',
            'negative': '#b65149'
        }
        col1, col2 = st.columns(2)

        with col1:
            if 'Weekday' in daytype_sentiment_counts['DayType'].values:
                weekday_data = daytype_sentiment_counts[daytype_sentiment_counts['DayType'] == 'Weekday']
                labels = ['Positive', 'Neutral', 'Negative']
                values = [weekday_data['Positive_Percentage'].iloc[0], 
                          weekday_data['Neutral_Percentage'].iloc[0], 
                          weekday_data['Negative_Percentage'].iloc[0]]
                total_reviews = int(weekday_data['Total'].iloc[0])
                fig_weekday = go.Figure(data=[
                    go.Pie(
                        labels=labels,
                        values=values,
                        marker_colors=[sentiment_colors['positive'], sentiment_colors['neutral'], sentiment_colors['negative']],
                        textinfo='percent+label',
                        hoverinfo='label+percent+value',
                        hole=0.3
                    )
                ])
                fig_weekday.update_layout(
                    title=f"Weekday Sentiment ({total_reviews} reviews)",
                    showlegend=True,
                    height=400
                )
                st.plotly_chart(fig_weekday, use_container_width=True)
            else:
                st.info("No weekday reviews in the selected period.")

        with col2:
            if 'Weekend' in daytype_sentiment_counts['DayType'].values:
                weekend_data = daytype_sentiment_counts[daytype_sentiment_counts['DayType'] == 'Weekend']
                labels = ['Positive', 'Neutral', 'Negative']
                values = [weekend_data['Positive_Percentage'].iloc[0], 
                          weekend_data['Neutral_Percentage'].iloc[0], 
                          weekend_data['Negative_Percentage'].iloc[0]]
                total_reviews = int(weekend_data['Total'].iloc[0])
                fig_weekend = go.Figure(data=[
                    go.Pie(
                        labels=labels,
                        values=values,
                        marker_colors=[sentiment_colors['positive'], sentiment_colors['neutral'], sentiment_colors['negative']],
                        textinfo='percent+label',
                        hoverinfo='label+percent+value',
                        hole=0.3
                    )
                ])
                fig_weekend.update_layout(
                    title=f"Weekend Sentiment ({total_reviews} reviews)",
                    showlegend=True,
                    height=400
                )
                st.plotly_chart(fig_weekend, use_container_width=True)
            else:
                st.info("No weekend reviews in the selected period.")

    except Exception as e:
        st.error(f"Error generating weekday vs. weekend trends: {e}")
else:
    st.info("No data available for weekday vs. weekend trends due to missing 'Time', 'label', or 'Rating' column or empty filtered data.")

st.markdown("---")