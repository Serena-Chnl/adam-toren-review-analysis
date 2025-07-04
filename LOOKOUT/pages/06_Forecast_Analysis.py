# pages/06_Forecast.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PIL import Image # Import Image
from pathlib import Path # Import Path
import numpy as np
from prophet import Prophet
import holidays
from scipy.stats import chi2_contingency
import datetime
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

page_icon_forecast_analysis = "🔮" # Default emoji icon for this page
try:
    img_icon_forecast_analysis = Image.open(LOOKOUT_LOGO_PATH)
    page_icon_forecast_analysis = img_icon_forecast_analysis
except Exception as e:
    st.warning(f"Cannot use '{LOOKOUT_LOGO_PATH.name}' as page icon for Forecast Analysis: {e}. Using default emoji icon.")

st.set_page_config(page_title="Forecast Analysis - A'DAM LOOKOUT", page_icon=page_icon_forecast_analysis, layout="wide")

# --- Logo and Title Section ---
col_title, col_spacer, col_logo = st.columns([0.65, 0.05, 0.3])
with col_title:
    st.title("Forecast Analysis")
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
st.sidebar.header("Prediction Filters")
filtered_data = all_data
start_date_pred = None
end_date_pred = None

if 'Time' in all_data.columns and pd.api.types.is_datetime64_any_dtype(all_data['Time']) and not all_data['Time'].isna().all():
    min_year = all_data['Time'].dt.year.min() if not all_data['Time'].empty else 2020
    min_date = pd.to_datetime(f"{min_year}-01-01").date()
    max_date = all_data['Time'].max().date()
    default_start = min_date
    default_end = max_date
    if default_start > default_end:
        default_start = default_end
    start_date_pred = st.sidebar.date_input(
        "Start date (for model training)",
        default_start,
        min_value=min_date,
        max_value=max_date,
        key="prediction_start_date"
    )
    end_date_pred = st.sidebar.date_input(
        "End date (for model training)",
        default_end,
        min_value=min_date,
        max_value=max_date,
        key="prediction_end_date"
    )
    if start_date_pred > end_date_pred:
        st.sidebar.error("Error: Start date cannot be after end date.")
        filtered_data = pd.DataFrame(columns=all_data.columns)
    else:
        start_datetime = pd.to_datetime(start_date_pred)
        end_datetime = pd.to_datetime(end_date_pred) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        filtered_data = all_data[
            (all_data['Time'] >= start_datetime) &
            (all_data['Time'] <= end_datetime)
        ]
else:
    st.sidebar.warning("Date filter cannot be applied: 'Time' column issue.")

# --- Color Customization ---
chart_background_color = "#f8f9fa"
neutral_bar_color = "#e8dfce"
historical_line_color = "#5a5a5a"
predicted_line_color = "#c12028"
anomaly_marker_color = "#ea1b28"

# --- Date Range Display ---
st.markdown("<br>", unsafe_allow_html=True)
if start_date_pred and end_date_pred and start_date_pred <= end_date_pred:
    st.markdown(f"Using data from **{start_date_pred.strftime('%Y-%m-%d')}** to **{end_date_pred.strftime('%Y-%m-%d')}** for predictions.")
else:
    st.markdown("Using **Full Dataset** for predictions.")

# --- Predictive KPIs ---
st.subheader("▸ Predictive Insights")
st.markdown("<br>", unsafe_allow_html=True)

predicted_total_reviews_str = "N/A"
predicted_positive_prop_str = "N/A"
predicted_5_star_day_str = "N/A"
predicted_1_star_day_str = "N/A"

if not filtered_data.empty:
    # Prepare daily review data
    daily_reviews = filtered_data.set_index('Time').resample('D').size().reset_index(name='y')
    daily_reviews.rename(columns={'Time': 'ds'}, inplace=True)
    daily_reviews['y'] = daily_reviews['y'].fillna(0)  # Fill missing days with 0
    daily_reviews = daily_reviews[daily_reviews['ds'].notna()]  # Remove invalid dates

    # --- Prophet Forecast for Review Volume ---
    if len(daily_reviews) >= 28 and daily_reviews['y'].sum() > 0:  # Require 28 days and non-zero reviews
        try:
            # Add Dutch holidays
            nl_holidays = holidays.Netherlands(years=[2024, 2025])
            holidays_df = pd.DataFrame({
                'holiday': 'Dutch Holiday',
                'ds': pd.to_datetime(list(nl_holidays.keys())),
                'lower_window': -1,
                'upper_window': 1
            })

            model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True, holidays=holidays_df)
            model.fit(daily_reviews)
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)

            predicted_total_reviews = forecast[forecast['ds'] > daily_reviews['ds'].max()]['yhat'].sum()
            predicted_total_reviews_str = f"{int(predicted_total_reviews)} reviews"

            # Sentiment Forecast (Positive Proportion)
            daily_sentiment = filtered_data.groupby([filtered_data['Time'].dt.date, 'label']).size().unstack(fill_value=0)
            daily_sentiment['total'] = daily_sentiment.sum(axis=1)
            daily_sentiment['positive_prop'] = daily_sentiment.get('positive', 0) / daily_sentiment['total'].replace(0, np.nan)
            daily_sentiment = daily_sentiment.reset_index()[['Time', 'positive_prop']].rename(columns={'Time': 'ds', 'positive_prop': 'y'})
            daily_sentiment['ds'] = pd.to_datetime(daily_sentiment['ds'])
            daily_sentiment = daily_sentiment[daily_sentiment['y'].notna()]  # Remove NaN proportions

            if len(daily_sentiment) >= 28 and daily_sentiment['y'].sum() > 0:
                sentiment_model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True, holidays=holidays_df)
                sentiment_model.fit(daily_sentiment)
                sentiment_future = sentiment_model.make_future_dataframe(periods=30)
                sentiment_forecast = sentiment_model.predict(sentiment_future)
                predicted_positive_prop = sentiment_forecast[sentiment_forecast['ds'] > daily_sentiment['ds'].max()]['yhat'].mean()
                predicted_positive_prop_str = f"{predicted_positive_prop:.2%}"
            else:
                predicted_positive_prop_str = "Insufficient sentiment data"
        except Exception as e:
            st.warning(f"Prophet model failed: {e}")
            predicted_total_reviews_str = "Model Error"
            predicted_positive_prop_str = "Model Error"

    # --- Day of Week Prediction ---
    if 'Rating' in filtered_data.columns:
        day_of_week_data = filtered_data.copy()
        day_of_week_data['DayOfWeek'] = day_of_week_data['Time'].dt.day_name()
        five_star_reviews = day_of_week_data[day_of_week_data['Rating'] == 5]
        one_star_reviews = day_of_week_data[day_of_week_data['Rating'] == 1]

        if not five_star_reviews.empty:
            day_counts = five_star_reviews['DayOfWeek'].value_counts()
            predicted_5_star_day_str = f"<span style='color:#66c2ff'>{day_counts.idxmax()}</span>"
        if not one_star_reviews.empty:
            day_counts = one_star_reviews['DayOfWeek'].value_counts()
            predicted_1_star_day_str = f"<span style='color:#ff6666'>{day_counts.idxmax()}</span>"

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
with kpi_col1:
    st.markdown(create_styled_metric("Predicted Total Reviews (Next 30 Days)", predicted_total_reviews_str), unsafe_allow_html=True)
with kpi_col2:
    st.markdown(create_styled_metric("Predicted Positive Review Proportion", predicted_positive_prop_str), unsafe_allow_html=True)
with kpi_col3:
    st.markdown(create_styled_metric("Most Likely Day for 5-Star Review", predicted_5_star_day_str), unsafe_allow_html=True)
with kpi_col4:
    st.markdown(create_styled_metric("Most Likely Day for 1-Star Review", predicted_1_star_day_str), unsafe_allow_html=True)

st.markdown("---")

# --- Review Volume Forecast Chart ---
st.subheader("▸ 30-Day Review Volume Forecast with High Volume Days")
if not filtered_data.empty:
    # Prepare daily review data
    daily_reviews = filtered_data.set_index('Time').resample('D').size().reset_index(name='y')
    daily_reviews.rename(columns={'Time': 'ds'}, inplace=True)
    daily_reviews['y'] = daily_reviews['y'].fillna(0)  # Fill missing days with 0
    daily_reviews = daily_reviews[daily_reviews['ds'].notna()]  # Remove invalid dates

    # Calculate threshold for high volume days (75% of max)
    max_daily_reviews = daily_reviews['y'].max()
    review_threshold = max_daily_reviews * 0.75  # 75% of maximum

    # Detect high volume days
    anomalies = daily_reviews[daily_reviews['y'] > review_threshold]

    # Prophet Forecast for Review Volume
    forecast = None
    if len(daily_reviews) >= 28 and daily_reviews['y'].sum() > 0:  # Require 28 days and non-zero reviews
        try:
            # Add Dutch holidays
            nl_holidays = holidays.Netherlands(years=[2024, 2025])
            holidays_df = pd.DataFrame({
                'holiday': 'Dutch Holiday',
                'ds': pd.to_datetime(list(nl_holidays.keys())),
                'lower_window': -1,
                'upper_window': 1
            })

            model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True, holidays=holidays_df)
            model.fit(daily_reviews)
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
        except Exception as e:
            st.warning(f"Prophet model failed: {e}")

    # Create combined plot
    fig = go.Figure()
    # Historical reviews
    fig.add_trace(go.Scatter(
        x=daily_reviews['ds'],
        y=daily_reviews['y'],
        mode='lines',
        name='Historical Reviews',
        line=dict(color=historical_line_color)
    ))
    # High volume days
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies['ds'],
            y=anomalies['y'],
            mode='markers',
            name='High Volume Days',
            marker=dict(color=anomaly_marker_color, size=10)
        ))
    # Forecasted data
    if forecast is not None:
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name='Predicted Reviews',
            line=dict(color=predicted_line_color, dash='dash')
        ))
        # Confidence interval
        # Replace the confidence interval section in the existing 07_Forecast.py
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            fill=None,
            mode='lines',
            line_color='rgba(200, 200, 200, 0.3)',  # Light gray for upper CI line
            name='95% CI Upper'
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(200, 200, 200, 0.3)',  # Light gray for lower CI line and fill
            name='95% CI Lower'
        ))

    fig.update_layout(
        title="Historical Reviews, High Volume Days, and 30-Day Forecast",
        xaxis_title="Date",
        yaxis_title="Number of Reviews",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor=chart_background_color,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig, use_container_width=True)

    # Summary text
    if not anomalies.empty:
        st.markdown(f"**High Volume Days Detected**: Days with more than {review_threshold:.1f} reviews (75% of maximum {max_daily_reviews:.1f}). Investigate these dates for events or campaigns.")
    else:
        st.info(f"No days with more than {review_threshold:.1f} reviews (75% of maximum {max_daily_reviews:.1f}) detected.")
    if forecast is None:
        st.info("Not enough data to generate a 30-day forecast. Ensure at least 28 days of data.")
else:
    st.info("Not enough data to generate forecast or detect high volume days. Ensure sufficient data.")

st.markdown("---")

# --- Day of Week Sentiment Analysis ---
st.subheader("▸ Likelihood of Review Ratings by Day of the Week")
if not filtered_data.empty and 'day_of_week_data' in locals():
    rating_day_counts = day_of_week_data.groupby(['DayOfWeek', 'Rating']).size().unstack(fill_value=0)
    rating_day_percentage = rating_day_counts.apply(lambda x: (x / x.sum()) * 100, axis=1)
    
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    rating_day_percentage = rating_day_percentage.reindex(days_order)

    # Perform chi-square test for statistical significance
    chi2_stat, p_val, _, _ = chi2_contingency(rating_day_counts)
    if p_val < 0.05:
        st.markdown(f"**Note**: Differences in rating distributions across days are statistically significant (p-value: {p_val:.4f}).")
    else:
        st.markdown(f"**Note**: No significant differences in rating distributions across days (p-value: {p_val:.4f}).")

    fig = go.Figure()
    # Update 
    rating_color_map = {
        1: "#d8575d",  # Negative (matches negative_bar_color, deep red)
        2: "#e39b9a",  # Soft coral for a gentle step from negative
        3: "#e8dfce",  # Neutral (matches neutral_bar_color, warm beige)
        4: "#b8cee3",  # Pale blue for a smooth shift toward positive
        5: "#a3bbce"   # Positive (matches positive_bar_color, muted blue)
    }
    for rating in sorted(rating_day_percentage.columns):
        fig.add_trace(go.Bar(
            y=rating_day_percentage.index,
            x=rating_day_percentage[rating],
            name=f'{rating} Star',
            orientation='h',
            marker_color=rating_color_map.get(rating),
            text=rating_day_percentage[rating].round(1).astype(str) + '%',
            textposition='inside'
        ))
    
    fig.update_layout(
        title="Percentage of Each Rating by Day of the Week",
        xaxis_title="Percentage of Reviews (%)",
        yaxis_title="Day of the Week",
        barmode='stack',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor=chart_background_color,
        legend_title_text='Rating',
        xaxis_range=[0, 100]
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **How to read this chart:** This chart shows the composition of ratings for each day of the week. For example, a longer blue bar on Friday means a higher percentage of reviews received on Fridays are 5-star, compared to other ratings on that same day.
    """)
else:
    st.info("Not enough data to analyze rating distribution by day of the week.")

st.markdown("---")