# pages/7_Appendix.py

import streamlit as st
import pandas as pd
from PIL import Image # Import Image
from pathlib import Path # Import Path
import os # Import os for path handling if needed elsewhere (though Path is preferred)

def app():
    # --- Page Configuration ---
    # Define the base directory for this script
    BASE_DIR = Path(__file__).resolve().parent

    # Path to lookout_logo_01.png (assuming it's in the parent directory of 'pages')
    LOOKOUT_LOGO_PATH = BASE_DIR.parent / "lookout_logo_01.png"

    page_icon_appendix = "📖" # Default emoji icon for this page
    try:
        img_icon_appendix = Image.open(LOOKOUT_LOGO_PATH)
        page_icon_appendix = img_icon_appendix
    except Exception as e:
        st.warning(f"Cannot use '{LOOKOUT_LOGO_PATH.name}' as page icon for Appendix: {e}. Using default emoji icon.")

    st.set_page_config(page_title="Appendix: Language-wise Rating Distribution", page_icon=page_icon_appendix, layout="wide")


    st.title("☺︎ Appendix: Language-wise Rating Distribution (Full Dataset)")

    # Access the full processed data from session state
    processed_data = st.session_state.get('processed_data')

    if processed_data is None or processed_data.empty:
        st.warning("No data available. Please go back to the 'Home' page to load the data.")
        return

    st.markdown("""
        This appendix provides a detailed breakdown of review ratings by language for **all available reviews** in the dataset.
    """)
    st.markdown("<br>", unsafe_allow_html=True)

    # Ensure 'Rating' and 'Language' columns exist
    if 'Rating' not in processed_data.columns or 'Language' not in processed_data.columns:
        st.error("Required columns 'Rating' or 'Language' are missing from the dataset.")
        return

    # Calculate review counts per language and rating
    language_rating_counts = processed_data.groupby(['Language', 'Rating']).size().unstack(fill_value=0)
    total_reviews_per_language = processed_data.groupby('Language').size()

    # Prepare data for the table
    appendix_data = []
    
    # Get unique languages, ordered by most frequent
    language_order = total_reviews_per_language.sort_values(ascending=False).index

    serial_number = 1
    for lang in language_order:
        # 将列名从 'Serial Number' 改为 'No.'
        row = {'No.': serial_number, 'Language': lang} 
        total_lang_reviews = total_reviews_per_language.get(lang, 0)
        
        # Add Total Reviews column
        row['Total Reviews'] = str(total_lang_reviews) # Convert to string for consistent alignment

        for rating in range(5, 0, -1): # Iterate from 5 down to 1
            rating_count = language_rating_counts.loc[lang, rating] if rating in language_rating_counts.columns else 0
            percentage = (rating_count / total_lang_reviews * 100) if total_lang_reviews > 0 else 0.0

            # Format numbers and percentages as strings
            row[f'{rating}-star Reviews'] = str(rating_count)
            row[f'% of {rating}-star Reviews'] = f"{percentage:.2f}%"
        
        appendix_data.append(row)
        serial_number += 1

    appendix_df = pd.DataFrame(appendix_data)

    if not appendix_df.empty:
        # Convert 'No.' to string type before setting as index
        appendix_df['No.'] = appendix_df['No.'].astype(str)

        # Define columns that should be centered (all except 'No.' and 'Language')
        columns_to_center = [col for col in appendix_df.columns if col not in ['No.', 'Language']]

        # Apply styling:
        # 1. Set 'No.' as index.
        # 2. Center align all columns in 'columns_to_center'.
        # 3. Left align the 'Language' column.
        # 4. Left align the index (No.).
        styled_df = appendix_df.set_index('No.').style \
            .set_properties(**{'text-align': 'center'}, subset=columns_to_center) \
            .set_properties(**{'text-align': 'left'}, subset=['Language']) \
            .set_properties(**{'text-align': 'left'}, axis=0, props='text-align: left;') # Style for the index

        st.dataframe(styled_df, use_container_width=True)
    else:
        st.info("No language data found in the dataset.")

# Run the app function when this file is executed
if __name__ == "__main__":
    app()