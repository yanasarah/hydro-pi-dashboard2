import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px

# Global state to store uploaded CSV
if "csv_data" not in st.session_state:
    st.session_state.csv_data = None

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="MAIN MENU",
        options=["home", "project", "contact"],
        icons=["house", "gear", "envelope"],
        menu_icon="cast",
        default_index=0
    )

# Home page: CSV uploader
if selected == "home":
    st.title("🌱 Welcome to Hydro-Pi Smart Farming Dashboard")
    st.write("Upload your sensor CSV file below to begin analysis:")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.csv_data = df  # Save to session state
            st.success("✅ File uploaded successfully!")
            st.write("Preview of uploaded data:")
            st.dataframe(df)
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

# Project page: show charts
elif selected == "project":
    st.title("📊 Sensor Data Charts")

    if st.session_state.csv_data is not None:
        df = st.session_state.csv_data

        # Optional: convert timestamp column to datetime if available
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # For each sensor column, plot a chart if it exists
        sensor_columns = ['LDR', 'pH', 'TDS', 'Temperature', 'Distance']

        for sensor in sensor_columns:
            if sensor in df.columns:
                st.subheader(f"{sensor} Reading")
                fig = px.line(df, x='timestamp' if 'timestamp' in df.columns else df.index,
                              y=sensor, markers=True, title=f"{sensor} Over Time")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"⚠️ '{sensor}' data not found in the uploaded CSV.")
    else:
        st.warning("⚠️ Please upload a CSV file first on the **Home** page.")

# Contact page
elif selected == "contact":
    st.title("📞 This is the Contact Page")
