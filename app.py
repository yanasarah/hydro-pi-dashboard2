import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Hydro-Pi Smart Farming Dashboard", layout="wide")

st.title("🌱 Hydro-Pi Smart Farming Dashboard")
st.markdown("Monitor your smart hydroponic system in real-time.")

uploaded_file = st.file_uploader("Upload your sensor CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)

    # Time range slider
    min_time = df.index.min()
    max_time = df.index.max()
    time_range = st.slider("Select Time Range", min_value=min_time, max_value=max_time, value=(min_time, max_time))
    df = df.loc[time_range[0]:time_range[1]]

    # Sensor tabs
    sensor_tabs = st.tabs(df.columns)
    for sensor, tab in zip(df.columns, sensor_tabs):
        with tab:
            st.subheader(f"{sensor} Over Time")
            fig = px.line(df, y=sensor, title=f"{sensor} Over Time")
            st.plotly_chart(fig, use_container_width=True)

            # Warning indicators
            if sensor == "pH":
                if df[sensor].max() > 7.5:
                    st.warning("⚠️ High pH level detected!")
                elif df[sensor].min() < 5.5:
                    st.warning("⚠️ Low pH level detected!")
else:
    st.info("Please upload a CSV file to view the dashboard.")
