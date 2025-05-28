import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Hydro-Pi Smart Farming", layout="wide")

st.title("🌿 Hydro-Pi Smart Plant Dashboard")

st.markdown("Monitor your hydroponic environment in real time. Upload CSV data to see trends in temperature, pH, TDS, light, and more.")

# Upload section
uploaded_file = st.file_uploader("📤 Upload your sensor CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.strip() for col in df.columns]  # Clean column names

    time_col = df.columns[0]
    sensor_cols = df.columns[1:]

    # Dashboard Tabs
    tabs = st.tabs(sensor_cols)

    for i, sensor in enumerate(sensor_cols):
        with tabs[i]:
            st.subheader(f"📈 {sensor} over Time")

            fig = px.line(df, x=time_col, y=sensor, markers=True,
                          labels={time_col: "Timestamp", sensor: sensor},
                          title=f"{sensor} Trend")

            fig.update_layout(height=400, margin=dict(l=30, r=30, t=30, b=30))
            st.plotly_chart(fig, use_container_width=True)

            st.write("**Stats:**")
            st.metric("Average", round(df[sensor].mean(), 2))
            st.metric("Max", round(df[sensor].max(), 2))
            st.metric("Min", round(df[sensor].min(), 2))

else:
    st.info("Upload a CSV file to begin. Example format: Time, Temperature, pH, TDS, Light, WaterLevel")
