import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Hydro-Pi Smart Farming Dashboard", layout="wide")

st.title("🌱 Hydro-Pi Smart Farming Dashboard")
st.markdown("""
Monitor your smart hydroponic system in real-time.  
Upload sensor data to see insights, warnings, and trends.
""")

# File uploader
uploaded_file = st.file_uploader("📤 Upload your sensor CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.strip() for col in df.columns]

    time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col])

    sensor_cols = df.columns[1:]

    # Time Range Slider
    min_time = df[time_col].min()
    max_time = df[time_col].max()
    time_range = st.slider("⏱️ Select Time Range", min_value=min_time, max_value=max_time,
                           value=(min_time, max_time), format="MM/DD/YY - HH:mm")

    filtered_df = df[(df[time_col] >= time_range[0]) & (df[time_col] <= time_range[1])]

    # Warning messages
    st.subheader("⚠️ Sensor Warnings")
    warning_zone = st.container()

    if "pH" in sensor_cols:
        ph_data = pd.to_numeric(filtered_df["pH"], errors="coerce")
        if ph_data.max() > 7.5:
            warning_zone.warning("⚠️ High pH level detected (> 7.5)")
        elif ph_data.min() < 5.5:
            warning_zone.warning("⚠️ Low pH level detected (< 5.5)")

    if "Water Level (cm)" in sensor_cols:
        water_data = pd.to_numeric(filtered_df["Water Level (cm)"], errors="coerce")
        if water_data.min() < 8:
            warning_zone.warning("⚠️ Low Water Level detected (< 8 cm)")

    # Charts for each sensor
    st.subheader("📊 Sensor Data Trends")
    for sensor in sensor_cols:
        numeric_data = pd.to_numeric(filtered_df[sensor], errors='coerce')
        fig = px.line(filtered_df, x=time_col, y=numeric_data, markers=True,
                      title=f"{sensor} Trend", labels={time_col: "Timestamp", sensor: sensor})
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.write(f"**Stats for {sensor}:**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg", f"{numeric_data.mean():.2f}")
        col2.metric("Max", f"{numeric_data.max():.2f}")
        col3.metric("Min", f"{numeric_data.min():.2f}")
        st.markdown("---")
else:
    st.info("Please upload a CSV file. You can use the [sample CSV](sandbox:/mnt/data/hydro_pi_sample_data.csv) for testing.")
