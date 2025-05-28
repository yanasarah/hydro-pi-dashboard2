import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Hydro-Pi Dashboard", layout="wide")

st.title("🌿 Hydro-Pi Smart Plant Dashboard")

uploaded_file = st.file_uploader("📤 Upload your sensor CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Data Preview")
    st.dataframe(df, use_container_width=True)

    time_column = df.columns[0]
    sensor_options = df.columns[1:]

    sensor = st.selectbox("📊 Choose a sensor to visualize", sensor_options)

    fig = px.line(df, x=time_column, y=sensor, markers=True,
                  title=f"{sensor} Over Time",
                  labels={time_column: "Timestamp", sensor: sensor})

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please upload a CSV file to begin.")
