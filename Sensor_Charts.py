import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sensor Charts", layout="wide")
st.title("📈 Sensor Charts")

uploaded_file = st.file_uploader("Upload your CSV sensor data", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["Timestamp"])
    st.dataframe(df)

    with st.expander("📊 Show Charts"):
        for column in df.columns:
            if column != "Timestamp":
                st.subheader(column)
                fig, ax = plt.subplots()
                ax.plot(df["Timestamp"], df[column])
                ax.set_xlabel("Timestamp")
                ax.set_ylabel(column)
                st.pyplot(fig)
else:
    st.info("Upload a CSV file to see sensor charts.")
