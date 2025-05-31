import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Hydro-Pi Dashboard", layout="wide")

st.title("🌱 Hydro-Pi Smart Farming Dashboard")

uploaded_file = st.file_uploader("📤 Upload your CSV sensor data", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)
