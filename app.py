import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Set Streamlit page config
st.set_page_config(page_title="Hydro-Pi Smart Dashboard", layout="wide")

# Sidebar navigation menu
with st.sidebar:
    selected = option_menu(
        menu_title="🌿 Hydro-Pi Dashboard",  # Sidebar title
        options=["Home", "Environment Monitor", "Growth Consistency", "Insights", "Contact"],
        icons=["house", "bar-chart", "activity", "lightbulb", "envelope"],
        menu_icon="cast",
        default_index=0
    )

