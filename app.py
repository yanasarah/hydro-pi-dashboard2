
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from streamlit_option_menu import option_menu
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from fpdf import FPDF
import base64

# Set Streamlit page configuration
st.set_page_config(page_title="Hydro-Pi Smart Dashboard", layout="wide")

# Define target users info banner for Home page
TARGET_USER_BANNER = """
### 👤 Who is this for?
- 🧑‍🌾 **Urban Home Gardeners** – grow fresh vegetables even in small spaces.
- 👩‍💼 **Busy Workers** – automate and monitor your plants while you're at work.
- 👪 **Families & Hobbyists** – a fun and educational system for all ages.
"""

# Update menu with beginner-friendly terms
with st.sidebar:
    selected = option_menu(
        menu_title="🌿 Hydro-Pi Dashboard",
        options=[
            "Home", "About Us", "📈 Nutrient & Environment Stats", "🌡️ Real-Time Monitor",
            "📊 Growth Score & Consistency", "💡 Smart Tips & Forecast", "🔍 Crop Comparison Tool",
            "🤖 Beginner Advisor + FAQ", "📬 Contact Us"
        ],
        icons=["house", "info-circle", "bar-chart", "thermometer", "activity", "lightbulb",
               "search", "robot", "envelope"],
        menu_icon="cast",
        default_index=0
    )

# Add onboarding message
if "onboard_shown" not in st.session_state:
    st.session_state.onboard_shown = True
    st.info("👋 Welcome! This dashboard is designed for non-technical users. Start on the Home page to learn more.")

# Load remaining unchanged code...
# (Your existing functions and page-specific logic remains as is)

# Inject target user banner into Home page
if selected == "Home":
    st.markdown(TARGET_USER_BANNER)

# Optional: Add Quick Start toggle
if selected == "Home":
    if st.checkbox("🧭 Show Quick Start Guide"):
        st.markdown("""
        **Step 1**: Go to 📈 Nutrient Stats to see past trends  
        **Step 2**: Use 🌡️ Real-Time Monitor to check today’s values  
        **Step 3**: Tap 💡 Smart Tips to get AI suggestions  
        **Step 4**: Use 🔍 Crop Comparison to evaluate past batches  
        """)

# You can continue with your existing Home page layout below...
# Any time you use technical terms like "TDS", consider tooltips or inline help.

