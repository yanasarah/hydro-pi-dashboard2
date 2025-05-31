import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd

# Sidebar with clickable menu
with st.sidebar:
    selected = option_menu(
        menu_title="MAIN MENU",  # Title for the sidebar menu
        options=["home", "project", "contact"],  # Menu options
        icons=["house", "gear", "envelope"],     # Optional icons
        menu_icon="cast",                        # Optional top icon
        default_index=0                          # Default selected index
    )

# Page content changes based on menu selection
if selected == "home":
    st.title("🌱 Welcome to Hydro-Pi Smart Farming Dashboard")
    st.write("Upload your sensor CSV file below to begin analysis:")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            st.write("Preview of uploaded data:")
            st.dataframe(df)
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

elif selected == "project":
    st.title("🔧 This is the Project Page")

elif selected == "contact":
    st.title("📞 This is the Contact Page")
