import streamlit as st
from streamlit_option_menu import option_menu

# Sidebar with clickable menu
with st.sidebar:
    selected = option_menu(
        menu_title="MAIN MENU",  # Title for the sidebar menu
        options=["home", "project", "contact"],  # Menu options
        icons=["house", "gear", "envelope"],     # Optional icons (you can remove this line)
        menu_icon="cast",                        # Optional top icon
        default_index=0                          # Default selected index
    )

# Page content changes based on menu selection
if selected == "home":
    st.title("🌱 Welcome to Hydro-Pi Smart Farming Dashboard")

elif selected == "project":
    st.title("🔧 This is the Project Page")

elif selected == "contact":
    st.title("📞 This is the Contact Page")
