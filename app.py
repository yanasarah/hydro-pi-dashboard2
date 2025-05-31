import streamlit as st
from streamlit_option_menu import option_menu

with st.sidebar:
selected = option_menu(
  menu_title="MAIN MENU",
  options=["home","project","contact"],
)
if selected == "home":
  st.title(f"🌱 Welcome to Hydro-Pi Smart Farming Dashboard { selected }")
if selected == "project":
  st.title(f"🌱 this is project { selected }")
  if selected == "contact":
  st.title(f"🌱this is contact { selected }")
  

