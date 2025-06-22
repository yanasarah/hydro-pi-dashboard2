import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from streamlit_option_menu import option_menu
from datetime import datetime

# Set Streamlit page configuration
st.set_page_config(page_title="Hydro-Pi Smart Dashboard", layout="wide")

# ============= STYLING ==============
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins&display=swap" rel="stylesheet">
    <style>
        .stApp {
            background-color: #f5fff5;
            color: #006400 !important;
            font-family: 'Poppins', sans-serif;
        }
        [data-testid="stSidebar"] {
            background-color: #c0ebc0 !important;
            min-width: 200px !important;
        }
        .stMetric label {
            color: #006400 !important;
        }
        .css-1dp5vir {
            background-color: #c0ebc0 !important;
        }
    </style>
""", unsafe_allow_html=True)

# ============= NAVIGATION =============
with st.sidebar:
    selected = option_menu(
        menu_title="🌿 Hydro-Pi Dashboard",
        options=["Home", "About Us", "Historical Data", "Environment Monitor", "Growth Consistency", "Insights", "Contact"],
        icons=["house", "info-circle", "clock-history", "bar-chart", "activity", "lightbulb", "envelope"],
        menu_icon="cast",
        default_index=0
    )

# ============= DATA LOADING FUNCTIONS =============
@st.cache_data
def load_main_data():
    try:
        df = pd.read_excel("summary data.xlsx", sheet_name="summary data")
        df['Week'] = df['Week'].ffill()
        df['DateTime'] = pd.to_datetime(df['Day'].astype(str)) + pd.to_timedelta(df['Time'])
        return df
    except:
        return pd.DataFrame()
@st.cache_data

def load_weekly():
    try:
        return pd.read_excel("summary data.xlsx", sheet_name="weekly trend ")
    except Exception as e:
        st.error(f"Error loading weekly trend data: {e}")
        return pd.DataFrame()



@st.cache_data
def load_daily():
    try:
        return pd.read_excel("summary data.xlsx", sheet_name="Average Daily")
    except:
        return pd.DataFrame()

# ============= HOME PAGE =============
# ============= HOME PAGE =============
if selected == "Home":
    st.markdown("""
    <div style="padding: 2rem; background: linear-gradient(135deg, #a8e6cf, #dcedc1); 
                border-radius: 20px; box-shadow: 0 8px 20px rgba(0,0,0,0.15); text-align: center;">
        <h1 style="color: #2e7d32; font-size: 3rem;">🌱 Hydro-Pi Smart Farming</h1>
        <p style="color: #388e3c; font-size: 1.2rem; margin-top: -10px;">Smarter Growth. Greener Future. 🌿</p>
    </div>
    <br>
    """, unsafe_allow_html=True)

    st.markdown("### 🌿 Quick Overview", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])  # Left (2x size), Right (1x size)

    with col1:
        st.markdown("""
        <div style="background-color: #f0fff4; border-left: 6px solid #66bb6a; 
                    padding: 1.5rem; border-radius: 15px; box-shadow: 0 3px 10px rgba(0,0,0,0.1); 
                    font-size: 1rem;">
            🌼 <strong>Did you know?</strong><br>
            <em>"A garden is a friend you can visit anytime. Start your journey to smarter farming today!"</em>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Current Plant box
        st.markdown("""
        <div style="background-color: #ffffff; border: 2px solid #c8e6c9; 
                    border-radius: 15px; padding: 1.2rem; text-align: center; 
                    box-shadow: 0 3px 10px rgba(0,0,0,0.1);">
            <h4 style="margin-top: 0; color: #388e3c;">🌿 Current Plant</h4>
            <p style="font-weight: bold; font-size: 1.1rem; color: #2e7d32;">🥬 Spinach</p>
            <img src="https://www.pngmart.com/files/13/Spinach-PNG-Transparent-Image.png" 
                 alt="Spinach" width="90" style="margin-top: 5px;">
        </div>
        """, unsafe_allow_html=True)

        # Small Date box under the plant card
        st.markdown(f"""
        <div style="margin-top: 1rem; background: #e8f5e9; padding: 1rem; border-radius: 12px; 
                    text-align: center; box-shadow: 0 3px 10px rgba(0,0,0,0.1);">
            <h5 style="color: #2e7d32; margin-bottom: 5px;">📅 Today</h5>
            <p style="font-size: 16px; font-weight: bold; color: #1b5e20;">
                {datetime.now().strftime("%A, %d %B %Y")}
            </p>
        </div>
        """, unsafe_allow_html=True)



# ========= historical data==================== 
elif selected == "Historical Data": 
    st.markdown(""" 
    <style> 
        html, body, [class*="st-"] { color: #006400 !important; } 
        h1, h2, h3, h4, h5, h6 { color: #2e8b57 !important; } 
        .stMetric label, .stMetric div { color: #006400 !important; } 
        .dataframe td, .dataframe th { color: #006400 !important; } 
        label, .stTextInput, .stSelectbox, .stRadio, .stSlider, .stFileUploader { color: #006400 !important; } 
        .stAlert, .stSuccess, .stWarning { color: #006400 !important; } 
    </style> 
    """, unsafe_allow_html=True) 

    st.markdown("<h1 style='color:#2e8b57;'>        Historical Data Analysis</h1>", unsafe_allow_html=True) 

    data_source = st.radio("Select data source:", 
                           ["Use built-in dataset", "Upload your own Excel file"], 
                           horizontal=True) 

    if data_source == "Upload your own Excel file":
        uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file, sheet_name="Sheet1")
                df['Week'] = df['Week'].ffill()
                st.success("File uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.stop()
        else:
            st.info("Please upload an Excel file or switch to built-in dataset")
            st.stop()
    else:
        df = load_main_data()
        if df.empty:
            st.error("Built-in data not available. Please upload a file instead.")
            st.stop()
        st.success("Using built-in Hydro-Pi dataset")

    # ===== FILTERING =====
    st.subheader("        Filter Data")
    filter_col1, filter_col2 = st.columns(2)

    if 'Week' in df.columns:
        selected_week = filter_col1.selectbox("Select Week", df['Week'].dropna().unique())
    else:
        selected_week = None

    if 'Day' in df.columns and selected_week is not None:
        available_days = df[df['Week'] == selected_week]['Day'].unique()
        selected_day = filter_col2.selectbox("Select Day", available_days)
    else:
        selected_day = None

    if selected_week is not None and selected_day is not None:
        filtered_df = df[(df['Week'] == selected_week) & (df['Day'] == selected_day)]
    else:
        filtered_df = df.copy()

    # ===== SUMMARY =====
    st.subheader("           Data Overview") 
    col1, col2, col3 = st.columns(3) 
    col1.metric("Total Records", len(df)) 
    col2.metric("Days Recorded", df['Day'].nunique() if 'Day' in df.columns else "N/A") 
    col3.metric("Weeks Recorded", df['Week'].nunique() if 'Week' in df.columns else "N/A") 

    # ===== WEEKLY STATISTICS =====
    st.subheader("Weekly Summary (Mean and Standard Deviation)")

    if 'Week' in df.columns:
        stat_table = df.groupby("Week")[['TDS', 'pH', 'DHT22 1', 'HUM 1', 'DHT 22 2', 'HUM 2', 'DS18B20']].agg(['mean', 'std'])
        st.dataframe(stat_table.style.format("{:.2f}"), height=350, use_container_width=True)
    else:
        st.warning("Week column not found — unable to compute weekly summary.")




    # ===== CORRELATION ANALYSIS =====
    st.subheader("🔗 Parameter Correlations")

    # Select only numeric columns and exclude datetime
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) >= 2:
        try:
            # Create correlation matrix
            corr_matrix = filtered_df[numeric_cols].corr()
            
            # Create figure with larger size
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Customize heatmap appearance
            sns.heatmap(
                corr_matrix, 
                annot=True, 
                cmap="YlGnBu", 
                ax=ax,
                annot_kws={"size": 10, "color": "black"},  # Darker annotation text
                linewidths=.5
            )
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            
            # Ensure tight layout to prevent cutoff
            plt.tight_layout()
            
            # Display in Streamlit
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error generating correlation heatmap: {e}")
    else:
        st.warning(f"Need at least 2 numeric columns for correlation. Found: {numeric_cols}")

    # ===== GROWTH SCORE MODEL =====
    st.subheader("🌿 Plant Health Analysis")
    
    if st.checkbox("Calculate Growth Score", True, help="Calculate plant health score based on environmental factors"):
        # Check required columns exist
        required_cols = ['DS18B20', 'HUM 1', 'TDS', 'pH']
        if all(col in filtered_df.columns for col in required_cols):
            filtered_df['Growth_Score'] = (
                0.3 * filtered_df['DS18B20'] + 
                0.2 * (100 - filtered_df['HUM 1']) + 
                0.25 * filtered_df['TDS'] / 100 + 
                0.25 * filtered_df['pH']
            )
            filtered_df['Growth_Score'] = ((filtered_df['Growth_Score'] - filtered_df['Growth_Score'].min()) / 
                                        (filtered_df['Growth_Score'].max() - filtered_df['Growth_Score'].min())) * 100
            
            st.line_chart(filtered_df.set_index('Time' if 'Time' in filtered_df.columns else filtered_df.index)['Growth_Score'])
            
            # Machine Learning Prediction
            if st.checkbox("Show Advanced Predictions", help="Show machine learning predictions vs actual growth"):
                X = filtered_df[['pH', 'TDS', 'DS18B20', 'HUM 1']]
                y = filtered_df['Growth_Score']
                
                model = RandomForestRegressor()
                model.fit(X, y)
                predictions = model.predict(X)
                
                pred_df = pd.DataFrame({
                    'Time': filtered_df['Time'] if 'Time' in filtered_df.columns else filtered_df.index,
                    'Actual': y,
                    'Predicted': predictions
                }).set_index('Time')
                
                st.line_chart(pred_df)
                st.metric("Prediction Accuracy", 
                         f"{100 - mean_squared_error(y, predictions, squared=False):.1f}%")
        else:
            st.warning("Missing required columns for growth score calculation")

    # ===== RECOMMENDATIONS =====
    st.subheader("💡 Optimization Recommendations")
    
    if 'pH' in filtered_df.columns:
        avg_pH = filtered_df['pH'].mean()
        if avg_pH < 5.8:
            st.warning("⚠️ pH is slightly low. Consider adding pH Up solution.")
        elif avg_pH > 6.2:
            st.warning("⚠️ pH is slightly high. Consider adding pH Down solution.")
        else:
            st.success("✅ pH level is optimal")
    else:
        st.warning("pH data not available for recommendations")

    if 'TDS' in filtered_df.columns:
        avg_tds = filtered_df['TDS'].mean()
        if avg_tds < 650:
            st.warning("⚠️ Nutrient levels low. Consider adding fertilizer.")
        elif avg_tds > 750:
            st.warning("⚠️ Nutrient levels high. Consider diluting solution.")
        else:
            st.success("✅ Nutrient levels are optimal")
    else:
        st.warning("TDS data not available for nutrient recommendations")

    # ===== RAW DATA =====
    st.subheader("📋 Detailed Measurements")
    st.dataframe(filtered_df.style.background_gradient(cmap='YlGn'), 
                height=300,
                use_container_width=True)
#========ENVIROMENT PART==============
elif selected == "Environment Monitor":
    st.title("📊 Environmental Monitoring Dashboard")

    # Only show uploader if file not already stored
    if "weekly_file" not in st.session_state:
        uploaded_file = st.file_uploader("📂 Upload your summary data Excel file", type=["xlsx"])
        if uploaded_file:
            st.session_state["weekly_file"] = uploaded_file
    else:
        uploaded_file = st.session_state["weekly_file"]

    if uploaded_file:
        try:
            weekly_df = pd.read_excel(uploaded_file, sheet_name="weekly trend ")
            st.success("✅ Sheet 'weekly trend ' loaded successfully")
            st.write("Preview of Weekly Data:", weekly_df.head())

            st.subheader("📈 Weekly Sensor Trends")

            if 'Week' not in weekly_df.columns:
                st.warning("⚠️ 'Week' column not found in uploaded data.")
            else:
                weekly_df = weekly_df.set_index('Week')

                # Titles for each graph
                trend_labels = {
                    'Avg TDS': '📉 Average TDS per Week (ppm)',
                    'Avg pH': '🔬 Average pH per Week',
                    'Avg DHT22 1': '🌡️ Avg Air Temperature (Sensor 1)',
                    'Avg HUM 1': '💧 Avg Humidity (Sensor 1)',
                    'Avg DHT 22 2': '🌡️ Avg Air Temperature (Sensor 2)',
                    'Avg HUM 2': '💧 Avg Humidity (Sensor 2)',
                    'Avg DS18B20': '🌡️ Avg Water Temperature (DS18B20)'
                }

                for col, label in trend_labels.items():
                    if col in weekly_df.columns:
                        st.markdown(f"### {label}")
                        st.line_chart(weekly_df[[col]])
                    else:
                        st.warning(f"⚠️ Column '{col}' not found in uploaded data.")

        except Exception as e:
            st.error(f"❌ Error reading Excel file: {e}")
    else:
        st.info("ℹ️ Please upload your Excel file to view environmental trends.")




# ============= GROWTH CONSISTENCY PAGE =============
elif selected == "Growth Consistency":
    st.title("Growth Consistency Analysis")
    daily_df = load_daily()

    # Debug: Show the loaded data
    st.write("Loaded Daily Data:", daily_df)

    if not daily_df.empty:
        st.subheader("Daily Variation Analysis")

        available_params = [col for col in ['Avg TDS', 'Avg pH', 'Avg DHT22 1', 'Avg HUM 1', 'Avg DS18B20'] if col in daily_df.columns]

        parameters = st.multiselect(
            "Select parameters to analyze",
            available_params,
            default=['Avg TDS', 'Avg pH'] if 'Avg TDS' in daily_df.columns and 'Avg pH' in daily_df.columns else available_params[:2]
        )

        if parameters:
            # Coefficient of variation
            cv_data = daily_df[parameters].std() / daily_df[parameters].mean()

            st.subheader("Consistency Metrics (Coefficient of Variation)")
            st.bar_chart(cv_data)

            st.info("""
            **Interpretation:**
            - Lower values = more consistent
            - Higher values = more variable
            - Ideal: Below 0.1 (10% variation)
            """)

            # Time series with rolling average
            if 'Day' in daily_df.columns:
                st.subheader("7-Day Moving Average")
                rolling_df = daily_df.set_index('Day')[parameters].rolling(7).mean()
                st.line_chart(rolling_df)
            else:
                st.warning("'Day' column not found for moving average.")

            # Inconsistency alerts
            st.subheader("Inconsistency Alerts")
            for param in parameters:
                std_dev = daily_df[param].std()
                mean_val = daily_df[param].mean()
                if mean_val != 0 and std_dev > (mean_val * 0.15):
                    st.warning(f"High variability in {param} (Std Dev: {std_dev:.2f})")
    else:
        st.warning("No daily data available. Please check your data source.")


# ============= OTHER PAGES =============
elif selected == "About Us":
    st.title("About Hydroponic Systems")
    # ... (keep your existing About Us content)

elif selected == "Insights":
    st.title("💡 Insights & Recommendations")
    st.info("Advanced insights coming in next update!")

elif selected == "Contact":
    st.title("📞 Contact Us")
    st.markdown("""
        **Hydro-Pi Team**  
        📧 Email: support@hydro-pi.local  
        🌍 Website: [www.hydro-pi.local](#)
    """)

if __name__ == "__main__":
    pass
