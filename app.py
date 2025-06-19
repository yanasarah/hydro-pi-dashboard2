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
        return pd.read_excel("summary data.xlsx", sheet_name="weekly trend")
    except:
        return pd.DataFrame()

@st.cache_data
def load_daily():
    try:
        return pd.read_excel("summary data.xlsx", sheet_name="Average Daily")
    except:
        return pd.DataFrame()

# ============= HOME PAGE =============
if selected == "Home":
    st.markdown("""
    <div style="padding: 2rem; background: linear-gradient(to right, #bdfcc9, #e0ffe0); 
                border-radius: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); text-align: center;">
        <h1 style="color: #2e8b57;">🌱 Welcome to Hydro-Pi Smart Farming</h1>
        <p style="color: #4d774e; font-size: 18px;">Monitor. Predict. Grow smarter 🌿</p>
    </div>
    <br>""", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
            <div style='background-color: #e6ffe6; border-left: 5px solid #66bb66; 
                        padding: 1rem; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                🌿 <em>"Grow your health, grow a garden."</em>
            </div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background-color: #ffffff; border: 2px solid #cceccc; 
                        border-radius: 12px; padding: 1rem; text-align: center;'>
                <h4 style='margin-top: 0;'>Your Plant:</h4>
                <p style='font-weight: bold; color: #2e7d32;'>🥬 Spinach</p>
                <img src="https://www.pngmart.com/files/13/Spinach-PNG-Transparent-Image.png" 
                     alt="Spinach" width="100">
            </div>""", unsafe_allow_html=True)

    st.markdown(f"""
        <br>
        <div style="background-color: #e0f5e9; padding: 1.5rem; border-radius: 15px; 
                    text-align: center; box-shadow: 2px 2px 8px rgba(0,0,0,0.1);">
            <h4 style="color: #1e4620;">📅 Today</h4>
            <p style="font-size: 20px; color: #1e4620;">{datetime.now().strftime("%A, %d %B %Y")}</p>
        </div>
    """, unsafe_allow_html=True)
#========= historical data====================

elif selected == "Historical Data":
    # Custom CSS for consistent dark green text
       st.markdown("""
    <style>
        /* Global text color (including sidebar, headers, markdown, etc.) */
        html, body, [class*="st-"] {
            color: #006400 !important;
        }

        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #2e8b57 !important;
        }

        /* Streamlit metric label and value */
        .stMetric label, .stMetric div {
            color: #006400 !important;
        }

        /* Dataframe text */
        .dataframe td, .dataframe th {
            color: #006400 !important;
        }

        /* Input components (labels and text) */
        label, .stTextInput, .stSelectbox, .stRadio, .stSlider, .stFileUploader {
            color: #006400 !important;
        }

        /* Alerts */
        .stAlert, .stSuccess, .stWarning {
            color: #006400 !important;
        }

        /* Color text in matplotlib/seaborn plots (axes, ticks) if used */
        .css-1cpxqw2, .css-1offfwp {
            color: #006400 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='color:#2e8b57;'>🌱 Historical Data Analysis</h1>", unsafe_allow_html=True)

    # ===== DATA SOURCE SELECTION =====
    data_source = st.radio("Select data source:",
                           ["Use built-in dataset", "Upload your own Excel file"],
                           horizontal=True)
    
    if data_source == "Upload your own Excel file":
        uploaded_file = st.file_uploader("📤 Upload Excel File", type=["xlsx"])
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file)
                st.success("✅ File uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.stop()
        else:
            st.info("ℹ️ Please upload an Excel file or switch to built-in dataset")
            st.stop()
    else:
        # Use the built-in dataset
        df = load_main_data()
        if df.empty:
            st.error("Built-in data not available. Please upload a file instead.")
            st.stop()
        st.success("✅ Using built-in Hydro-Pi dataset")

    # ===== DATA EXPLORATION =====
    st.subheader("🔍 Data Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df), help="Total number of data records")
    col2.metric("Days Recorded", df['Day'].nunique() if 'Day' in df.columns else "N/A", 
               help="Number of unique days in dataset")
    col3.metric("Weeks Recorded", df['Week'].nunique() if 'Week' in df.columns else "N/A",
               help="Number of unique weeks in dataset")

    # ===== INTERACTIVE FILTERS =====
    st.subheader("📅 Filter Data")
    
    # Create columns for filters
    filter_col1, filter_col2 = st.columns(2)
    
    # Week filter
    if 'Week' in df.columns:
        selected_week = filter_col1.selectbox("Select Week", df['Week'].unique(),
                                            help="Filter data by specific week")
    else:
        selected_week = None
    
    # Day filter
    if 'Day' in df.columns and selected_week is not None:
        available_days = df[df['Week'] == selected_week]['Day'].unique()
        selected_day = filter_col2.selectbox("Select Day", available_days,
                                           help="Filter data by specific day")
    else:
        selected_day = None
    
    # Apply filters
    if selected_week is not None and selected_day is not None:
        filtered_df = df[(df['Week'] == selected_week) & (df['Day'] == selected_day)]
    else:
        filtered_df = df.copy()
        st.warning("⚠️ Some filter columns not found - showing all data")

    # ===== KEY METRICS =====
    st.subheader("📊 Daily Summary Metrics")
    metric_cols = st.columns(4)
    
    # pH metric
    if 'pH' in filtered_df.columns:
        metric_cols[0].metric("Avg pH", f"{filtered_df['pH'].mean():.2f}",
                             help="Average pH level (ideal: 5.8-6.2)")
    else:
        metric_cols[0].metric("Avg pH", "N/A")
    
    # TDS metric
    if 'TDS' in filtered_df.columns:
        metric_cols[1].metric("Avg TDS", f"{filtered_df['TDS'].mean():.1f} ppm",
                             help="Average Total Dissolved Solids (ideal: 650-750 ppm)")
    else:
        metric_cols[1].metric("Avg TDS", "N/A")
    
    # Temperature metric
    if 'DS18B20' in filtered_df.columns:
        metric_cols[2].metric("Avg Temp", f"{filtered_df['DS18B20'].mean():.1f}°C",
                             help="Average water temperature")
    else:
        metric_cols[2].metric("Avg Temp", "N/A")
    
    # Humidity metric
    if 'HUM 1' in filtered_df.columns:
        metric_cols[3].metric("Avg Humidity", f"{filtered_df['HUM 1'].mean():.1f}%",
                             help="Average humidity level")
    else:
        metric_cols[3].metric("Avg Humidity", "N/A")

    # ===== VISUALIZATIONS =====
    st.subheader("📈 Environmental Trends")
    
    # Prepare chart data
    chart_data = filtered_df.set_index('Time' if 'Time' in filtered_df.columns else filtered_df.index)
    columns_to_plot = []
    
    for col in ['pH', 'TDS', 'DS18B20', 'HUM 1']:
        if col in filtered_df.columns:
            columns_to_plot.append(col)
    
    if columns_to_plot:
        st.line_chart(chart_data[columns_to_plot])
    else:
        st.warning("No compatible data columns found for visualization")

    # ===== CORRELATION ANALYSIS =====
    st.subheader("🔗 Parameter Correlations")
    
    # Select only numeric columns
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) >= 2:
        corr_matrix = filtered_df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", ax=ax,
                   annot_kws={"color": "#006400"})  # Dark green correlation numbers
        st.pyplot(fig)
    else:
        st.warning("Need at least 2 numeric columns for correlation analysis")

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

# ============= ENVIRONMENT MONITOR PAGE =============
elif selected == "Environment Monitor":
    st.title("📊 Environmental Monitoring Dashboard")
    weekly_df = load_weekly()
    
    if not weekly_df.empty:
        selected_week = st.selectbox("Select Week", weekly_df['Week'].unique())
        week_data = weekly_df[weekly_df['Week'] == selected_week].iloc[0]
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Weekly Avg TDS", f"{week_data['Avg TDS']:.1f} ppm")
        col2.metric("Weekly Avg pH", f"{week_data['Avg pH']:.2f}")
        col3.metric("Avg Water Temp", f"{week_data['Avg DS18B20']:.1f}°C")
        
        # Weekly trends
        st.subheader("Weekly Trends Over Time")
        fig, ax = plt.subplots(figsize=(10, 6))
        weekly_df.set_index('Week')[['Avg TDS', 'Avg pH', 'Avg DS18B20']].plot(ax=ax)
        st.pyplot(fig)
        
        # Environmental stability
        st.subheader("Environmental Stability")
        stability_data = weekly_df.set_index('Week')[['Avg TDS', 'Avg pH']].std().reset_index()
        st.bar_chart(stability_data.set_index('index'))

# ============= GROWTH CONSISTENCY PAGE =============
elif selected == "Growth Consistency":
    st.title("🌱 Growth Consistency Analysis")
    daily_df = load_daily()
    
    if not daily_df.empty:
        st.subheader("Daily Variation Analysis")
        parameters = st.multiselect(
            "Select parameters to analyze",
            ['Avg TDS', 'Avg pH', 'Avg DHT22 1', 'Avg HUM 1', 'Avg DS18B20'],
            default=['Avg TDS', 'Avg pH']
        )
        
        if parameters:
            # Calculate coefficient of variation
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
            st.subheader("7-Day Moving Average")
            rolling_df = daily_df.set_index('Day')[parameters].rolling(7).mean()
            st.line_chart(rolling_df)
            
            # Highlight inconsistencies
            st.subheader("⚠️ Inconsistency Alerts")
            for param in parameters:
                std_dev = daily_df[param].std()
                if std_dev > (daily_df[param].mean() * 0.15):
                    st.warning(f"High variability in {param} (Std Dev: {std_dev:.2f})")

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
