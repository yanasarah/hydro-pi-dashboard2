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

# ============= HISTORICAL DATA PAGE =============
elif selected == "Historical Data":
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
    col1.metric("Total Records", len(df))
    col2.metric("Days Recorded", df['Day'].nunique() if 'Day' in df.columns else "N/A")
    col3.metric("Weeks Recorded", df['Week'].nunique() if 'Week' in df.columns else "N/A")

    # ===== INTERACTIVE FILTERS =====
    st.subheader("📅 Filter Data")
    
    # Create columns for filters
    filter_col1, filter_col2 = st.columns(2)
    
    # Week filter (only if column exists)
    if 'Week' in df.columns:
        selected_week = filter_col1.selectbox("Select Week", df['Week'].unique())
    else:
        selected_week = None
    
    # Day filter (only if column exists)
    if 'Day' in df.columns and selected_week is not None:
        available_days = df[df['Week'] == selected_week]['Day'].unique()
        selected_day = filter_col2.selectbox("Select Day", available_days)
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
        metric_cols[0].metric("Avg pH", f"{filtered_df['pH'].mean():.2f}")
    else:
        metric_cols[0].metric("Avg pH", "N/A")
    
    # TDS metric
    if 'TDS' in filtered_df.columns:
        metric_cols[1].metric("Avg TDS", f"{filtered_df['TDS'].mean():.1f} ppm")
    else:
        metric_cols[1].metric("Avg TDS", "N/A")
    
    # Temperature metric
    if 'DS18B20' in filtered_df.columns:
        metric_cols[2].metric("Avg Temp", f"{filtered_df['DS18B20'].mean():.1f}°C")
    else:
        metric_cols[2].metric("Avg Temp", "N/A")
    
    # Humidity metric
    if 'HUM 1' in filtered_df.columns:
        metric_cols[3].metric("Avg Humidity", f"{filtered_df['HUM 1'].mean():.1f}%")
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
        sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Need at least 2 numeric columns for correlation analysis")

    # ===== GROWTH SCORE MODEL =====
    st.subheader("🌿 Plant Health Analysis")
    
    if st.checkbox("Calculate Growth Score", True):
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
            if st.checkbox("Show Advanced Predictions"):
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
   elif selected == "About Us":
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h1 style="color: #2e8b57; font-family: Poppins; font-size: 2rem; margin-bottom: 0.1rem;">
            About Hydroponic Systems
        </h1>
        <h2 style="color: #3a6b35; font-family: Poppins; font-size: 2.9rem; margin-top: 0;">
            At the forefront of innovation
        </h2>
    </div>
    """, unsafe_allow_html=True)

    # Two columns for explanation and image
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <p style="color: #3a4f41; font-size: 18px; line-height: 1.6; text-align: justify;">
            Hydroponics is a sustainable method of cultivating plants without soil by using nutrient-rich water solutions. 
            This technique promotes faster plant growth, higher yields, and more efficient use of space and resources. 
            It’s particularly beneficial in environments where arable land is limited, allowing year-round food production and water conservation.
        </p>
        """, unsafe_allow_html=True)

    with col2:
        st.image("Untitled-design-2.jpg", caption="Hydroponic Farming System", use_container_width=True)
       

    # 2nd section: "Why it matters" with image on the left
    col_img, spacer, col_txt = st.columns([2, 0.5, 3], vertical_alignment="top")
    with col_img:
        st.image(
            "Hydro-tower2.png",
            caption="Benefits of Hydroponics",
            use_container_width=True
        )
    with col_txt:
        st.markdown("""
            <p style="color: #3a4f41; font-size: 17px; line-height: 1.5; text-align: justify;">
                <strong>Why it matters:</strong><br>
                With climate change, urbanization, and rising food demands, hydroponics offers a smart solution.
                It uses up to 90% less water than traditional farming and can be set up virtually anywhere — from rooftops to indoor facilities.
                It brings food production closer to consumers and helps reduce the carbon footprint.
            </p>
        """, unsafe_allow_html=True)

    # How it works section
    st.markdown("""<hr style="margin-top: 2rem; margin-bottom: 1.5rem;">""", unsafe_allow_html=True)

    st.markdown("""
    <div style="padding: 1rem 3rem;">
        <h3 style="color: #2e8b57; font-family: Poppins; font-size: 1.8rem; margin-bottom: 0.5rem;">
            How Our System Works for You
        </h3>
        <p style="color: #3a4f41; font-size: 17px; line-height: 1.6; text-align: justify;">
            Our hydroponic system is fully automated and beginner-friendly. Sensors monitor water quality, temperature, and light — ensuring optimal plant health at all times. 
            Customers can easily check the status of their crops through our mobile app or web dashboard.
        </p>
        <p style="color: #3a4f41; font-size: 17px; line-height: 1.6; text-align: justify;">
            Whether you're a home gardener, a school project team, or a commercial grower, our smart system scales with your needs. 
            You get real-time updates, AI-powered growth predictions, and tips — all to make sure your plants thrive without the guesswork.
        </p>
        <p style="color: #2e8b57; font-size: 17px; font-weight: 600; text-align: justify;">
            Experience the future of farming — sustainable, smart, and surprisingly simple.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Vision & Mission Section
    st.markdown("""<hr style="margin-top: 2rem; margin-bottom: 2rem;">""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
    <div style="background-color: #cce6cc; border-radius: 12px; padding: 1.5rem; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);">
        <h3 style="color: black; font-family: Poppins; text-align: center;">🌐 Vision</h3>
        <p style="color: black; font-size: 17px; line-height: 1.6; text-align: justify;">
            To revolutionize agriculture through smart hydroponic technologies, making sustainable and efficient food production accessible to all, regardless of location or experience.
        </p>
    </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
    <div style="background-color: #cce6cc; border-radius: 12px; padding: 1.5rem; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);">
        <h3 style="color: black; font-family: Poppins; text-align: center;">🌐 Mission</h3>
        <p style="color: black; font-size: 17px; line-height: 1.6; text-align: justify;">
            To empower communities and individuals by delivering user-friendly, data-driven hydroponic solutions that support a greener planet and a healthier future.
        </p>
    </div>
        """, unsafe_allow_html=True)
#======insight=========================

elif selected == "Insights":
    st.title("💡 Insights & Recommendations")
    st.info("Advanced insights coming in next update!")

#======contact=========================

elif selected == "Contact":
    st.title("📞 Contact Us")
    st.markdown("""
        **Hydro-Pi Team**  
        📧 Email: support@hydro-pi.local  
        🌍 Website: [www.hydro-pi.local](#)
    """)

if __name__ == "__main__":
    pass
