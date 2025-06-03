import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Sidebar with clickable menu
with st.sidebar:
    selected = option_menu(
        menu_title="MAIN MENU",
        options=["home", "project", "contact"],
        icons=["house", "gear", "envelope"],
        menu_icon="cast",
        default_index=0
    )

# ====================== HOME PAGE ======================
if selected == "home":
    st.title("🌱 Welcome to Hydro-Pi Smart Farming Dashboard")
    st.write("Upload your hydroponic sensor data to predict plant growth trends.")

    uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

    if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Raw Uploaded Data")
    st.dataframe(df)

    if 'plant_growth' not in df.columns:
        st.error("Your CSV must include a column named 'plant_growth' for prediction.")
    else:
        # Drop target and select only numeric features
        X = df.drop(columns=['plant_growth'])
        X = X.select_dtypes(include=['float64', 'int64'])
        y = df['plant_growth']

        # Preprocessing: impute + scale
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Train model
        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        # Predict and evaluate
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        st.subheader("📈 Predicted vs Actual Plant Growth")
        results = pd.DataFrame({'Actual': y_test.values, 'Predicted': predictions})
        st.dataframe(results)

        fig = px.line(results, y=['Actual', 'Predicted'], title="Growth Trend")
        st.plotly_chart(fig)

        st.markdown(f"✅ **R² Score:** `{r2:.2f}`")
        st.markdown(f"📉 **Mean Squared Error:** `{mse:.2f}`")


# ====================== PROJECT PAGE ======================
elif selected == "project":
    st.title("🔧 Project: Sensor Data Charts")

    uploaded_file = st.file_uploader("📂 Upload CSV file to view sensor charts", type=["csv"], key="project")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Sensor Charts")

        for column in df.columns:
            if df[column].dtype in ['int64', 'float64'] and column != "plant_growth":
                st.write(f"### {column}")
                fig = px.line(df, y=column, title=f"{column} over Time")
                st.plotly_chart(fig, use_container_width=True)

# ====================== CONTACT PAGE ======================
elif selected == "contact":
    st.title("📞 Contact")
    st.markdown("For more information, please contact us at:")
    st.markdown("- 📧 Email: hydro-pi@smartfarming.com")
    st.markdown("- 📱 Phone: +123 456 7890")
    st.markdown("- 🌐 Website: [Hydro-Pi](https://yourwebsite.com)")
