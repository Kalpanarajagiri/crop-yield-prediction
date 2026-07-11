import streamlit as st
import joblib
import numpy as np

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="centered"
)

# -------------------------------
# Load Model & Encoders
# -------------------------------
model = joblib.load("crop_yield_model.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")
season_encoder = joblib.load("season_encoder.pkl")
state_encoder = joblib.load("state_encoder.pkl")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("🌾 Crop Yield Prediction")

st.sidebar.markdown("## TFI Summer Internship Project")

st.sidebar.write("""
This application predicts crop yield using
Machine Learning.

### Input Parameters
🌾 Crop Name

🌱 Season

📍 State

📏 Land Area

🌧 Rainfall

🧪 Fertilizer

🐞 Pesticide

🌡 Temperature
""")

st.sidebar.success("Model : Random Forest Regressor")

# -------------------------------
# Title
# -------------------------------
st.title("🌾 Crop Yield Prediction Using Machine Learning")

st.write(
    "Enter the farming details below and click **Predict Yield**."
)

st.markdown("---")

# -------------------------------
# User Inputs
# -------------------------------

crop = st.selectbox(
    "🌾 Select Crop",
    crop_encoder.classes_
)

season = st.selectbox(
    "🌱 Select Season",
    season_encoder.classes_
)

state = st.selectbox(
    "📍 Select State",
    state_encoder.classes_
)

area = st.number_input(
    "📏 Land Area (Hectares)",
    min_value=0.0,
    value=1.0,
    step=0.1
)

rainfall = st.number_input(
    "🌧 Annual Rainfall (mm)",
    min_value=0.0,
    value=500.0,
    step=10.0
)

fertilizer = st.number_input(
    "🧪 Fertilizer Used (kg)",
    min_value=0.0,
    value=100.0,
    step=10.0
)

pesticide = st.number_input(
    "🐞 Pesticide Used (kg)",
    min_value=0.0,
    value=10.0,
    step=1.0
)

temperature = st.number_input(
    "🌡 Average Temperature (°C)",
    min_value=0.0,
    value=25.0,
    step=1.0
)

st.markdown("---")

# -------------------------------
# Prediction
# -------------------------------

if st.button("🔍 Predict Yield"):

    crop_value = crop_encoder.transform([crop])[0]
    season_value = season_encoder.transform([season])[0]
    state_value = state_encoder.transform([state])[0]

    sample = np.array([[
        crop_value,
        season_value,
        state_value,
        area,
        rainfall,
        fertilizer,
        pesticide,
        temperature
    ]])

    prediction = model.predict(sample)

    st.success("Prediction Completed Successfully!")

    st.markdown("## 🌾 Prediction Result")

    st.metric(
        label="Estimated Crop Yield",
        value=f"{prediction[0]:.2f}"
    )

    st.balloons()

# -------------------------------
#