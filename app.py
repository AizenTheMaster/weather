# weather_app.py 🌤️ Weather Predictor (Fixed Input Feature Count)

import streamlit as st
import numpy as np
import pickle

# 🔄 Load the saved model and encoders
with open("weather_model_fresh.sav", "rb") as file:
    model_data = pickle.load(file)

model = model_data['model']
label_encoders = model_data['label_encoders']

# ✅ Label encoder for Season (only 1 encoded input)
season_classes = label_encoders['Season'].classes_
weathertype_classes = label_encoders['WeatherType'].classes_

# 🌐 Streamlit Page Setup
st.set_page_config(page_title="Weather Type Predictor", page_icon="🌦️", layout="centered")
st.title("🌦️ Weather Type Prediction App")

st.markdown("Enter weather details below and predict the likely **Weather Type** (e.g. Sunny, Rainy, Cloudy).")

# 🧾 Input Fields (9 numeric + 1 encoded)
col1, col2 = st.columns(2)
with col1:
    temperature = st.number_input("🌡️ Temperature (°C)")
    humidity = st.number_input("💧 Humidity (%)")
    pressure = st.number_input("📈 Atmospheric Pressure (hPa)")
    uv_index = st.number_input("🔆 UV Index")
    visibility = st.number_input("👁️ Visibility (km)")
with col2:
    dew_point = st.number_input("🧊 Dew Point (°C)")
    wind_speed = st.number_input("💨 Wind Speed (km/h)")
    cloud_cover = st.number_input("☁️ Cloud Cover (%)")
    wind_direction = st.number_input("🧭 Wind Direction (°)")
    season = st.selectbox("📅 Season", season_classes)

# 🔄 Encode categorical input (Season only)
season_encoded = label_encoders['Season'].transform([season])[0]

# 🧠 Final input array (MUST match training order: 9 numeric + 1 encoded)
input_data = np.array([[temperature, humidity, pressure, uv_index,
                        visibility, dew_point, wind_speed, cloud_cover,
                        wind_direction, season_encoded]])

# Debug info (optional)
# st.write("🔎 Model expects:", model.n_features_in_, "features")
# st.write("📥 Provided:", input_data.shape[1])

# 🔍 Predict Weather Type
if st.button("🔍 Predict Weather"):
    prediction_encoded = model.predict(input_data)[0]
    prediction_label = label_encoders['WeatherType'].inverse_transform([prediction_encoded])[0]
    st.success(f"🌤️ Predicted Weather Type: **{prediction_label}**")
