import streamlit as st
import numpy as np
import joblib
from tensorflow import keras
from keras.models import load_model

# -------------------------------
# Load model & scaler
# -------------------------------
model = load_model("model/driving_model.keras")
scaler = joblib.load("model/scaler.pkl")

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Driving Behavior Detection",
    layout="centered"
)

st.title("🚗 Driving Behavior Classification")
st.write(
    "This application predicts whether driving behavior is **normal** or **unsafe** "
    "based on vehicle telematics features."
)

st.divider()

# -------------------------------
# Input fields
# -------------------------------
avg_speed = st.number_input("Average Speed (km/h)", 0.0, 150.0, 40.0)
speed_std = st.number_input("Speed Variability", 0.0, 50.0, 5.0)
max_speed = st.number_input("Maximum Speed (km/h)", 0.0, 200.0, 80.0)
avg_acc = st.number_input("Average Acceleration (m/s²)", -5.0, 5.0, 0.1)
throttle_var = st.number_input("Throttle Variance", 0.0, 50.0, 3.0)
rpm_mean = st.number_input("Average RPM", 0.0, 8000.0, 2000.0)
idle_ratio = st.slider("Idle Time Ratio", 0.0, 1.0, 0.1)
hard_brake = st.number_input("Hard Brake Count", 0, 20, 0)
hard_accel = st.number_input("Hard Acceleration Count", 0, 20, 0)

st.caption(
    "⚠️ Model uses a **lower decision threshold (0.3)** to improve detection of unsafe driving."
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Driving Behavior"):

    X = np.array([[  
        avg_speed,
        speed_std,
        max_speed,
        avg_acc,
        throttle_var,
        rpm_mean,
        idle_ratio,
        hard_brake,
        hard_accel
    ]])

    X_scaled = scaler.transform(X)
    prob = model.predict(X_scaled)[0][0]

    threshold = 0.3
    prediction = int(prob >= threshold)

    st.subheader("Result")

    st.progress(float(min(prob, 1.0)))

    if prediction == 1:
        st.error(f"⚠️ Unsafe Driving Detected\n\nRisk Probability: **{prob:.2f}**")
    else:
        st.success(f"✅ Normal Driving\n\nRisk Probability: **{prob:.2f}**")
