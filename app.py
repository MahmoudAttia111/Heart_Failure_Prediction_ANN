import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model


model = load_model("heart_failure_model.keras")
scaler = joblib.load("scaler.pkl")
le_sex = joblib.load("le_sex.pkl")
le_chest = joblib.load("le_chest.pkl")
le_ecg = joblib.load("le_resting_ecg.pkl")
le_exercise = joblib.load("le_exercise.pkl")
le_slope = joblib.load("le_st_slope.pkl")

st.title("💓 Heart Failure Prediction App")
st.write("أدخل بياناتك الطبية للتنبؤ بخطر فشل القلب")


age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("RestingBP", min_value=50, max_value=200, value=120)
cholesterol = st.number_input("Cholesterol", min_value=50, max_value=400, value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
resting_ecg = st.selectbox("RestingECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])
oldpeak = st.number_input("Oldpeak", min_value=-2.0, max_value=6.0, value=1.0, step=0.1)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])


sex_val = le_sex.transform([sex])[0]
chest_val = le_chest.transform([chest_pain])[0]
ecg_val = le_ecg.transform([resting_ecg])[0]
exercise_val = le_exercise.transform([exercise_angina])[0]
slope_val = le_slope.transform([st_slope])[0]


input_data = np.array([[
    age,
    sex_val,
    chest_val,
    resting_bp,
    cholesterol,
    fasting_bs,
    ecg_val,
    max_hr,
    exercise_val,
    oldpeak,
    slope_val
]])


input_data[:, [0, 3, 4, 7, 9]] = scaler.transform(input_data[:, [0, 3, 4, 7, 9]])


if st.button("Predict"):
    prediction = model.predict(input_data)
    result = (prediction > 0.5).astype(int)[0][0]

    if result == 1:
        st.error("⚠️ احتمالية عالية لفشل القلب")
    else:
        st.success("✅ لا توجد خطورة كبيرة")
