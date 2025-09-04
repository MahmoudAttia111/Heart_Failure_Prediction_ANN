import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# تحميل الموديل والـ scaler
model = load_model("heart_failure_model.keras")
scaler = joblib.load("scaler.pkl")

st.title("💓 Heart Failure Prediction App")
st.write("أدخل بياناتك الطبية للتنبؤ بخطر فشل القلب")

# --- مدخلات المستخدم ---
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("RestingBP", min_value=50, max_value=200, value=120)
cholesterol = st.number_input("Cholesterol", min_value=50, max_value=400, value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])  # في الداتا أصلاً 0/1
resting_ecg = st.selectbox("RestingECG", ["Normal", "ST", "LVH"])
max_hr = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])
oldpeak = st.number_input("Oldpeak", min_value=-2.0, max_value=6.0, value=1.0, step=0.1)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# --- تحويل نفس الـ encoding اللي عملته في التدريب ---
sex_map = {"M": 1, "F": 0}
chest_map = {"ATA": 0, "NAP": 1, "TA": 2, "ASY": 3}
resting_ecg_map = {"LVH": 0, "Normal": 1, "ST": 2}
exercise_map = {"N": 0, "Y": 1}
slope_map = {"Down": 0, "Flat": 1, "Up": 2}

# تجهيز بيانات الإدخال
input_data = np.array([[
    age,
    sex_map[sex],
    chest_map[chest_pain],
    resting_bp,
    cholesterol,
    fasting_bs,
    resting_ecg_map[resting_ecg],
    max_hr,
    exercise_map[exercise_angina],
    oldpeak,
    slope_map[st_slope]
]])

# تطبيق الـ scaler على الأعمدة الرقمية (نفس الأعمدة اللي دربته عليها)
input_data[:, [0, 3, 4, 7, 9]] = scaler.transform(input_data[:, [0, 3, 4, 7, 9]])

# زر التنبؤ
if st.button("Predict"):
    prediction = model.predict(input_data)
    result = (prediction > 0.5).astype(int)[0][0]

    if result == 1:
        st.error("⚠️ احتمالية عالية لفشل القلب")
    else:
        st.success("✅ لا توجد خطورة كبيرة")
