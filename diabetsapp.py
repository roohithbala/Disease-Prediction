import streamlit as st
import numpy as np
import joblib
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")
st.title("Diabetes Detection App")
st.sidebar.header("Input Patient Details")
glucose = st.sidebar.number_input("Glucose Level", min_value=0, max_value=300, value=120, step=1)
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=200, value=80, step=1)
skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20, step=1)
insulin = st.sidebar.number_input("Insulin Level", min_value=0, max_value=900, value=85, step=1)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30, step=1)

input_data = np.array([[glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
if st.sidebar.button("Predict"):
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    probability = model.predict_proba(input_data_scaled)[0][1] * 100
    if prediction[0] == 1:
        st.error(f"The model predicts **diabetes** with a probability of {probability:.2f}%.")
    else:
        st.success(f"The model predicts **no diabetes** with a probability of {probability:.2f}%.")
