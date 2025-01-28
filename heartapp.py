import pandas as pd
import joblib
import streamlit as st
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scalerheart.pkl")
st.title("Heart Disease Detection App")
st.sidebar.header("Enter Patient's Health Details")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
cp = st.sidebar.selectbox("Chest Pain Type", ("Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"))
trestbps = st.sidebar.number_input("Resting Blood Pressure (in mm Hg)", min_value=50, max_value=200, value=120)
chol = st.sidebar.number_input("Serum Cholesterol (in mg/dl)", min_value=100, max_value=500, value=250)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ("True", "False"))
restecg = st.sidebar.selectbox("Resting Electrocardiographic Results", ("Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"))
thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
exang = st.sidebar.selectbox("Exercise Induced Angina", ("Yes", "No"))
oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise Relative to Rest", min_value=0.0, max_value=10.0, value=1.0)
slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", ("Upsloping", "Flat", "Downsloping"))
ca = st.sidebar.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=4, value=0)
thal = st.sidebar.selectbox("Thalassemia", ("Normal", "Fixed Defect", "Reversible Defect"))
sex = 1 if sex == "Male" else 0
cp_mapping = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
cp = cp_mapping[cp]
fbs = 1 if fbs == "True" else 0
restecg_mapping = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
restecg = restecg_mapping[restecg]
exang = 1 if exang == "Yes" else 0
slope_mapping = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
slope = slope_mapping[slope]
thal_mapping = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
thal = thal_mapping[thal]
input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                          columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"])
if st.sidebar.button("Submit"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.write("### The patient is at **high risk** of heart disease.")
    else:
        st.write("### The patient is at **low risk** of heart disease.")
