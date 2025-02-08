import streamlit as st
import pandas as pd
import joblib
parkinsons_model = joblib.load("parkinsons_model.pkl")
parkinsons_scaler = joblib.load("parkinsonscaler.pkl")
heart_disease_model = joblib.load("heart_disease_model.pkl")
heart_scaler = joblib.load("scalerheart.pkl")
diabetes_model = joblib.load("diabetes_model.pkl")
diabetes_scaler = joblib.load("scaler.pkl")
st.title("Disease Prediction System")
st.markdown("""
    This app predicts the likelihood of three major health conditions based on user input:
    - Parkinson's Disease
    - Heart Disease
    - Diabetes
""")
disease = st.sidebar.selectbox(
    "Select Disease to Predict",
    ["Parkinson's Disease", "Heart Disease", "Diabetes"]
)
if disease == "Parkinson's Disease":
    st.header("Parkinson's Disease Prediction")
    st.write("Enter the following health details:")
    col1, col2 = st.columns(2)
    with col1:
        MDVP_Fo_Hz = st.number_input("MDVP:Fo(Hz)", value=120.0, step=0.1)
        MDVP_Fhi_Hz = st.number_input("MDVP:Fhi(Hz)", value=150.0, step=0.1)
        MDVP_Flo_Hz = st.number_input("MDVP:Flo(Hz)", value=85.0, step=0.1)
        MDVP_Jitter = st.number_input("MDVP:Jitter(%)", value=0.005, format="%.3f")
        MDVP_Jitter_Abs = st.number_input("MDVP:Jitter(Abs)", value=0.00003, format="%.5f")
        MDVP_RAP = st.number_input("MDVP:RAP", value=0.003, format="%.3f")
        MDVP_PPQ = st.number_input("MDVP:PPQ", value=0.003, format="%.3f")
    with col2:
        Jitter_DDP = st.number_input("Jitter:DDP", value=0.010, format="%.3f")
        MDVP_Shimmer = st.number_input("MDVP:Shimmer", value=0.020, format="%.3f")
        MDVP_Shimmer_dB = st.number_input("MDVP:Shimmer(dB)", value=0.5, format="%.1f")
        Shimmer_APQ3 = st.number_input("Shimmer:APQ3", value=0.015, format="%.3f")
        Shimmer_APQ5 = st.number_input("Shimmer:APQ5", value=0.020, format="%.3f")
        MDVP_APQ = st.number_input("MDVP:APQ", value=0.025, format="%.3f")
        Shimmer_DDA = st.number_input("Shimmer:DDA", value=0.045, format="%.3f")
        NHR = st.number_input("NHR", value=0.005, format="%.3f")
    HNR = st.number_input("HNR", value=20.0, format="%.1f")
    RPDE = st.number_input("RPDE", value=0.5, format="%.3f")
    DFA = st.number_input("DFA", value=0.7, format="%.3f")
    spread1 = st.number_input("Spread1", value=-5.0, format="%.1f")
    spread2 = st.number_input("Spread2", value=0.2, format="%.1f")
    D2 = st.number_input("D2", value=2.0, format="%.1f")
    PPE = st.number_input("PPE", value=0.2, format="%.3f")
    input_features = pd.DataFrame([[MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter, MDVP_Jitter_Abs, MDVP_RAP, 
                                     MDVP_PPQ, Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5,
                                     MDVP_APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]],
                                   columns=["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", 
                                            "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", 
                                            "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", 
                                            "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", 
                                            "spread2", "D2", "PPE"])
    if st.button("Predict Parkinson's"):
        input_scaled = parkinsons_scaler.transform(input_features)
        prediction = parkinsons_model.predict(input_scaled)
        if prediction[0] == 1:
            st.error("🧠 The patient is likely to have Parkinson's Disease.")
        else:
            st.success("🌿 The patient is unlikely to have Parkinson's Disease.")
elif disease == "Heart Disease":
    st.header("Heart Disease Prediction")
    st.write("Enter the following health details:")
    age = st.number_input("Age", value=50, min_value=1, max_value=120)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure (in mm Hg)", value=120, min_value=50, max_value=200)
    chol = st.number_input("Serum Cholesterol (in mg/dl)", value=250, min_value=100, max_value=500)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
    restecg = st.selectbox("Resting Electrocardiographic Results", ["Normal", "ST-T Wave Abnormality", 
                                                                    "Left Ventricular Hypertrophy"])
    thalach = st.number_input("Maximum Heart Rate Achieved", value=150, min_value=50, max_value=250)
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest", value=1.0, min_value=0.0, max_value=10.0)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", value=0, min_value=0, max_value=4)
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
    sex = 1 if sex == "Male" else 0
    cp = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}[cp]
    fbs = 1 if fbs == "True" else 0
    restecg = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}[restecg]
    exang = 1 if exang == "Yes" else 0
    slope = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}[slope]
    thal = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}[thal]
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                              columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", 
                                       "oldpeak", "slope", "ca", "thal"])

    if st.button("Predict Heart Disease"):
        input_scaled = heart_scaler.transform(input_data)
        prediction = heart_disease_model.predict(input_scaled)
        if prediction[0] == 1:
            st.error("💔 The patient is at high risk of heart disease.")
        else:
            st.success("❤️ The patient is at low risk of heart disease.")
elif disease == "Diabetes":
    st.header("Diabetes Prediction")
    st.write("Enter the following health details:")
    glucose = st.number_input("Glucose Level", value=120, min_value=0, max_value=1000)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", value=80, min_value=0, max_value=300)
    skin_thickness = st.number_input("Skin Thickness (mm)", value=20, min_value=0, max_value=100)
    insulin = st.number_input("Insulin Level (μU/ml)", value=85, min_value=0, max_value=1200)
    bmi = st.number_input("Body Mass Index (BMI)", value=25.0, min_value=0.0, max_value=100.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", value=0.5, min_value=0.0, max_value=2.5, step=0.1)
    age = st.number_input("Age", value=35, min_value=1, max_value=120)

    input_features = pd.DataFrame([[glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
                                  columns=["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", 
                                           "DiabetesPedigreeFunction", "Age"])
    if st.button("Predict Diabetes"):
        input_scaled = diabetes_scaler.transform(input_features)
        prediction = diabetes_model.predict(input_scaled)
        if prediction[0] == 1:
            st.error("🍭 The patient is likely to have diabetes.")
        else:
            st.success("🌱 The patient is unlikely to have diabetes.")
