import streamlit as st
import pandas as pd
import joblib

# Load the saved scaler and model
scaler = joblib.load("parkinsonscaler.pkl")
model = joblib.load("parkinsons_model.pkl")

# Streamlit App Title
st.title("Parkinson's Disease Detection App")

# Input Form in Sidebar for Patient Details
st.sidebar.header("Enter Patient's Health Details")

MDVP_Fo_Hz = st.sidebar.number_input("MDVP:Fo(Hz) - Average Fundamental Frequency", min_value=0.0, step=0.1, value=120.0)
MDVP_Fhi_Hz = st.sidebar.number_input("MDVP:Fhi(Hz) - Maximum Fundamental Frequency", min_value=0.0, step=0.1, value=150.0)
MDVP_Flo_Hz = st.sidebar.number_input("MDVP:Flo(Hz) - Minimum Fundamental Frequency", min_value=0.0, step=0.1, value=85.0)
MDVP_Jitter = st.sidebar.number_input("MDVP:Jitter(%)", min_value=0.0, step=0.001, format="%.3f", value=0.005)
MDVP_Jitter_Abs = st.sidebar.number_input("MDVP:Jitter(Abs)", min_value=0.0, step=0.00001, format="%.5f", value=0.00003)
MDVP_RAP = st.sidebar.number_input("MDVP:RAP", min_value=0.0, step=0.001, format="%.3f", value=0.003)
MDVP_PPQ = st.sidebar.number_input("MDVP:PPQ", min_value=0.0, step=0.001, format="%.3f", value=0.003)
Jitter_DDP = st.sidebar.number_input("Jitter:DDP", min_value=0.0, step=0.001, format="%.3f", value=0.010)
MDVP_Shimmer = st.sidebar.number_input("MDVP:Shimmer", min_value=0.0, step=0.001, format="%.3f", value=0.020)
MDVP_Shimmer_dB = st.sidebar.number_input("MDVP:Shimmer(dB)", min_value=0.0, step=0.1, format="%.1f", value=0.5)
Shimmer_APQ3 = st.sidebar.number_input("Shimmer:APQ3", min_value=0.0, step=0.001, format="%.3f", value=0.015)
Shimmer_APQ5 = st.sidebar.number_input("Shimmer:APQ5", min_value=0.0, step=0.001, format="%.3f", value=0.020)
MDVP_APQ = st.sidebar.number_input("MDVP:APQ", min_value=0.0, step=0.001, format="%.3f", value=0.025)
Shimmer_DDA = st.sidebar.number_input("Shimmer:DDA", min_value=0.0, step=0.001, format="%.3f", value=0.045)
NHR = st.sidebar.number_input("NHR", min_value=0.0, step=0.001, format="%.3f", value=0.005)
HNR = st.sidebar.number_input("HNR", min_value=0.0, step=0.1, format="%.1f", value=20.0)
RPDE = st.sidebar.number_input("RPDE", min_value=0.0, step=0.001, format="%.3f", value=0.5)
DFA = st.sidebar.number_input("DFA", min_value=0.0, step=0.001, format="%.3f", value=0.7)
spread1 = st.sidebar.number_input("Spread1", min_value=-10.0, step=0.1, format="%.1f", value=-5.0)
spread2 = st.sidebar.number_input("Spread2", min_value=0.0, step=0.1, format="%.1f", value=0.2)
D2 = st.sidebar.number_input("D2", min_value=0.0, step=0.1, format="%.1f", value=2.0)
PPE = st.sidebar.number_input("PPE", min_value=0.0, step=0.001, format="%.3f", value=0.2)

# Convert user input into a DataFrame
input_features = pd.DataFrame(
    [[MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter, MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP,
      MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA,
      spread1, spread2, D2, PPE]],
    columns=["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", 
             "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", 
             "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"]
)

# Predict button
if st.sidebar.button("Predict"):
    # Scale the input data
    input_scaled = scaler.transform(input_features)

    # Make a prediction
    prediction = model.predict(input_scaled)

    # Display the result
    if prediction[0] == 1:
        st.sidebar.error("The patient is likely to have Parkinson's Disease.")
    else:
        st.sidebar.success("The patient is unlikely to have Parkinson's Disease.")
