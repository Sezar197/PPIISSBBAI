import streamlit as st
import numpy as np
import joblib
#from tensorflow.keras.models import load_model
def timoshenko_deflection(E, nu, b, h, P, L=4000):
    A = b * h
    I = (b * h**3) / 12
    G = E / (2 * (1 + nu))
    k = 5/6
    
    delta_b = (P * L**3) / (48 * E * I)
    delta_s = (P * L) / (4 * k * G * A)
    
    return delta_b + delta_s
#rf = joblib.load("rf_model.pkl")
xgb = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
#ann = load_model("ann_model.h5")

st.title("Beam Deflection Predictor")

E = st.number_input("Young's Modulus", value=210000.0)
nu = st.number_input("Poisson Ratio", value=0.3)
w = st.number_input("Width", value=200.0)
d = st.number_input("Depth", value=200.0)
P = st.number_input("Load", value=1000000.0)

if st.button("Predict"):
    X = np.array([[E, nu, w, d, P]])
    # Analytical value
    theo = timoshenko_deflection(E, nu, w, d, P)

    # Error %
    error = abs((xgb_pred - theo) / theo) * 100
    #rf_pred = rf.predict(X)[0]
    xgb_pred = xgb.predict(X)[0]

    X_scaled = scaler.transform(X)
    #ann_pred = ann.predict(X_scaled)[0][0]

    st.write("### Predictions")
    st.write(f"XGBoost: {xgb_pred}")
    #st.write(f"ANN: {ann_pred}")
    st.write(f"Timoshenko (Analytical): {theo}")
    st.write(f"Error (%): {error:.4f}")
