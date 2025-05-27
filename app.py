import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

# Load the model and scaler
model = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Crime Rate Predictor", layout="centered")
st.title("🔍 Crime Rate Predictor")
st.write("Enter the socio-economic factors of a state:")

# User input
density = st.slider("Population Density (people/km²)", 0, 5000, 1000)
income = st.slider("Per Capita Income (₹)", 0, 300000, 100000)
literacy = st.slider("Literacy Rate (%)", 0.0, 100.0, 75.0)
unemployment = st.slider("Unemployment Rate (%)", 0.0, 30.0, 7.0)
addiction = st.slider("Drug Addiction Rate (%)", 0.0, 50.0, 10.0)

if st.button("Predict Crime Category"):
    data = np.array([[density, income, literacy, unemployment, addiction]])
    scaled_data = scaler.transform(data)
    result = model.predict(scaled_data)
    
    if result[0] == 1:
        st.error("🔴 High Crime Rate")
    else:
        st.success("🟢 Low Crime Rate")


# Check if model is multiclass
st.subheader("🔍 Model Info")

try:
    st.write("Classes:", model.classes_)
    st.write("Number of Classes:", len(model.classes_))
    st.write("Model multi_class setting:", model.multi_class if hasattr(model, 'multi_class') else "Not available")
except Exception as e:
    st.warning(f"Couldn't read model attributes: {e}")
