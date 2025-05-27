import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

# Load model and scaler
model = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')

# Page configuration
st.set_page_config(page_title="Crime Rate Predictor", layout="centered", page_icon="🔍")

# Custom background and layout styling
st.markdown(
    """
    <style>
        /* Set background color or image */
        .stApp {
            background-color: #F0F2F6;  /* Light gray-blue background */
            background-image: url("https://www.transparenttextures.com/patterns/white-wall.png");
            background-size: cover;
        }

        /* Style for the containers and inputs */
        .css-1d391kg {  /* Streamlit container */
            background-color: #ffffff !important;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
        }

        /* Change font and spacing */
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
            font-size: 16px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Header section
st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color: #FF4B4B;">🔍 Crime Rate Predictor</h1>
        <p style="font-size: 18px;">Predict the crime rate category based on socio-economic factors</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# Input form inside a container
with st.container():
    st.subheader("🧾 Input Socio-Economic Data")
    
    col1, col2 = st.columns(2)

    with col1:
        density = st.slider(
            "📊 Population Density (people/km²)",
            min_value=0, max_value=5000, value=1000,
            help="Higher density might correlate with higher crime rates"
        )
        literacy = st.slider(
            "📚 Literacy Rate (%)",
            min_value=0.0, max_value=100.0, value=75.0,
            help="Literacy is inversely related to crime rate in many regions"
        )
        addiction = st.slider(
            "💊 Drug Addiction Rate (%)",
            min_value=0.0, max_value=50.0, value=10.0,
            help="Higher addiction rates may increase crime incidents"
        )

    with col2:
        income = st.slider(
            "💰 Per Capita Income (₹)",
            min_value=0, max_value=300000, value=100000,
            help="Economic stability may reduce crime"
        )
        unemployment = st.slider(
            "📉 Unemployment Rate (%)",
            min_value=0.0, max_value=30.0, value=7.0,
            help="Unemployment may contribute to crime rates"
        )

# Prediction button
st.markdown("---")
if st.button("🧠 Predict Crime Category"):
    input_data = np.array([[density, income, literacy, unemployment, addiction]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    probability = model.predict_proba(scaled_input)[0][1]  # Probability of High Crime

    # Show confidence/probability visually
    st.subheader("📈 Model Confidence")
    st.progress(int(probability * 100))
    st.markdown(f"🔢 **Confidence:** `{round(probability*100, 2)}%` chance of High Crime Rate")

    # Show prediction message
    if prediction[0] == 1:
        st.markdown(
            "<div style='background-color:#FFCCCC;padding:20px;border-radius:10px;'>"
            "<h3 style='color:red;'>🔴 Prediction: High Crime Rate</h3>"
            "<p>Consider improving literacy, employment, and addiction support programs.</p>"
            "</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div style='background-color:#CCFFCC;padding:20px;border-radius:10px;'>"
            "<h3 style='color:green;'>🟢 Prediction: Low Crime Rate</h3>"
            "<p>Keep up the good work in maintaining social wellbeing!</p>"
            "</div>",
            unsafe_allow_html=True
        )

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center;font-size:12px;'>Developed by Santosh | Powered by Logistic Regression</p>",
    unsafe_allow_html=True
)
