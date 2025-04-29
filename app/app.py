import streamlit as st 
import pandas as pd 
import joblib 
import numpy 

# Load model dan scaler 
model  = joblib.load("../model/model.pkl")
scaler = joblib.load("../model/scaler.pkl")

# konfigurasi streamlit 
st.set_page_config(page_title="Diabetes Predicter")

st.title("Aplikasi Prediksi Penyakit Diabetes Dengan Machine Learning")
st.markdown("Gunakan data medis untuk memprediksi kemungkinan seseorang terkena diabetes.")

# card informasi model 
with st.expander("Tentang Model"):
    st.markdown("""
    - **Model**:Logistic Regression
    - **Preprocessing**: Penghapusan outlier, normalisasi 
    - **Tuning**: GridSearchCV (C, penalty)
    - **Fitur yang digunakan**:
        - Pregnancies 
        - Glucose 
        - BloodPressure 
        - SkinThickness
        - BMI
        - DiabetesPedigreeFunction
        - Age
    """)

# input form 
st.header("Input data pasien")

with st.form("form_pasien"):
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Jumlah Kehamilan (Pregnancies)", min_value = 0, max_value=20, value=0)
        glucose = st.number_input("Kadar Glukosa", min_value=0, max_value=300, value=100)
        blood_pressure = st.number_input('Tekanan darah (mmHg)', min_value=0, max_value=200, value=70)
        skin_thicknees = st.number_input('ketebalan kulit', min_value=0, max_value=100, value=20)
    with col2:
        insulin = st.number_input("Kadar Insulin", min_value=0, max_value=900, value=79)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
        dpf = st.number_input("Diabetes Pediagree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.2f")
        age = st.number_input("Usia", min_value=1, max_value=120, value=33)

    submit = st.form_submit_button("Prediksi")