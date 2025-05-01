import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model dan scaler
model = joblib.load("../model/model.pkl")
scaler = joblib.load("../model/scaler.pkl")

# Konfigurasi Streamlit
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺")

st.title("🩺 Aplikasi Prediksi Diabetes dengan Machine Learning")
st.markdown("Gunakan data medis untuk memprediksi kemungkinan seseorang terkena diabetes.")

# Card informasi model
with st.expander("ℹ️ Tentang Model"):
    st.markdown("""
    - *Model*: Logistic Regression  
    - *Preprocessing*: Penghapusan outlier, normalisasi  
    - *Tuning*: GridSearchCV (C, penalty)  
    - *Fitur yang Digunakan*:  
        - Pregnancies  
        - Glucose  
        - BloodPressure  
        - SkinThickness  
        - Insulin  
        - BMI  
        - DiabetesPedigreeFunction  
        - Age
    """)

# Input form
st.header("📝 Input Data Pasien")

with st.form("form_pasien"):
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Jumlah Kehamilan (Pregnancies)", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Kadar Glukosa", min_value=0, max_value=300, value=100)
        blood_pressure = st.number_input("Tekanan Darah (mmHg)", min_value=0, max_value=200, value=70)
        skin_thickness = st.number_input("Ketebalan Kulit", min_value=0, max_value=100, value=20)
    with col2:
        insulin = st.number_input("Kadar Insulin", min_value=0, max_value=900, value=79)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.2f")
        age = st.number_input("Usia", min_value=1, max_value=120, value=33)

    submit = st.form_submit_button("Prediksi")

# Proses prediksi
if submit:
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0][prediction]

    label = "Positif Diabetes" if prediction == 1 else "Negatif Diabetes"
    color = "red" if prediction == 1 else "green"

    # Card hasil prediksi
    st.subheader("📋 Hasil Prediksi")
    st.markdown(f"""
    <div style="background-color:{color};padding:15px;border-radius:10px">
        <h3 style="color:white;">{label}</h3>
        <p style="color:white;">Probabilitas: {prediction_proba:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

    # Tampilkan data input dan hasil prediksi
    hasil_df = pd.DataFrame(input_data, columns=[
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ])
    hasil_df["Prediction"] = label

    st.write("📄 Tabel Data Input & Prediksi")
    st.dataframe(hasil_df)

    # Simpan hasil (opsional - bisa simpan ke csv atau db)
    hasil_df.to_csv("hasil_prediksi.csv", mode='a', header=False, index=False)