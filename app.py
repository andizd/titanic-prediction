import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load Model
with open("titanic_3feature_new.pkcls", "rb") as file:
    model = pickle.load(file)

# Judul Aplikasi
st.title("🚢 Prediksi Keselamatan Penumpang Titanic")

st.markdown("""
Aplikasi ini memprediksi apakah seorang penumpang **selamat** atau **tidak selamat** berdasarkan data dasar seperti kelas penumpang, usia, dan jenis kelamin.
""")

# Input Data dari Pengguna
status = st.selectbox("Kelas Penumpang", ["first", "second", "third", "crew"])
age = st.selectbox("Umur Penumpang", ["child", "adult"])
sex = st.selectbox("Jenis Kelamin", ["male", "female"])

# Buat one-hot encoding untuk input data
features = {
    "status_crew": 0,
    "status_first": 0,
    "status_second": 0,
    "status_third": 0,
    "age_adult": 0,
    "age_child": 0,
    "sex_female": 0,
    "sex_male": 0,
    }

# Aktifkan fitur sesuai input pengguna
features[f"status_{status}"] = 1
features[f"age_{age}"] = 1
features[f"sex_{sex}"] = 1

input_data = pd.DataFrame([features])

# input_data = pd.DataFrame({
#     "survived": [survived],
#     "status": [status],
#     "sex": [sex],
#     "age": [age],
# })


# --- Prediksi ---
try:
    X = input_data.to_numpy()

    prediction = model(X)[0]
    probability = model(X, model.Probs)[0][int(prediction)]

    # --- Tampilkan Hasil ---
    st.subheader("🎯 Hasil Prediksi:")
    if prediction == 1:
        st.success(f"✅ Penumpang diprediksi **Selamat** (Probabilitas: {probability:.2f})")
    else:
        st.error(f"❌ Penumpang diprediksi **Tidak Selamat** (Probabilitas: {probability:.2f})")

except Exception as e:
    st.warning("⚠️ Terjadi kesalahan saat memproses data. Pastikan urutan fitur sesuai dengan model di Orange.")
    st.text(str(e))

# --- Footer ---
st.caption("Created by Zaid and Nabila - Data Mining 2025")