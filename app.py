# app.py

import streamlit as st
import numpy as np
import joblib

# Muat model K-Nearest Neighbors yang sudah dilatih
# Pastikan file 'harga_rumah.pkl' berada di direktori yang sama dengan app.py
try:
    model = joblib.load('harga_rumah.pkl')
except FileNotFoundError:
    st.error("File model 'harga_rumah.pkl' tidak ditemukan. Pastikan Anda sudah menjalankan skrip training dan menyimpan modelnya.")
    st.stop()

# Judul aplikasi
st.title("Prediksi Harga Rumah Sederhana")
st.write("Aplikasi ini memprediksi harga rumah (dalam jutaan rupiah) berdasarkan luas, jumlah kamar tidur, dan jumlah kamar mandi menggunakan algoritma KNN.")

# Form untuk input data oleh pengguna
with st.form("form_harga_rumah"):
    # Input untuk luas rumah
    luas = st.number_input('Luas Tanah (m²)', min_value=100, max_value=10000, step=100, help="Masukkan luas tanah dalam meter persegi.")
    
    # Input untuk jumlah kamar tidur
    kamar_tidur = st.number_input('Jumlah Kamar Tidur', min_value=1, max_value=10, step=1, help="Masukkan jumlah kamar tidur.")
    
    # Input untuk jumlah kamar mandi
    kamar_mandi = st.number_input('Jumlah Kamar Mandi', min_value=1, max_value=10, step=1, help="Masukkan jumlah kamar mandi.")
    
    # Tombol untuk submit form
    submit = st.form_submit_button("Prediksi Harga")

# Ketika tombol "Prediksi Harga" ditekan
if submit:
    # Format input dari pengguna ke dalam bentuk array numpy
    # Sesuai dengan format yang digunakan saat training model: ['luas', 'kasur', 'km']
    fitur = np.array([[luas, kamar_tidur, kamar_mandi]])
    
    # Lakukan prediksi harga dengan model yang sudah dimuat
    prediksi_harga = model.predict(fitur)[0]
    
    # Tampilkan hasil prediksi
    st.success(f"Prediksi Harga Rumah: Rp {prediksi_harga:,.2f} Juta")
