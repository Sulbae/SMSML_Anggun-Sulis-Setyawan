import streamlit as st
import pandas as pd
from preprocess_prediction import data_preprocessing, prediction
import json

# Input Data
columns = [
        "pH", "Hardness, Solids", 
        "Chloramines", "Sulfate", "Conductivity", 
        "Organic_carbon", "Trihalomethanes", "Turbidity"
]

# Streamlit UI
st.title("Form Input Data Kualitas Air")

## Input data numerik
ph = st.number_input("pH", min_value=0.1, max_value=14)
hardness = st.number_input("Hardness", min_value=0.1, max_value=1000)
solids = st.number_input("Solids", min_value=0.1, max_value=100000)
chloramines = st.number_input("Chloramines", min_value=0.1, max_value=100)
sulfate = st.number_input("Sulfate", min_value=0.1, max_value=1000)
conductivity = st.number_input("Conductivity", min_value=0.1, max_value=1000)
organic_carbon = st.number_input("Organic_carbon", min_value=0.1, max_value=1000)
trihalomethanes = st.number_input("Trihalomethanes", min_value=0.1, max_value=1000)
turbidity = st.number_input("Turbidity", min_value=0.1, max_value=100)

# Menyimpan data ke dalam DataFrame
data = pd.DataFrame([[
    ph, hardness, solids, 
    chloramines, sulfate, conductivity, 
    organic_carbon, trihalomethanes, turbidity
]], columns=columns)

# Tombol untuk menampilkan data
if st.button("Simpan & Tampilkan Data"):
    st.write("### Data yang dimasukkan:")
    st.dataframe(data)

    new_data = data_preprocessing(data=data)
    st.write("### Data setelah diolah:")
    st.dataframe(new_data)

    result = prediction(new_data)

    # Tampilkan hasil prediksi
    st.write("### Hasil Prediksi")
    st.write(result)