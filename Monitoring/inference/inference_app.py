import streamlit as st
import pandas as pd
from Monitoring.inference.preprocess_prediction import data_preprocessing, prediction
import json
from prometheus_client import start_http_server, Counter, Summary 
import time
import joblib

# Set up prometheus
REQUEST_COUNT = Counter(
    "streamlit_request_count",
    "Jumlah request prediksi",
    ["model_name"]
)

PREDICTION_TIME = Summary(
    "streamlit_prediction_latency_seconds",
    "Waktu yang dibutuhkan untuk prediksi",
    ["model_name"]
)

MODEL_NAME = "water_potability"
try:
    start_http_server(8000)
    st.sidebar.success("Prometheus metrics aktif (serving) di port 8000")
except OSError:
    st.sidebar.warning("Server Prometheus sudah berjalan.")

# Streamlit UI
st.title("Form Input Data Kualitas Air")
st.markdown("Masukkan data setiap parameter untuk memprediksi **Water Potability (Kelayakan Minum Air)**!")

# Load model
@st.cache_resource
def load_model():
    model = joblib.load("water_potability.pkl")
    return model

model = load_model()

if model is not None:
    st.write(f"Model {model} sudah siap!")

# Input Data
columns = [
        "pH", "Hardness, Solids", 
        "Chloramines", "Sulfate", "Conductivity", 
        "Organic_carbon", "Trihalomethanes", "Turbidity"
]

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
if st.button("Prediksi Kelayakan Air", type="primary"):
    st.write("### Data yang dimasukkan:")
    st.dataframe(data)

    if model is None:
        st.error("Maaf, tidak dapat melakukan prediksi karena model gagal dimuat.")
    else:
        start_time = time.time()
        prediction_status = "Success"
        try:
            # Preprocessing
            new_data = data_preprocessing(data=data)
            st.write("### Data setelah diolah:")
            st.dataframe(new_data)

            json_output = {
                "dataframe_split": {
                    "columns": new_data.columns.tolist(),
                    "data": new_data.values.tolist()
                }
            }

            data_testing = json.dumps(json_output)

            # Prediksi
            result = prediction(data_testing)

            # Tampilkan hasil prediksi
            st.write("### Hasil Prediksi")

            if result == 1:
                st.success("Air **LAYAK** untuk diminum.")
            else: 
                st.warning("Air **TIDAK LAYAK** untuk diminum!")
        
        except Exception as e:
            prediction_status = "error"
            st.error(f"Terjadi kesalahan sistem: {e}")
            result = {"Error": str(e)}

        finally:
            # End
            end_time = time.time()
            latency = end_time - start_time

            # Metrik jumlah requests
            REQUEST_COUNT.labels(model_name=MODEL_NAME, status=prediction_status).inc()

            # Metrik Latency
            PREDICTION_TIME.labels(model_name=MODEL_NAME).observe(latency)

            st.caption(f"Waktu Proses: {latency:.4f} detik")
