import streamlit as st
import pandas as pd
from preprocess_prediction import data_preprocessing
import json
from prometheus_client import start_http_server, CollectorRegistry, Counter, Summary, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time
import joblib
import psutil

MODEL_NAME = "water_potability"
MODEL_VERSION = "v1.0.0"

@st.cache_resource
def init_metrics():
    registry = CollectorRegistry()

    metrics = {
        # Tracking Jumlah Request
        "REQUEST_COUNT": Counter("streamlit_request_count", "Jumlah request prediksi", ["model_name", "status"]),
        # Tracking Waktu Response
        "PREDICTION_TIME": Summary("streamlit_prediction_latency_seconds", "Waktu yang dibutuhkan untuk prediksi", ["model_name"]),
        # Penggunaan Sistem: CPU dan RAM
        "CPU_USAGE": Gauge('system_cpu_usage', 'CPU Usage Percentage'),
        "RAM_USAGE": Gauge('system_ram_usage', 'RAM Usage Percentage'),
        # Distribusi Output
        "OUTPUT_POTABILITY_COUNT": Counter("model_output_count", "Hitung jumlah utput per kelas", ["prediction"])
    }

    # Versi Model Aktif
    MODEL_VERSION_GAUGE = Gauge("model_version", "Versi model saat ini", ["version"])
    MODEL_VERSION_GAUGE.labels(version=MODEL_VERSION).set(1)

    try:
        start_http_server(8000, registry=registry)
        st.sidebar.success("Prometheus metrics aktif (serving) di port 8000")
    except OSError:
        st.sidebar.warning("Server Prometheus sudah berjalan.")
    
    return metrics

METRICS = init_metrics()

@st.cache_resource
def load_model():
    try:    
        model = joblib.load("water_potability.pkl")
        st.sidebar.success(f"Model {type(model).__name__} ({MODEL_VERSION}) sudah dimuat!")
        return model
    except Exception as e:
        st.sidebar.error(f"Gagal memuat model: {e}. Pastikan file model ada.")
        return None

def update_system_metrics():
    METRICS["CPU_USAGE"].set(psutil.cpu_percent(interval=None))
    METRICS["RAM_USAGE"].set(psutil.virtual_memory().percent)

MODEL = load_model()

# Streamlit UI
st.title("Aplikasi Prediksi Kelayakan Air")
st.markdown("Masukkan data setiap parameter untuk memprediksi **Water Potability (Kelayakan Minum Air)**!")

# Cek model
if MODEL is not None:
    st.caption(f"Status: Model {MODEL_NAME} ({MODEL_VERSION}) siap.")
else:
    st.error("MODEL GAGAL DIMUAT. Periksa log Docker untuk memastikan file model tersedia.")

# Input Data
columns = [
        "ph", "Hardness", "Solids", 
        "Chloramines", "Sulfate", "Conductivity", 
        "Organic_carbon", "Trihalomethanes", "Turbidity"
]

## Input data numerik
ph = st.number_input("ph", min_value=0.1, max_value=14.0, value=7.0)
hardness = st.number_input("Hardness", min_value=0.1, max_value=1000.0, value=200.0)
solids = st.number_input("Solids", min_value=0.1, max_value=100000.0, value=20000.0)
chloramines = st.number_input("Chloramines", min_value=0.1, max_value=100.0, value=7.0)
sulfate = st.number_input("Sulfate", min_value=0.1, max_value=1000.0, value=300.0)
conductivity = st.number_input("Conductivity", min_value=0.1, max_value=1000.0, value=400.0)
organic_carbon = st.number_input("Organic_carbon", min_value=0.1, max_value=1000.0, value=15.0)
trihalomethanes = st.number_input("Trihalomethanes", min_value=0.1, max_value=1000.0, value=80.0)
turbidity = st.number_input("Turbidity", min_value=0.1, max_value=100.0, value=4.0)

# Menyimpan data ke dalam DataFrame
data_input = pd.DataFrame([[
    ph, hardness, solids, 
    chloramines, sulfate, conductivity, 
    organic_carbon, trihalomethanes, turbidity
]], columns=columns)

# Tombol untuk menampilkan data
if st.button("Prediksi Kelayakan Air", type="primary"):
    update_system_metrics()

    st.write("### Data yang dimasukkan:")
    st.dataframe(data_input)

    start_time = time.time()

    if MODEL is None:
        prediction_status = "error"
        st.error("Maaf, tidak dapat melakukan prediksi karena model gagal dimuat.")
    else:
        prediction_status = "success"
        
        try:
            # Preprocessing
            new_data = data_preprocessing(data_input)
            st.write("### Data setelah diolah:")
            st.dataframe(new_data)

            # Prediksi
            prediction_array = MODEL.predict(new_data)
            result = int(prediction_array[0])
            
            # Metrik Distribusi Output
            METRICS["OUTPUT_POTABILITY_COUNT"].labels(prediction=str(result)).inc()
            
            # Tampilkan hasil prediksi
            st.success(f"Hasil Prediksi: {'Air Layak Minum.' if result == 1 else 'Air Tidak Layak Minum!'}")
        
        except Exception as e:
            prediction_status = "error"
            st.error(f"Terjadi kesalahan sistem: {e}")
            result = {"Error": str(e)}

        finally:
            # End
            end_time = time.time()
            latency = end_time - start_time

            # Metrik jumlah requests
            METRICS["REQUEST_COUNT"].labels(model_name=MODEL_NAME, status=prediction_status).inc()
            # Metrik Latency
            METRICS["PREDICTION_TIME"].labels(model_name=MODEL_NAME).observe(latency)

            st.caption(f"Waktu Proses: {latency:.4f} detik")