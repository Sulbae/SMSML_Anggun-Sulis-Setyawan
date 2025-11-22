import requests
import json
import time
import logging

# Configure logging
logging.basicConfig(filename="api_model_logs.log", level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# API endpoint
API_URL = "https://127.0.0.1:5005/invocations" # Sesuaikan dengan URL endpoint model yang ingin digunakan

# Contoh input data untuk model
input_data = {
    "dataframe_split": {
        "columns": ["pH", "Hardness, Solids", "Chloramines", "Sulfate", "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity", "Potability"], 
        "data": [[7.036752103833548,204.8904554713363,20791.318980747023,7.300211873184757,368.51644134980336,564.3086541722439,10.3797830780847,86.9909704615088,2.9631353806316407,0]]
    }
}

# Konversi input data ke format JSON
headers = {"Content-Type": "application/json"}
payload = json.dumps(input_data)

# Mulai mencatat waktu eksekusi
start_time = time.time()

try:
    # Kirim request ke API
    response = requests.post(API_URL, headers=headers, data=payload)

    # Hitung response time
    response_time = time.time() - start_time

    if response.status_code == 200:
        predictions = response.json() # Ambil hasil prediksi dari response

        # Logging hasil request
        logging.info(f"Request: {input_data}, Response: {predictions}, Response Time: {response_time:.4f} sec")

        print(f"Predictions: {predictions}")
        print(f"Response Time: {response_time:.4f} sec")
    else:
        logging.error(f"Error {response.status_code}: {response.text}")
        print(f"Error {response.status_code}: {response.text}")

except Exception as e:
    logging.error(f"Exception: {str(e)}")
    print(f"Exception: {str(e)}")