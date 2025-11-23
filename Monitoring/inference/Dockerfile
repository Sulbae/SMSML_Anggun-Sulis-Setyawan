# Berdasarkan Base Image
FROM sulbae/mlflow-model:latest AS trainer

WORKDIR /app

FROM python:3.12-slim

WORKDIR /app

# Salin dependensi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin model
COPY --from=trainer /app/water_potability.pkl /app/water_potability.pkl

# Salin app 
COPY inference_app.py .
COPY preprocess_prediction.py .

# Port 8501 untuk Streamlit
EXPOSE 8501
# Port 8000 untuk Prometheus
EXPOSE 8000

# Running Command (CMD)
CMD [ "streamlit", "run", "inference_app.py", "--server.port", "8501", "--server.address", "0.0.0.0" ]