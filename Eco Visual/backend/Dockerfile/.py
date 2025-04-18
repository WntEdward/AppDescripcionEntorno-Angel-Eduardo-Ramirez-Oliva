FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Descargar el modelo YOLO durante el build (opcional)
# RUN python -c "from ultralytics import YOLO; YOLO('yolov8n-oiv7.pt')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
