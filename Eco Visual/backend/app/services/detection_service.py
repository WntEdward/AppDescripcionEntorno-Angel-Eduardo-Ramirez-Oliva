from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import logging

class DetectionService:
    def __init__(self):
        try:
            self.model = YOLO("app/models/yolov8n-oiv7.pt")
            # Verificación simple del modelo
            dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
            self.model.predict(dummy_image, verbose=False)
            logging.info("Modelo YOLO cargado correctamente")
        except Exception as e:
            logging.error(f"Error al cargar el modelo YOLO: {str(e)}")
            raise

    async def detect_objects(self, image_bytes: bytes) -> list:
        try:
            # Convertir bytes a imagen
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convertir a formato OpenCV
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Ejecutar detección
            results = self.model(image_cv, verbose=False)
            
            # Procesar resultados
            detections = []
            for result in results:
                for box in result.boxes:
                    detections.append({
                        "object": result.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "position": {
                            "x1": int(box.xyxy[0][0]),
                            "y1": int(box.xyxy[0][1]),
                            "x2": int(box.xyxy[0][2]),
                            "y2": int(box.xyxy[0][3])
                        }
                    })
            
            return detections
            
        except Exception as e:
            logging.error(f"Error en detect_objects: {str(e)}")
            raise ValueError(f"Error en el procesamiento de imagen: {str(e)}")