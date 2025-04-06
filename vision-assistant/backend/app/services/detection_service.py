from ultralytics import YOLO
from typing import List, Dict
import cv2
import numpy as np
from PIL import Image
import io

class DetectionService:
    def __init__(self):
        # Carga el modelo preentrenado
        self.model = YOLO("app/models/yolov8n-oiv7.pt")
        
    async def detect_objects(self, image_bytes: bytes) -> List[Dict]:
        """Detecta objetos en una imagen y devuelve descripciones accesibles"""
        try:
            # Convertir bytes a imagen OpenCV
            image = Image.open(io.BytesIO(image_bytes))
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Ejecutar detección
            results = self.model(image_cv)
            
            # Procesar resultados para descripción accesible
            detections = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    label = result.names[class_id]
                    confidence = float(box.conf)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    detections.append({
                        "object": label,
                        "confidence": confidence,
                        "position": {
                            "x_center": (x1 + x2) / 2,
                            "y_center": (y1 + y2) / 2,
                            "width": x2 - x1,
                            "height": y2 - y1
                        }
                    })
            
            # Ordenar por confianza (mayor primero)
            detections.sort(key=lambda x: x["confidence"], reverse=True)
            
            return detections
            
        except Exception as e:
            raise ValueError(f"Error en detección de objetos: {str(e)}")
