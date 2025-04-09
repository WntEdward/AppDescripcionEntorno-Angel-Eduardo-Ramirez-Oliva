import cv2
import numpy as np
from ultralytics import YOLO
from typing import Generator

class RealtimeProcessor:
    def __init__(self):
        self.model = YOLO('app/models/yolov8n-oiv7.pt')
        # Configuración para baja latencia
        self.model.overrides['conf'] = 0.5  # Umbral de confianza
        self.model.overrides['iou'] = 0.3  # Umbral de IoU
        self.model.overrides['agnostic_nms'] = True  # NMS agnóstico
        self.model.overrides['max_det'] = 10  # Máximo de detecciones
        
    async def process_frame(self, frame_data: bytes) -> dict:
        """Procesa un frame individual"""
        try:
            # Convertir bytes a imagen OpenCV
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Ejecutar detección
            results = self.model(frame, verbose=False)
            
            # Procesar resultados
            detections = []
            for result in results:
                for box in result.boxes:
                    detections.append({
                        'class': result.names[int(box.cls)],
                        'confidence': float(box.conf),
                        'bbox': box.xyxy[0].tolist()
                    })
            
            return {'success': True, 'detections': detections}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def generate_frames(self, frame_gen: Generator[bytes, None, None]) -> Generator[dict, None, None]:
        """Procesa un stream de frames"""
        for frame_data in frame_gen:
            yield await self.process_frame(frame_data)