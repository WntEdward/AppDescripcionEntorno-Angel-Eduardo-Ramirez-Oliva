from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.detection_service import DetectionService
from typing import List, Dict
import logging

router = APIRouter()
detection_service = DetectionService()

@router.post("/detect", response_model=List[Dict])
async def detect_objects(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Formato de archivo no soportado")
            
        image_bytes = await file.read()
        detections = await detection_service.detect_objects(image_bytes)
        
        # Generar descripciones accesibles
        descriptions = []
        for i, detection in enumerate(detections[:5]):  # Limitar a 5 objetos principales
            descriptions.append(
                f"Objeto {i+1}: {detection['object']} con {detection['confidence']*100:.1f}% de confianza"
            )
            
        return {"descriptions": descriptions, "raw_detections": detections}
        
    except Exception as e:
        logging.error(f"Error en /detect: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
