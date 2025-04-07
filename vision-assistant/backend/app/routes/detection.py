from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.detection_service import DetectionService
from fastapi.responses import JSONResponse
import logging

router = APIRouter()
detection_service = DetectionService()

@router.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    try:
        # Validación básica del archivo
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="Solo se permiten archivos de imagen (JPEG, PNG)"
            )
        
        # Leer contenido de la imagen
        image_bytes = await file.read()
        
        if not image_bytes or len(image_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail="El archivo de imagen está vacío"
            )
        
        # Procesar imagen
        detections = await detection_service.detect_objects(image_bytes)
        
        # Generar descripciones
        descriptions = [
            f"Objeto {i+1}: {det['object']} ({det['confidence']*100:.1f}%)"
            for i, det in enumerate(detections[:5])
        ]
        
        return JSONResponse(content={
            "success": True,
            "descriptions": descriptions,
            "objects": detections
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error en el endpoint /detect")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando la imagen: {str(e)}"
        )