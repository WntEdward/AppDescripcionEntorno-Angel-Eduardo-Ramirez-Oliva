from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.services.detection_service import DetectionService
import logging

router = APIRouter()
detection_service = DetectionService()

@router.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    try:
        # Validaciones básicas
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "Solo se permiten imágenes (JPEG/PNG)")
        
        image_bytes = await file.read()
        if len(image_bytes) < 1024:  # 1KB mínimo
            raise HTTPException(400, "Imagen demasiado pequeña")
        
        # Procesamiento
        detections = await detection_service.process_image(image_bytes)
        return JSONResponse(content={
            "success": True,
            "objects": detections,
            "count": len(detections)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Detection error: {str(e)}")
        raise HTTPException(500, "Error procesando la imagen")