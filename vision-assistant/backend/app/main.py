from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import detection, realtime
import logging

app = FastAPI(
    title="Vision Assistant API",
    description="API para detección de objetos en tiempo real",
    version="1.0.0"
)

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(detection.router, prefix="/api/v1")
app.include_router(realtime.router, prefix="/api/v1")

# Health Check
@app.get("/")
async def health_check():
    return {"status": "active", "service": "vision-assistant"}