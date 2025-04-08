from fastapi import FastAPI
from app.routes.detection import router as detection_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Vision Assistant API",
    description="API para aplicación de asistencia visual",
    version="0.1.0"
)

# Configurar CORS para desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detection_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Vision Assistant API está funcionando"}