import cv2
from ultralytics import YOLO
import pyttsx3
import time
import numpy as np

class ObstacleDetector:
    def __init__(self):
        # Cargar modelo YOLO con configuración mejorada
        self.model = YOLO("vision-assistant/backend/app/models/yolov8n-oiv7.pt")
        self.model.overrides['conf'] = 0.35  # Umbral más bajo para detectar más objetos
        self.model.overrides['iou'] = 0.45   # Balance entre precisión y detección
        
        # Configurar motor de voz robusto
        self.engine = self.setup_voice_engine()
        
        # Área de interés frontal (20% central del frame)
        self.front_zone = (0.4, 0.6)  # Rango X (40%-60%)
        self.danger_zone = 0.7         # Porción inferior (70%-100%)
        
        # Control de anuncios
        self.last_announce = 0
        self.announce_cooldown = 2  # Segundos entre anuncios
        
        # Obstáculos prioritarios (orden de importancia)
        self.priority_obstacles = [
            'person', 'bicycle', 'car', 'motorcycle', 
            'trash can', 'chair', 'table', 'pole'
        ]
    
    def setup_voice_engine(self):
        """Configuración robusta del motor de voz"""
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.setProperty('volume', 1.0)
            voices = engine.getProperty('voices')
            if len(voices) > 1:
                engine.setProperty('voice', voices[1].id)  # Voz más clara
            return engine
        except Exception as e:
            print(f"Error de voz: {e}. Instala: sudo apt-get install espeak")
            return None
    
    def is_in_danger_zone(self, x_center, y_bottom, frame_width, frame_height):
        """Determina si un objeto está en el área de peligro frontal"""
        x_min, x_max = self.front_zone
        x_relative = x_center / frame_width
        y_relative = y_bottom / frame_height
        
        # Objeto está en el centro horizontal y parte inferior
        return (x_min < x_relative < x_max) and (y_relative > self.danger_zone)
    
    def detect_obstacles(self, frame):
        """Detecta objetos potencialmente peligrosos"""
        results = self.model(frame, imgsz=640, verbose=False)
        obstacles = []
        
        for result in results:
            for box in result.boxes:
                label = result.names[int(box.cls)]
                conf = float(box.conf)
                
                if label in self.priority_obstacles and conf > 0.35:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x_center = (x1 + x2) / 2
                    y_bottom = y2
                    
                    if self.is_in_danger_zone(x_center, y_bottom, frame.shape[1], frame.shape[0]):
                        obstacles.append({
                            'label': label,
                            'confidence': conf,
                            'distance': 'muy cerca' if (y_bottom/frame.shape[0]) > 0.85 else 'cerca'
                        })
        
        return sorted(obstacles, key=lambda x: (
            -self.priority_obstacles.index(x['label']),  # Prioridad según lista
            -x['confidence']                            # Luego por confianza
        ))
    
    def announce_danger(self, obstacles):
        """Anuncia el obstáculo más peligroso"""
        if not obstacles or not self.engine:
            return
            
        current_time = time.time()
        if current_time - self.last_announce < self.announce_cooldown:
            return
        
        main_obstacle = obstacles[0]
        message = f"Cuidado: {main_obstacle['label']} {main_obstacle['distance']}"
        
        print(f"\n🚨 {message} 🚨")  # Feedback visual
        try:
            self.engine.say(message)
            self.engine.runAndWait()
            self.last_announce = current_time
        except Exception as e:
            print(f"Error en voz: {e}")
    
    def run(self, video_source):
        """Ejecuta el detector de obstáculos"""
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("Error al abrir el video/cámara")
            return
        
        print("\n🔍 Iniciando detector de obstáculos frontales...")
        print("Solo anunciará objetos en tu camino directo")
        print("Presiona Ctrl+C para detener\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Voltear horizontalmente para simular espejo (como visión humana)
                frame = cv2.flip(frame, 1)
                
                # Detección y anuncio
                obstacles = self.detect_obstacles(frame)
                self.announce_danger(obstacles)
                
                # Control de velocidad (~15 FPS)
                time.sleep(0.065)
                
        except KeyboardInterrupt:
            print("\nDeteniendo detector...")
        finally:
            cap.release()
            print("Proceso finalizado")

if __name__ == "__main__":
    detector = ObstacleDetector()
    
    # Para video (reemplaza con tu ruta)
    detector.run("/workspaces/AppDescripcionEntorno-Angel-Eduardo-Ramirez-Oliva/Videos de prueba/video_calle.mp4")
    
    # Para cámara web (descomenta):
    # detector.run(0)