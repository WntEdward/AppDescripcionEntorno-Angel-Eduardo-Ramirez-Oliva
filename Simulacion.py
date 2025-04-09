import cv2
from ultralytics import YOLO
import pyttsx3
import time
from collections import defaultdict

class GeneralObjectDetector:
    def __init__(self):
        # Cargar modelo YOLO
        self.model = YOLO("vision-assistant/backend/app/models/yolov8n-oiv7.pt")
        
        # Configurar motor de voz
        self.engine = self.init_voice_engine()
        self.last_announce_time = 0
        self.announce_cooldown = 5  # Segundos entre anuncios
        
    def init_voice_engine(self):
        """Inicializa el motor de voz con soporte para Codespaces"""
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            return engine
        except Exception as e:
            print(f"Advertencia: No se pudo inicializar voz. Error: {e}")
            print("Instala espeak con: sudo apt-get install espeak libespeak1")
            return None
    
    def describe_scene(self, frame):
        """Detecta y describe todos los objetos en el frame"""
        results = self.model(frame, verbose=False)
        object_counts = defaultdict(int)
        object_details = []
        
        for result in results:
            for box in result.boxes:
                label = result.names[int(box.cls)]
                conf = float(box.conf)
                if conf > 0.3:  # Umbral de confianza mínimo
                    object_counts[label] += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    object_details.append({
                        'label': label,
                        'confidence': conf,
                        'position': self.get_position_description(x1, x2, frame.shape[1])
                    })
        
        return object_counts, object_details
    
    def get_position_description(self, x1, x2, frame_width):
        """Describe la posición horizontal del objeto"""
        x_center = (x1 + x2) / 2
        if x_center < frame_width * 0.33:
            return "a la izquierda"
        elif x_center > frame_width * 0.66:
            return "a la derecha"
        else:
            return "en el centro"
    
    def generate_description(self, object_counts, object_details):
        """Genera una descripción completa de la escena"""
        if not object_counts:
            return "No se detectaron objetos"
        
        # Descripción general
        count_desc = ", ".join([f"{count} {obj}{'s' if count > 1 else ''}" 
                              for obj, count in object_counts.items()])
        main_desc = f"En la escena hay: {count_desc}. "
        
        # Descripción de objetos principales
        if object_details:
            main_objects = sorted(object_details, key=lambda x: x['confidence'], reverse=True)[:3]
            detail_desc = " ".join(
                [f"Hay un {obj['label']} {obj['position']}. " 
                 for obj in main_objects])
            return main_desc + detail_desc
        
        return main_desc
    
    def run_detection(self, video_source):
        """Ejecuta la detección en tiempo real"""
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("Error al abrir el video")
            return
        
        print("\n🔍 Iniciando descripción general del entorno...")
        print("Presiona Ctrl+C para detener\n")
        
        try:
            while True:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Procesar frame
                object_counts, object_details = self.describe_scene(frame)
                current_time = time.time()
                
                # Anunciar periódicamente
                if current_time - self.last_announce_time > self.announce_cooldown:
                    description = self.generate_description(object_counts, object_details)
                    print(f"\nDescripción: {description}")
                    
                    if self.engine:
                        self.engine.say(description)
                        self.engine.runAndWait()
                    
                    self.last_announce_time = current_time
                
                # Control de velocidad
                processing_time = time.time() - start_time
                wait_time = max(0.01, 1.0 - processing_time)  # ~1 FPS para descripciones claras
                time.sleep(wait_time)
                
        except KeyboardInterrupt:
            print("\nDetección detenida por el usuario")
        finally:
            cap.release()
            print("Proceso finalizado")

if __name__ == "__main__":
    detector = GeneralObjectDetector()
    
    # Para usar con video:
    video_path = "/workspaces/AppDescripcionEntorno-Angel-Eduardo-Ramirez-Oliva/Caminata.mp4"
    detector.run_detection(video_path)
    
    # Para usar con cámara web (descomenta la siguiente línea):
    # detector.run_detection(0)