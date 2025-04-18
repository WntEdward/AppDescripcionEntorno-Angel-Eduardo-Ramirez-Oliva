import cv2
from ultralytics import YOLO
import pyttsx3
import time
from collections import defaultdict
import pytesseract

class GeneralObjectDetector:
    def __init__(self):
        # Cargar modelo YOLO
        self.model = YOLO("yolov8m.pt")  # Modelo más robusto

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

            voices = engine.getProperty('voices')
            if voices:
                engine.setProperty('voice', voices[0].id)

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
                if conf > 0.2:  # Umbral de confianza más bajo
                    object_counts[label] += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    object_details.append({
                        'label': label,
                        'confidence': conf,
                        'position': self.get_position_description(x1, x2, frame.shape[1]),
                        'coords': (x1, y1, x2, y2)
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

        count_desc = ", ".join([f"{count} {obj}{'s' if count > 1 else ''}" 
                              for obj, count in object_counts.items()])
        main_desc = f"En la escena hay: {count_desc}. "

        if object_details:
            main_objects = sorted(object_details, key=lambda x: x['confidence'], reverse=True)[:3]
            detail_desc = " ".join([f"Hay un {obj['label']} {obj['position']}. " 
                                    for obj in main_objects])
            return main_desc + detail_desc

        return main_desc

    def detect_obstacle_ahead(self, object_details, frame_height):
        """Detecta si hay un objeto al frente que puede representar un obstáculo"""
        for obj in object_details:
            if obj['position'] == 'en el centro':
                if obj['label'] in ['person', 'car', 'chair', 'bench', 'bicycle', 'dog']:
                    return f"Cuidado, hay un {obj['label']} justo al frente."
        return ""

    def detect_text(self, frame):
        """Detecta texto en la imagen usando OCR"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        return text.strip()

    def run_detection(self, video_source):
        """Ejecuta la detección en tiempo real"""
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("Error al abrir el video")
            return

        # Inicializa el escritor de video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('detecciones.avi', fourcc, 20.0, (640, 480))

        print("\n🔍 Iniciando descripción general del entorno...")
        print("Presiona Ctrl+C para detener\n")

        try:
            while True:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                object_counts, object_details = self.describe_scene(frame)
                current_time = time.time()

                if current_time - self.last_announce_time > self.announce_cooldown:
                    description = self.generate_description(object_counts, object_details)
                    obstacle_warning = self.detect_obstacle_ahead(object_details, frame.shape[0])
                    text_found = self.detect_text(frame)

                    full_description = f"{description} {obstacle_warning}"

                    if text_found:
                        full_description += f" También se lee: {text_found}."

                    print(f"\n🗣️ Descripción: {full_description}")

                    if self.engine:
                        self.engine.say(full_description)
                        self.engine.runAndWait()

                    self.last_announce_time = current_time

                # Escribe el frame procesado en el archivo
                out.write(frame)

                processing_time = time.time() - start_time
                wait_time = max(0.01, 1.0 - processing_time)
                time.sleep(wait_time)

        except KeyboardInterrupt:
            print("\n Detección detenida por el usuario")
        finally:
            cap.release()
            out.release()  # Libera el escritor de video
            print("Proceso finalizado")

if __name__ == "__main__":
    detector = GeneralObjectDetector()

    # Para usar con video:
    video_path = "/workspaces/AppDescripcionEntorno-Angel-Eduardo-Ramirez-Oliva/Caminata.mp4"
    detector.run_detection(video_path)
