import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import pytesseract
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Cargar CSV con información de rotación
csv_path = "image_ids_and_rotation.csv"  # Cambia por la ruta de tu CSV
df = pd.read_csv(csv_path)  # Leer el archivo CSV

# Configurar Tesseract (para Windows, cambia la ruta)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

@app.route('/')
def home():
    return "Servidor funcionando"

@app.route('/procesar-imagen', methods=['POST'])
def procesar_imagen():
    try:
        # Obtener ID de la imagen
        image_id = request.form.get("image_id")
        file = request.files["imagen"]

        # Cargar imagen en formato PIL
        img = Image.open(io.BytesIO(file.read()))
        img = np.array(img)

        # Buscar la rotación correspondiente en el CSV
        rotation = df.loc[df["image_id"] == int(image_id), "rotation"].values
        if len(rotation) > 0:
            rotation_angle = rotation[0]  # Obtener el ángulo
            if rotation_angle != 0:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE if rotation_angle == 90 else
                                      cv2.ROTATE_180 if rotation_angle == 180 else
                                      cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Convertir a escala de grises y aplicar OCR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        texto = pytesseract.image_to_string(gray, lang="eng")

        return jsonify({"image_id": image_id, "texto_detectado": texto})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
