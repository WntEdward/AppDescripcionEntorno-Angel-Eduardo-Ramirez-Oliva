import requests
from pathlib import Path

# Configuración
IMAGE_PATH = Path("/workspaces/AppDescripcionEntorno-Angel-Eduardo-Ramirez-Oliva/Imagenes para la prueba/bicicleta.png") 
API_URL = "http://localhost:8000/api/v1/detect"

def run_test():
    try:
        # Verificar que la imagen existe
        if not IMAGE_PATH.exists():
            print(f"Error: No se encontró {IMAGE_PATH}")
            return

        # Enviar la imagen al servidor
        with open(IMAGE_PATH, 'rb') as img_file:
            response = requests.post(
                API_URL,
                files={'file': (IMAGE_PATH.name, img_file, 'image/jpeg')},
                timeout=10
            )

        # Mostrar resultados
        print("\n=== Resultado de la Prueba ===")
        print(f"Status Code: {response.status_code}")
        
        try:
            print("Respuesta JSON:", response.json())
        except ValueError:
            print("Respuesta del servidor:", response.text)

    except requests.exceptions.RequestException as e:
        print(f"\nError de conexión: {str(e)}")
    except Exception as e:
        print(f"\nError inesperado: {str(e)}")

if __name__ == "__main__":
    run_test()