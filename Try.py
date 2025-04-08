import requests
from pathlib import Path

image_path = Path("/workspaces/AppDescripcionEntorno-Angel-Eduardo-Ramirez-Oliva/perro2.png")
url = "http://localhost:8000/api/v1/detect"

def test_detection():
    try:
        # Verificar que la imagen existe
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        # Verificar que es un archivo de imagen
        if image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            raise ValueError("File is not a supported image format")
        
        with open(image_path, 'rb') as img_file:
            files = {'file': (image_path.name, img_file, 'image/jpeg')}
            headers = {}
            
            response = requests.post(url, files=files, headers=headers)
            
            print(f"\nStatus Code: {response.status_code}")
            print("Headers:", response.headers)
            
            try:
                print("JSON Response:", response.json())
            except ValueError:
                print("Raw Response:", response.text)
                
    except Exception as e:
        print(f"\nError during detection: {str(e)}")
        if hasattr(e, 'response') and e.response:
            print("Server response:", e.response.text)

if __name__ == "__main__":
    test_detection()
