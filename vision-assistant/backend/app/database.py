# backend/database.py
from dotenv import load_dotenv
from pymongo import MongoClient
import os

load_dotenv()  # Carga el .env
MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client.get_database("EcoVisual")  #
print("✅ Conectado a MongoDB")