import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", "./documents"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "./chroma_db"))
COLLECTION_NAME = "documents"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "5"))
MODEL = os.getenv("MODEL", "llama-3.3-70b-versatile")

DOCUMENTS_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)
