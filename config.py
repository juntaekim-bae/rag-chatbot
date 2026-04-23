import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
ACCESS_PASSWORD = os.getenv("ACCESS_PASSWORD", "0000")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "20220405")
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", "./documents"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "./chroma_db"))
COLLECTION_NAME = "documents"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "8"))
MODEL = os.getenv("MODEL", "claude-sonnet-4-6")

DOCUMENTS_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)
