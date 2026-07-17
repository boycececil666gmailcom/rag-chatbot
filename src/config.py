import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

import sys

# Gemini settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY and "pytest" in sys.modules:
    GEMINI_API_KEY = "dummy_key_for_testing"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.0"))

# FastAPI server settings
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")

# Chroma DB settings
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
if not os.path.isabs(CHROMA_PERSIST_DIR):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CHROMA_PERSIST_DIR = os.path.abspath(os.path.join(project_root, CHROMA_PERSIST_DIR))
