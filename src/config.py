"""Central configuration. Everything tunable lives here, sourced from env vars
so the same code runs locally, in Docker, and in CI without edits."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Models ---
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- Vector store (Qdrant, local file-mode — no server / cloud account needed) ---
QDRANT_PATH = os.getenv("QDRANT_PATH", str(BASE_DIR / "data" / "qdrant_storage"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "clinical_guidelines")

# --- Ingestion ---
GUIDELINES_DIR = os.getenv("GUIDELINES_DIR", str(BASE_DIR / "data" / "guidelines"))
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "")

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))

# --- Retrieval ---
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", 5))

# --- Guardrails ---
MIN_RETRIEVAL_SCORE = float(os.getenv("MIN_RETRIEVAL_SCORE", 0.72))
MAX_QUERY_CHARS = int(os.getenv("MAX_QUERY_CHARS", 500))
ABSTAIN_MESSAGE = (
    "I couldn't find a clinical guideline in the indexed knowledge base that "
    "directly addresses this question. Please consult a licensed clinician "
    "or a primary guideline source rather than relying on an unsupported answer."
)

# --- Eval ---
EVAL_DATASET_PATH = str(BASE_DIR / "src" / "evaluation" / "eval_dataset.json")
EVAL_REPORT_PATH = str(BASE_DIR / "src" / "evaluation" / "eval_report.json")
