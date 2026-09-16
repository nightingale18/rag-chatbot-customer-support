from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "data" / "knowledge_base"
CHROMA_DB_DIR = BASE_DIR / "data" / "chroma_db"

CHAT_MODEL = "llama3.2"

EMBEDDING_MODEL = "nomic-embed-text"


ENSEMBLE_WEIGHTS = [0.4, 0.6]


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


CHUNK_SIZE = 400
CHUNK_OVERLAP = 60


CHUNK_SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", " ", ""]


DEFAULT_TOP_K = 3
MIN_TOP_K = 1
MAX_TOP_K = 8


COLLECTION_NAME = "kb_v1_03_2026"


LLM_TEMPERATURE = 0


SYSTEM_PROMPT = """\
You are a professional customer support assistant for an automotive company.
Use ONLY the provided Context to answer. 

Rules:
1. If the answer is in the context, be clear and helpful.
2. If it is NOT in the context, say: "I'm sorry, I don't have that information. 
   Please contact support at support@bmw.com or call 089 125016000."
3. Never use outside knowledge.
4. Always end with a "Sources:" section listing the filenames used.

Context:
{context}
"""
