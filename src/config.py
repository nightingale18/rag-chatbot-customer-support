"""
config.py – Central settings for the RAG chatbot.
All parameters are gathered here for easy tuning and transparency.
"""

from pathlib import Path

# --- File Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "data" / "knowledge_base"
CHROMA_DB_DIR = BASE_DIR / "data" / "chroma_db"

# --- Model Selection ---
# Llama 3.2 (3B) is the sweet spot: small enough for CPU/laptop RAM,
# but smart enough to follow instructions and stay grounded.
CHAT_MODEL = "llama3.2"

# Nomic is lightweight (274MB) and consistently beats larger models
# on semantic search benchmarks (MTEB).
EMBEDDING_MODEL = "nomic-embed-text"

# --- Text Chunking ---
# 400 chars (approx 1 paragraph) keeps retrieval precise.
# Overlap prevents losing context at the edges of a cut.
CHUNK_SIZE = 400
CHUNK_OVERLAP = 60

# Separators prioritized to keep headers and paragraphs together.
CHUNK_SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", " ", ""]

# --- Retrieval Settings ---
# Top-K = 3 balances enough info for the LLM without hitting
# context window limits or introducing too much "noise."
DEFAULT_TOP_K = 3
MIN_TOP_K = 1
MAX_TOP_K = 8

# Versioning the collection name forces a fresh store if we re-index.
COLLECTION_NAME = "kb_v1_03_2026"

# --- LLM Generation ---
# Temperature 0 ensures factual, repeatable answers.
# Creativity is a bug, not a feature, for customer support.
LLM_TEMPERATURE = 0

# --- System Prompt ---
# Constraints: No hallucinations, admit ignorance, and always cite sources.
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
