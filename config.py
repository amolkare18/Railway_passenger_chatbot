import os

# ── API Keys ───────────────────────────────────────────────────────
# Get free keys from:
#   Groq    → https://console.groq.com
#   Sarvam  → https://dashboard.sarvam.ai
#   Pinecone→ https://app.pinecone.io



GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
SARVAM_API_KEY   = os.getenv("SARVAM_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ── Pinecone Settings ──────────────────────────────────────────────
PINECONE_INDEX  = "railway-bot"
PINECONE_CLOUD  = "aws"
PINECONE_REGION = "us-east-1"

# ── Local Paths ────────────────────────────────────────────────────
DATA_DIR        = "./data"           # put your PDFs here
PROCESSED_FILE  = "./processed.csv"  # tracks which PDFs were already ingested

# ── Model Settings ─────────────────────────────────────────────────
CHUNK_SIZE  = 500
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K       = 3
GROQ_MODEL  = "llama-3.1-8b-instant"
