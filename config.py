import os

# ── API Keys ───────────────────────────────────────────────────────
# Get free keys from:
#   Groq    → https://console.groq.com
#   Sarvam  → https://dashboard.sarvam.ai
#   Pinecone→ https://app.pinecone.io



import os
import streamlit as st

def get_secret(key):
    try:
        return st.secrets[key]   # Streamlit cloud
    except:
        return os.getenv(key)    # Local (.env)

GROQ_API_KEY     = get_secret("GROQ_API_KEY")
SARVAM_API_KEY   = get_secret("SARVAM_API_KEY")
PINECONE_API_KEY = get_secret("PINECONE_API_KEY")
LANGSMITH_API_KEY =os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = "railway-bot-project"

# LangSmith requires these as environment variables — set them immediately
os.environ["LANGCHAIN_TRACING_V2"]  = "true"
os.environ["LANGCHAIN_ENDPOINT"]    = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]     = LANGSMITH_API_KEY
os.environ["LANGCHAIN_PROJECT"]     = LANGSMITH_PROJECT

# ── Pinecone Settings ──────────────────────────────────────────────
PINECONE_INDEX  = "railway-bot"
PINECONE_CLOUD  = "aws"
PINECONE_REGION = "ap-south-1 (Mumbai)"

# ── Local Paths ────────────────────────────────────────────────────
DATA_DIR        = "./data"           # put your PDFs here
PROCESSED_FILE  = "./processed.csv"  # tracks which PDFs were already ingested

# ── Model Settings ─────────────────────────────────────────────────
CHUNK_SIZE  = 500
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K       = 3
GROQ_MODEL  = "llama-3.1-8b-instant"
