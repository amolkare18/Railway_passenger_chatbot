import os, hashlib
import pandas as pd
import PyPDF2
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

import sys; sys.path.append("..")
from config import (
    DATA_DIR, PROCESSED_FILE, CHUNK_SIZE, EMBED_MODEL,
    PINECONE_API_KEY, PINECONE_INDEX, PINECONE_CLOUD, PINECONE_REGION
)


# ── Step 1: Extract text from PDF ─────────────────────────────────
def pdf_to_text(path):
    text = ""
    with open(path, "rb") as f:
        for page in PyPDF2.PdfReader(f).pages:
            text += (page.extract_text() or "") + "\n"
    return text


# ── Step 2: Split text into chunks ────────────────────────────────
def chunk_text(text):
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i : i + CHUNK_SIZE])
        i += CHUNK_SIZE - 50          # 50-char overlap
    return [c.strip() for c in chunks if c.strip()]


# ── Step 3: Track processed files using a simple CSV ──────────────
def get_processed_hashes():
    if not os.path.exists(PROCESSED_FILE):
        return set()
    df = pd.read_csv(PROCESSED_FILE)
    return set(df["doc_id"].tolist())


def save_processed_hash(doc_id, filename):
    row = pd.DataFrame([{"doc_id": doc_id, "filename": filename}])
    # Append to CSV (create if doesn't exist)
    if os.path.exists(PROCESSED_FILE):
        row.to_csv(PROCESSED_FILE, mode="a", header=False, index=False)
    else:
        row.to_csv(PROCESSED_FILE, index=False)


# ── Step 4: Connect to Pinecone ────────────────────────────────────
def get_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX not in existing:
        print(f"[ingest] Creating Pinecone index: {PINECONE_INDEX}")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=384,              # all-MiniLM-L6-v2 produces 384-dim vectors
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
        )
    return pc.Index(PINECONE_INDEX)


# ── Step 5: Push chunks to Pinecone ───────────────────────────────
def push_to_pinecone(index, rows, model):
    texts   = [r["content"] for r in rows]
    vectors = model.encode(texts, show_progress_bar=True).tolist()

    upserts = [
        {
            "id":     f"{r['doc_id']}-{r['chunk_index']}",
            "values": vec,
            "metadata": {
                "text":     r["content"],
                "filename": r["filename"],
            }
        }
        for r, vec in zip(rows, vectors)
    ]

    # Pinecone recommends batches of 100
    for i in range(0, len(upserts), 100):
        index.upsert(vectors=upserts[i : i + 100])
        print(f"[ingest] Upserted batch {i // 100 + 1}")

    print(f"[ingest] ✅ Pushed {len(upserts)} vectors to Pinecone.")


# ── Main ingestion function ────────────────────────────────────────
def run_ingestion():
    os.makedirs(DATA_DIR, exist_ok=True)

    pdf_files = list(Path(DATA_DIR).glob("*.pdf"))
    if not pdf_files:
        print(f"[ingest] No PDFs found in '{DATA_DIR}'. Add PDFs and try again.")
        return

    done     = get_processed_hashes()
    model    = SentenceTransformer(EMBED_MODEL)
    pc_index = get_pinecone_index()
    rows     = []

    for pdf in pdf_files:
        doc_id = hashlib.md5(pdf.read_bytes()).hexdigest()

        if doc_id in done:
            print(f"[ingest] Skipping (already done): {pdf.name}")
            continue

        print(f"[ingest] Processing: {pdf.name}")
        text = pdf_to_text(str(pdf))

        for i, chunk in enumerate(chunk_text(text)):
            rows.append({
                "doc_id":      doc_id,
                "filename":    pdf.name,
                "content":     chunk,
                "chunk_index": i,
            })

        save_processed_hash(doc_id, pdf.name)

    if rows:
        push_to_pinecone(pc_index, rows, model)
    else:
        print("[ingest] No new PDFs to process.")

    print("[ingest] Done.")


if __name__ == "__main__":
    run_ingestion()
