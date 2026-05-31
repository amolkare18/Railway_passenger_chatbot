import hashlib
import json

import redis
import requests
from groq import Groq
from langdetect import detect
from langsmith import traceable
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

import sys; sys.path.append("..")
from config import (
    GROQ_API_KEY, SARVAM_API_KEY,
    PINECONE_API_KEY, PINECONE_INDEX,
    TOP_K, EMBED_MODEL, GROQ_MODEL,
    REDIS_URL
)


# ── Language map: langdetect code → Sarvam code ───────────────────
LANG_MAP = {
    "hi": "hi-IN", "bn": "bn-IN", "ta": "ta-IN", "te": "te-IN",
    "mr": "mr-IN", "gu": "gu-IN", "kn": "kn-IN", "ml": "ml-IN",
    "pa": "pa-IN", "or": "od-IN", "en": "en-IN",
}


def detect_lang(text):
    try:
        return LANG_MAP.get(detect(text), "en-IN")
    except:
        return "en-IN"


def translate(text, source_lang, target_lang):
    if source_lang == target_lang:
        return text
    res = requests.post(
        "https://api.sarvam.ai/translate",
        headers={"api-subscription-key": SARVAM_API_KEY},
        json={
            "input": text,
            "source_language_code": source_lang,
            "target_language_code": target_lang,
            "mode": "formal"
        },
    )
    return res.json().get("translated_text", text)


# ── Load once into memory ─────────────────────────────────────────
_pc_index = None
_model    = None
_redis    = redis.from_url(REDIS_URL, decode_responses=True)


def answer_en_cache_key(query_en):
    normalized = query_en.strip().lower()
    query_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"answer_en:v1:{query_hash}"


def get_cached_answer_en(query_en):
    try:
        cached = _redis.get(answer_en_cache_key(query_en))
        return json.loads(cached) if cached else None
    except redis.RedisError as exc:
        print(f"[redis] Cache read failed: {exc}")
        return None


def cache_answer_en(query_en, answer_en, chunks):
    payload = {"answer_en": answer_en, "chunks": chunks}
    try:
        _redis.setex(answer_en_cache_key(query_en), 60 * 60 * 24, json.dumps(payload))
    except redis.RedisError as exc:
        print(f"[redis] Cache write failed: {exc}")


def load_pinecone():
    global _pc_index, _model
    pc        = Pinecone(api_key=PINECONE_API_KEY)
    _pc_index = pc.Index(PINECONE_INDEX)
    _model    = SentenceTransformer(EMBED_MODEL)
    stats     = _pc_index.describe_index_stats()
    print(f"[rag] Connected to Pinecone. Total vectors: {stats['total_vector_count']}")


@traceable(name="retrieve", metadata={"component": "retriever"})
def retrieve(query_en):
    
    global _model, _pc_index

    if _model is None or _pc_index is None:
        load_pinecone()

    vec     = _model.encode([query_en]).tolist()
    results = _pc_index.query(vector=vec, top_k=TOP_K, include_metadata=True)
    return [m["metadata"]["text"] for m in results["matches"]]


@traceable(name="generate", metadata={"component": "llm"})
def generate(query_en, chunks):
    context = "\n\n".join(chunks)
    client  = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content":
                "You are a helpful assistant for Indian Railway passengers. "
                "Answer ONLY using the provided context. "
                "If the answer is not in the context, say 'I don't have that information.'"},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query_en}"},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()


@traceable(name="answer_query")
def answer_query(user_query, force_lang=None):
    lang      = force_lang or detect_lang(user_query)   # detect language
    query_en  = translate(user_query, lang, "en-IN")    # translate → English

    cached = get_cached_answer_en(query_en)
    if cached:
        answer_en = cached["answer_en"]
        chunks = cached["chunks"]
    else:
        chunks    = retrieve(query_en)                   # search Pinecone
        answer_en = generate(query_en, chunks)           # Groq LLM
        cache_answer_en(query_en, answer_en, chunks)

    answer    = translate(answer_en, "en-IN", lang)      # translate back
    return {
        "lang":     lang,
        "query_en": query_en,
        "answer":   answer,
        "context":  "\n\n".join(chunks),   # for LangSmith evaluators
        "chunks":   chunks,
    }
