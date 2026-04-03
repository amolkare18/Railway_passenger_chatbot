import streamlit as st
from app.ingest import run_ingestion
from app.rag import load_pinecone, answer_query

st.set_page_config(page_title="🚆 Railway Rights Bot", page_icon="🚆")
st.title("🚆 Indian Railway Passenger Rights Chatbot")
st.caption("Ask in Hindi, Tamil, Telugu, Bengali, or any Indian language!")
st.write("Key exists:", "PINECONE_API_KEY" in st.secrets)

@st.cache_resource(show_spinner="Loading... processing PDFs and connecting to Pinecone...")
def startup():
    run_ingestion()   # processes only new PDFs, skips already done
    load_pinecone()   # connect to Pinecone cloud


startup()


# ── Language selector ─────────────────────────────────────────────
lang_options = {
    "Auto-detect": None,   "English":  "en-IN",
    "Hindi":       "hi-IN", "Tamil":   "ta-IN",
    "Telugu":      "te-IN", "Bengali": "bn-IN",
    "Marathi":     "mr-IN", "Gujarati":"gu-IN",
    "Kannada":     "kn-IN",
}
lang_choice = st.selectbox("Response Language", list(lang_options.keys()))


# ── Chat history ──────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)


# ── Chat input ────────────────────────────────────────────────────
if query := st.chat_input("Type your question here..."):
    st.chat_message("user").write(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching railway rules..."):
            result = answer_query(query, force_lang=lang_options[lang_choice])

        st.write(result["answer"])

        # with st.expander("🔍 See details"):
        #     st.write(f"**Detected language:** `{result['lang']}`")
        #     st.write(f"**English query:** {result['query_en']}")
        #     for i, c in enumerate(result["chunks"], 1):
        #         st.text_area(f"Chunk {i}", c, height=80, key=f"c{i}{query[:5]}")

    st.session_state.history += [("user", query), ("assistant", result["answer"])]
