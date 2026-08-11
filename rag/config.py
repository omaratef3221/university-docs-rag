"""Central configuration.

Reads settings from (in order of precedence): Streamlit secrets, environment
variables, then the defaults below. This lets the same code run locally with a
`.env` file and on Streamlit Community Cloud with `st.secrets`.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "Docs"
INDEX_DIR = ROOT / "index"

# --- Models -----------------------------------------------------------------
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072
CHAT_MODEL = "gpt-5.4-mini"
# Small, cheap model used for query rewriting and reranking.
UTILITY_MODEL = "gpt-5.4-mini"

# --- Chunking ---------------------------------------------------------------
CHUNK_TOKENS = 700
CHUNK_OVERLAP_TOKENS = 120
MIN_CHUNK_TOKENS = 40

# --- Retrieval --------------------------------------------------------------
CANDIDATES_PER_RETRIEVER = 30  # dense and BM25 each fetch this many
RRF_K = 60  # reciprocal-rank-fusion smoothing constant
TOP_K = 6  # passages finally shown to the chat model
RERANK = True  # LLM reranking of fused candidates

# --- Index file names -------------------------------------------------------
FAISS_FILE = "faiss.index"
CHUNKS_FILE = "chunks.jsonl"
META_FILE = "meta.json"


def get_api_key() -> str | None:
    """Return the OpenAI API key from Streamlit secrets or the environment."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key.strip()
    try:  # only available when running inside Streamlit
        import streamlit as st

        secret = st.secrets.get("OPENAI_API_KEY")
        if secret:
            return str(secret).strip()
    except Exception:
        pass
    return None


def _setting(name: str, default):
    """Look up an override from the environment, falling back to `default`."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        try:
            return int(raw)
        except ValueError:
            return default
    return raw


CHAT_MODEL = _setting("RAG_CHAT_MODEL", CHAT_MODEL)
EMBED_MODEL = _setting("RAG_EMBED_MODEL", EMBED_MODEL)
TOP_K = _setting("RAG_TOP_K", TOP_K)
RERANK = _setting("RAG_RERANK", RERANK)
