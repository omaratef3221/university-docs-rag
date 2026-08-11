"""Hybrid index: dense FAISS vectors + a lexical BM25 index.

The dense index is built once and committed to the repository so that a
Streamlit Cloud deployment starts instantly and never re-pays the embedding
cost. BM25 is cheap to rebuild, so it is reconstructed from the chunk store at
load time rather than pickled.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi

from . import config
from .ingest import Chunk, load_chunks, save_chunks

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer used by the BM25 index."""
    return _TOKEN_RE.findall(text.lower())


def embed_texts(
    client: OpenAI,
    texts: list[str],
    *,
    batch_size: int = 64,
    progress=None,
) -> np.ndarray:
    """Embed texts and return L2-normalised float32 vectors."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        for attempt in range(5):
            try:
                resp = client.embeddings.create(model=config.EMBED_MODEL, input=batch)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        vectors.extend(item.embedding for item in resp.data)
        if progress:
            progress(min(start + batch_size, len(texts)), len(texts))
    arr = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(arr)
    return arr


@dataclass
class HybridIndex:
    """Loaded index: FAISS vectors, BM25, and the chunks they point at."""

    chunks: list[Chunk]
    faiss_index: faiss.Index
    bm25: BM25Okapi
    meta: dict

    @property
    def sources(self) -> list[str]:
        seen: dict[str, None] = {}
        for c in self.chunks:
            seen.setdefault(c.title, None)
        return list(seen)

    def dense_search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        scores, ids = self.faiss_index.search(query_vec.reshape(1, -1), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def lexical_search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]


def build_index(
    docs_dir: Path | None = None,
    index_dir: Path | None = None,
    progress=None,
) -> HybridIndex:
    """Chunk the PDFs, embed them, and write the index to disk."""
    from .ingest import build_chunks

    index_dir = index_dir or config.INDEX_DIR
    index_dir.mkdir(parents=True, exist_ok=True)

    print("Chunking documents...")
    chunks = build_chunks(docs_dir)
    print(f"Embedding {len(chunks)} chunks with {config.EMBED_MODEL}...")

    api_key = config.get_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=api_key)

    vectors = embed_texts(client, [c.embed_text for c in chunks], progress=progress)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(index_dir / config.FAISS_FILE))
    save_chunks(chunks, index_dir / config.CHUNKS_FILE)
    meta = {
        "embed_model": config.EMBED_MODEL,
        "dim": int(vectors.shape[1]),
        "n_chunks": len(chunks),
        "chunk_tokens": config.CHUNK_TOKENS,
        "chunk_overlap": config.CHUNK_OVERLAP_TOKENS,
        "documents": sorted({c.source for c in chunks}),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    (index_dir / config.META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote index to {index_dir}")

    bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
    return HybridIndex(chunks=chunks, faiss_index=index, bm25=bm25, meta=meta)


def load_index(index_dir: Path | None = None) -> HybridIndex:
    """Load a previously built index from disk."""
    index_dir = index_dir or config.INDEX_DIR
    faiss_path = index_dir / config.FAISS_FILE
    chunks_path = index_dir / config.CHUNKS_FILE
    if not faiss_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            f"No index found in {index_dir}. Run `python scripts/build_index.py` first."
        )
    chunks = load_chunks(chunks_path)
    index = faiss.read_index(str(faiss_path))
    meta_path = index_dir / config.META_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if index.ntotal != len(chunks):
        raise RuntimeError(
            f"Index/chunk mismatch: {index.ntotal} vectors vs {len(chunks)} chunks. Rebuild."
        )
    bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
    return HybridIndex(chunks=chunks, faiss_index=index, bm25=bm25, meta=meta)
