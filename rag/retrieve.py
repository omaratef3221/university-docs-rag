"""Hybrid retrieval: dense + BM25, fused with RRF, then LLM-reranked.

Policy questions mix paraphrase ("how long can I defer my studies?") with exact
institutional vocabulary ("Postponement of Registration", course code "0408 284").
Dense embeddings handle the first, BM25 the second, so the two are run in
parallel and fused rather than picking one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from openai import OpenAI

from . import config
from .ingest import Chunk
from .store import HybridIndex, embed_texts


@dataclass
class Hit:
    """A retrieved chunk with its provenance and score."""

    chunk: Chunk
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None

    @property
    def retrievers(self) -> str:
        parts = []
        if self.dense_rank is not None:
            parts.append("semantic")
        if self.lexical_rank is not None:
            parts.append("keyword")
        return "+".join(parts) or "none"


def _rrf(
    dense: list[tuple[int, float]],
    lexical: list[tuple[int, float]],
    k: int = config.RRF_K,
) -> dict[int, dict]:
    """Reciprocal-rank fusion of two ranked candidate lists."""
    fused: dict[int, dict] = {}
    for rank, (idx, _) in enumerate(dense):
        entry = fused.setdefault(idx, {"score": 0.0, "dense": None, "lexical": None})
        entry["score"] += 1.0 / (k + rank + 1)
        entry["dense"] = rank + 1
    for rank, (idx, _) in enumerate(lexical):
        entry = fused.setdefault(idx, {"score": 0.0, "dense": None, "lexical": None})
        entry["score"] += 1.0 / (k + rank + 1)
        entry["lexical"] = rank + 1
    return fused


_RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant_ids": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Candidate numbers, most relevant first.",
        }
    },
    "required": ["relevant_ids"],
    "additionalProperties": False,
}


def llm_rerank(client: OpenAI, question: str, hits: list[Hit], top_k: int) -> list[Hit]:
    """Reorder candidates by asking a model which actually answer the question.

    Falls back to the fused order if the model call fails — reranking is a
    quality improvement, never a hard dependency.
    """
    if len(hits) <= top_k:
        return hits
    listing = "\n\n".join(
        f"[{i}] {h.chunk.title}"
        + (f" — {h.chunk.section}" if h.chunk.section else "")
        + f"\n{h.chunk.text[:700]}"
        for i, h in enumerate(hits)
    )
    prompt = (
        f"Question: {question}\n\n"
        f"Candidate passages:\n{listing}\n\n"
        f"Return the numbers of up to {top_k} passages that contain information needed to "
        "answer the question, ordered most relevant first. Omit passages that merely "
        "mention the topic without answering it. If none are relevant, return an empty list."
    )
    try:
        resp = client.chat.completions.create(
            model=config.UTILITY_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You rank retrieved passages for a document question-answering system.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "rerank", "strict": True, "schema": _RERANK_SCHEMA},
            },
            reasoning_effort="low",
        )
        ids = json.loads(resp.choices[0].message.content)["relevant_ids"]
    except Exception:
        return hits[:top_k]

    seen: set[int] = set()
    ordered: list[Hit] = []
    for i in ids:
        if isinstance(i, int) and 0 <= i < len(hits) and i not in seen:
            seen.add(i)
            ordered.append(hits[i])
    if not ordered:  # model rejected everything; keep the top fused hits
        return hits[:top_k]
    # Backfill so the model can never starve the answer of context.
    for i, h in enumerate(hits):
        if len(ordered) >= top_k:
            break
        if i not in seen:
            ordered.append(h)
    return ordered[:top_k]


def retrieve(
    client: OpenAI,
    index: HybridIndex,
    question: str,
    *,
    top_k: int = config.TOP_K,
    candidates: int = config.CANDIDATES_PER_RETRIEVER,
    rerank: bool = config.RERANK,
    sources: list[str] | None = None,
) -> list[Hit]:
    """Return the best `top_k` passages for `question`."""
    query_vec = embed_texts(client, [question])[0]
    # Over-fetch when filtering so the filter cannot empty the result set.
    fetch = candidates * 3 if sources else candidates
    fetch = min(fetch, len(index.chunks))

    dense = index.dense_search(query_vec, fetch)
    lexical = index.lexical_search(question, fetch)

    if sources:
        allowed = set(sources)
        dense = [(i, s) for i, s in dense if index.chunks[i].title in allowed][:candidates]
        lexical = [(i, s) for i, s in lexical if index.chunks[i].title in allowed][:candidates]

    fused = _rrf(dense, lexical)
    hits = [
        Hit(
            chunk=index.chunks[idx],
            score=info["score"],
            dense_rank=info["dense"],
            lexical_rank=info["lexical"],
        )
        for idx, info in sorted(fused.items(), key=lambda kv: kv[1]["score"], reverse=True)
    ]
    shortlist = hits[: max(top_k * 4, 20)]
    if rerank and len(shortlist) > top_k:
        return llm_rerank(client, question, shortlist, top_k)
    return shortlist[:top_k]
