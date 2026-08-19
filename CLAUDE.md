# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
source /venv/main/bin/activate       # this instance; deps are already installed

streamlit run app.py                 # the chat UI
python scripts/build_index.py        # re-chunk + re-embed Docs/ into index/  (~2 min, a few cents)
python scripts/ask.py "question"     # answer one question in the terminal — fastest smoke test
python scripts/scrape_site.py        # refresh Docs/web/ from sharjah.ac.ae (then rebuild)
```

PyMuPDF writes `Consider using the pymupdf_layout package` to stderr on most
pages; pipe through `grep -v pymupdf_layout` to keep output readable.

## Verification

There is no test suite. Changes are verified two ways:

- **Retrieval/answer quality** — `python scripts/ask.py "..."` and read the cited
  sources. Useful probes: a policy that appears in several academic years
  (postponement of registration — the answer should surface the conflict), a
  study-plan table question (credit hours / course codes), and an out-of-scope
  question, which must be refused rather than answered from general knowledge.
- **The UI** — Streamlit's own harness runs `app.py` headlessly and surfaces
  exceptions that a browser would hide:

  ```python
  from streamlit.testing.v1 import AppTest
  at = AppTest.from_file("app.py", default_timeout=180).run()
  assert not at.exception
  at.chat_input[0].set_value("What is the minimum GPA required to graduate?").run()
  print(at.session_state["messages"][-1]["content"])
  ```

## Architecture

Question flow: `rewrite_query` (resolves follow-ups against chat history) →
dense FAISS + BM25 searches → reciprocal-rank fusion → `llm_rerank` → `answer_stream`
(grounded answer with `[n]` citations) → `app.py` renders source cards and the
original PDF page as an image.

`rag/ingest.py` → `rag/store.py` → `rag/retrieve.py` → `rag/generate.py`, with
`app.py` as the only UI layer and `rag/config.py` holding all tunables.

### Three source types, one chunk shape

`build_chunks` ingests `Docs/*.pdf` (policy manuals, study plans — PyMuPDF),
`Docs/*.docx` (course syllabi — python-docx), and `Docs/web/*.md` (pages scraped
from sharjah.ac.ae by `scripts/scrape_site.py`). DOCX and web chunks have
`page_start == page_end == 0`; `Chunk.pages()`/`citation()` and the UI omit page
numbers for them, and web chunks carry the source `url` (shown as a link — there
is no page preview or download for web sources). `Chunk.source` is the path
*relative to `Docs/`* (`web/<slug>.md` for scraped pages), so keep using it with
`config.DOCS_DIR / source`.

The syllabi are Word tables whose merged cells repeat text into every spanned
column — `_docx_table_rows` collapses consecutive duplicates; don't "simplify"
that away. Syllabus sections come from short colon-terminated label paragraphs
("Course Learning Outcomes:"), not numbered headings.

### The web crawl is deliberately capped

The sitemap has ~14,600 URLs; embedding all of it would make the committed FAISS
index several hundred MB (GitHub refuses files > 100 MB, Streamlit Cloud runs out
of memory). `scrape_site.py` crawls a priority-ordered English subset (default
cap 350 pages, seeds + Admissions/Degree/Eng/Student-Life first, news/events/staff
profiles excluded). Raise `--max-pages` only while watching `index/faiss.index`
size.

### `index/` is a committed build artifact

The FAISS index and chunk store are checked in so Streamlit Cloud deploys boot
without re-embedding. **Anything that changes chunk content or the embedding
space invalidates them**: editing `Docs/` (including re-scraping `Docs/web/`),
the chunking constants in `rag/config.py`, the heuristics in `rag/ingest.py`, or
`EMBED_MODEL`. After such a change, re-run `scripts/build_index.py` and commit
the regenerated `index/`.

`load_index()` catches a vector/chunk count mismatch, but **cannot detect a
changed embedding model** — that failure is silent and shows up only as bad
retrieval. `index/meta.json` records the model the index was built with; check it.

Two related invariants: a chunk's `id` equals its FAISS row index (`retrieve.py`
maps search results back via `index.chunks[idx]`), so chunk order must match
vector order; and `Chunk.embed_text` is derived at build time and deliberately
**not** persisted — `save_chunks` strips it, `load_chunks` leaves it empty. It is
the embedding input only; everything at query time uses `Chunk.text`.

### Ingestion heuristics are load-bearing

`rag/ingest.py` is tuned to these specific PDFs, and each rule fixes a concrete
failure — see the table in `README.md`. The ones most likely to be "simplified"
into a regression:

- Pages are read as **positioned blocks with tables rendered row-wise**. Plain
  `page.get_text()` flattens the study plans column-major into unusable noise, so
  course codes and credit hours stop retrieving.
- `_is_heading` separates real headings (`4.4.4 Promotion Procedures`) from
  numbered list items (`2. Papers written in English Language which...`) using
  length and terminal punctuation. Loosening it fills `Chunk.section` — which is
  shown in citations — with sentence fragments.
- Contents pages and dotted-leader lines are dropped; split headings (number on
  its own line) are reattached.

### Retrieval and generation

Retrieval is hybrid because the questions are: paraphrase ("how long can I defer
my studies?") needs embeddings, exact institutional vocabulary and course codes
("0408 284") need BM25. Neither alone is sufficient — keep both arms.

`llm_rerank` and `rewrite_query` **fail open**: on any API error they fall back to
the fused order or the raw question. Reranking also backfills from the fused list
so the model can never starve an answer of context. Keep them non-fatal.

The `SYSTEM_PROMPT` in `rag/generate.py` is what stops the model padding thin
retrieval with general knowledge about other universities. It also drives the
per-claim citation format the UI parses, and the instruction to attribute
conflicting policy versions to their source documents.

### Models

`gpt-5.x` models reject `max_tokens` — use `max_completion_tokens`. They accept
`temperature` and `reasoning_effort`. `UTILITY_MODEL` (rewriting, reranking) is
separate from `CHAT_MODEL` so the cheap paths can be downgraded independently.

### API key

Server-side only: `config.get_api_key()` reads `OPENAI_API_KEY` from the
environment, then Streamlit secrets. `app.py` stops with a configuration error if
neither is set. Do not reintroduce a UI field asking the visitor for a key.
