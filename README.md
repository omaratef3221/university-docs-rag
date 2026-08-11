# 📘 University Documents Assistant

A retrieval-augmented generation (RAG) chatbot over University of Sharjah policy,
procedure, internship, and study-plan documents. Ask a question in plain English
(or Arabic) and get an answer grounded in the source PDFs — with the exact
document, section, and page it came from, and a rendered image of that page.

Built with **OpenAI** embeddings + chat models, **FAISS**, **BM25**, and **Streamlit**.

---

## What makes it accurate

Most RAG demos embed pages of raw PDF text and hope for the best. These documents
break that approach in specific ways, so the pipeline handles each one:

| Problem in the source PDFs | What this project does |
| --- | --- |
| Study plans are tables; linear extraction flattens them column-major into noise (`3 0405 221 / Eng. Probability`) | Detects tables with PyMuPDF and renders them **row-wise** (`0408 284 \| Water Chemistry and Analysis \| 2`) |
| An 8-page table of contents full of dotted leaders pollutes retrieval | Contents pages and dotted-leader lines are filtered out |
| Numbered headings are split across lines (`4.10` / `Graduate Assistants Policy`) | Split headings are reattached and tracked as section metadata |
| Numbered list items look identical to headings | Length/punctuation heuristics separate `2. Papers written in English…` from `4.4.4 Promotion Procedures` |
| Questions mix paraphrase with exact institutional vocabulary and course codes | **Hybrid retrieval**: dense embeddings + BM25, fused with reciprocal-rank fusion |
| Top-k similarity often surfaces passages that mention a topic without answering it | An **LLM reranker** re-orders candidates before the answer is written |
| Follow-up questions ("and for a master's?") lose their referent | Queries are **rewritten against conversation history** before searching |
| Models pad thin retrieval with general knowledge | The prompt forbids outside knowledge, requires per-claim citations, and says so when the documents don't answer |
| The same policy appears in several academic years | The model is instructed to surface the conflict and attribute each version |

Every passage carries its document, page range, and section, so citations resolve
to something a student can actually go and read.

## Features

- 💬 **Chat interface** with streaming answers and multi-turn memory
- 🔍 **Hybrid search** — semantic + keyword, with each source labelled by how it was found
- 📄 **Source viewer** — expandable passages, plus the original PDF page rendered inline
- ⬇️ **Download** any source PDF straight from the answer
- 🎛️ **Live controls** — passages per answer, reranking on/off, restrict to specific documents
- 🚫 **Refuses out-of-scope questions** instead of inventing answers

## Project layout

```
.
├── app.py                  # Streamlit chat UI
├── rag/
│   ├── config.py           # settings (env vars / Streamlit secrets)
│   ├── ingest.py           # PDF → cleaned, page-tracked, section-aware chunks
│   ├── store.py            # FAISS + BM25 hybrid index
│   ├── retrieve.py         # RRF fusion + LLM reranking
│   └── generate.py         # query rewriting + grounded answers
├── scripts/
│   ├── build_index.py      # rebuild the index from Docs/
│   └── ask.py              # ask a question from the terminal
├── Docs/                   # source PDFs
├── index/                  # prebuilt index (committed, so deploys start instantly)
└── requirements.txt
```

## Run locally

```bash
git clone https://github.com/omaratef3221/university-docs-rag.git
cd university-docs-rag

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then put your OpenAI key in .env
streamlit run app.py
```

The index is committed, so it runs immediately. To rebuild after changing `Docs/`:

```bash
python scripts/build_index.py
```

That re-chunks and re-embeds everything (~40 seconds, a few cents). Commit the
regenerated `index/` directory afterwards.

Quick check without the UI:

```bash
python scripts/ask.py "What are the rules for postponement of registration?"
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app** → pick this repo.
3. Set **Main file path** to `app.py`.
4. Open **Advanced settings → Secrets** and add:

   ```toml
   OPENAI_API_KEY = "sk-proj-..."
   ```

5. Deploy.

Because `index/` is committed, the app boots without re-embedding anything — no
build step, no API cost at startup.

> **Never commit your API key.** `.env` and `.streamlit/secrets.toml` are both in
> `.gitignore`; the deployed app reads the key from Streamlit secrets.

## Configuration

Set these as environment variables or in Streamlit secrets to override the defaults:

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | Required |
| `RAG_CHAT_MODEL` | `gpt-5.4-mini` | Model that writes answers |
| `RAG_EMBED_MODEL` | `text-embedding-3-large` | Embedding model (rebuild the index if changed) |
| `RAG_TOP_K` | `6` | Passages passed to the model |
| `RAG_RERANK` | `true` | LLM reranking of retrieved candidates |

## Adding your own documents

Drop PDFs into `Docs/`, run `python scripts/build_index.py`, and commit the new
`index/`. Nothing in the pipeline is specific to these five files — the cleaning
heuristics are tuned for numbered policy manuals but degrade gracefully on
ordinary prose PDFs.

## How a question flows through the system

```
question
   │
   ├─► rewrite against chat history ──► standalone search query
   │
   ├─► dense search (FAISS, text-embedding-3-large)  ─┐
   ├─► keyword search (BM25)                         ─┴─► reciprocal-rank fusion
   │
   ├─► LLM rerank ──► top-k passages
   │
   └─► grounded answer with [n] citations ──► source cards + PDF page images
```

## Notes and limits

- Answers are only as current as the PDFs in `Docs/` (policy manual 2023–2024,
  internship policy 2024–25, study plans 2022–23 and 2023–24).
- Scanned or image-only PDFs would need OCR; the current documents all contain
  extractable text.
- Always confirm consequential decisions against the official document — the app
  shows you the page so you can.

## License

MIT
