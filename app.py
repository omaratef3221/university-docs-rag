"""Streamlit chat UI for asking questions about the university documents.

Run locally with:  streamlit run app.py
"""

from __future__ import annotations

import re

import pymupdf
import streamlit as st
from openai import OpenAI

from rag import config
from rag.generate import answer_stream, rewrite_query
from rag.retrieve import retrieve
from rag.store import load_index

st.set_page_config(
    page_title="CWDE Program Virtual Course Advisor",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
    .stChatMessage { padding-top: 0.25rem; }
    .src-card {
        border: 1px solid rgba(49,51,63,.15);
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
        background: var(--secondary-background-color);
    }
    .src-num {
        display: inline-block;
        min-width: 1.5rem;
        text-align: center;
        background: #1f6feb;
        color: #fff;
        border-radius: 4px;
        font-weight: 600;
        margin-right: .4rem;
    }
    .src-meta { color: #5b6472; font-size: .85rem; }
    .cite { color: #1f6feb; font-weight: 600; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

EXAMPLE_QUESTIONS = [
    "What is the minimum GPA required to graduate?",
    "What are the rules for postponement of registration?",
    "How many training hours does the internship require?",
    "What are the prerequisites for Chemical Thermodynamics I?",
    "How can a student appeal a final exam grade?",
]

ABOUT_NOTE = """\
The proposed University of Sharjah Virtual Academic Advisor is an online platform \
designed to help students plan their academic pathways, track degree requirements, \
and make informed course-registration decisions. This is a complementary service \
and does not in any way replace the role of academic advisors. It will be useful \
when academic advisors are on vacation or out of reach due to circumstances beyond \
their (or the student user's) control. This platform only works for BSc Chemical \
and Water Desalination Engineering Students for now. The goal of the developers is \
to ensure that the application is useful for the entire university in the long run.

**Developed by Dr. Adewale Giwa and Eng. Omar Elgendy.**
"""

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


# --------------------------------------------------------------------------- #
# Resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading search index…")
def get_index():
    return load_index()


@st.cache_resource(show_spinner=False)
def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


@st.cache_data(show_spinner=False, max_entries=64)
def render_pdf_page(source: str, page: int, zoom: float = 2.0) -> bytes | None:
    """Render one PDF page to a PNG so the user can see the cited text in place."""
    path = config.DOCS_DIR / source
    if not path.exists():
        return None
    doc = pymupdf.open(path)
    try:
        if not 1 <= page <= doc.page_count:
            return None
        pix = doc[page - 1].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        return pix.tobytes("png")
    finally:
        doc.close()


@st.cache_data(show_spinner=False)
def read_doc_bytes(source: str) -> bytes | None:
    path = config.DOCS_DIR / source
    return path.read_bytes() if path.exists() else None


def highlight_citations(text: str) -> str:
    """Colour the [n] markers so citations stand out in the answer."""
    return re.sub(r"\[(\d{1,2})\]", r"<span class='cite'>[\1]</span>", text)


def render_sources(sources: list[dict], key_prefix: str) -> None:
    """Show the passages an answer was built from, with page previews."""
    if not sources:
        return
    st.markdown(f"**Sources** ({len(sources)})")
    for i, src in enumerate(sources, start=1):
        is_pdf = src["source"].lower().endswith(".pdf")
        is_docx = src["source"].lower().endswith(".docx")
        # DOCX syllabi and scraped web pages have no page numbers (page_start 0).
        pages = ""
        if src["page_start"]:
            pages = (
                f"page {src['page_start']}"
                if src["page_start"] == src["page_end"]
                else f"pages {src['page_start']}–{src['page_end']}"
            )
        label = f"[{i}] {src['title']}" + (f" — {pages}" if pages else "")
        with st.expander(label, expanded=False):
            meta = f"`{src['source']}` · matched by {src['retrievers']} search"
            if pages:
                meta = f"`{src['source']}` · {pages} · matched by {src['retrievers']} search"
            if src.get("section"):
                meta = f"**Section {src['section']}**  \n{meta}"
            st.markdown(meta)
            if src.get("url"):
                st.markdown(f"🔗 [Open this page on sharjah.ac.ae]({src['url']})")
            st.markdown("---")
            st.text(src["text"][:2000] + ("…" if len(src["text"]) > 2000 else ""))

            col1, col2 = st.columns([1, 1])
            show = False
            if is_pdf:
                with col1:
                    show = st.toggle(
                        "Show document page",
                        key=f"{key_prefix}-page-{i}",
                        help="Render the original PDF page this passage came from.",
                    )
            if is_pdf or is_docx:
                data = read_doc_bytes(src["source"])
                if data:
                    with col2:
                        st.download_button(
                            "Download PDF" if is_pdf else "Download syllabus (DOCX)",
                            data=data,
                            file_name=src["source"].rsplit("/", 1)[-1],
                            mime="application/pdf" if is_pdf else DOCX_MIME,
                            key=f"{key_prefix}-dl-{i}",
                        )
            if show:
                png = render_pdf_page(src["source"], src["page_start"])
                if png:
                    st.image(png, caption=f"{src['source']} — page {src['page_start']}")
                else:
                    st.info("Original PDF not available in this deployment.")


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
# The key is supplied by the deployment, never by the visitor. If it is missing
# that is a server misconfiguration, so fail fast rather than degrade.
api_key = config.get_api_key()
if not api_key:
    st.error("**Configuration error:** this app has no OpenAI API key configured.")
    st.markdown(
        "The key is provided by the server, not by visitors. Whoever deploys this app "
        "should set it and restart:\n\n"
        "- **Streamlit Community Cloud** — app menu → *Settings* → *Secrets*, then add "
        "`OPENAI_API_KEY = \"sk-...\"`\n"
        "- **Local development** — put `OPENAI_API_KEY=sk-...` in a `.env` file, or "
        "export it in the shell"
    )
    st.stop()

with st.sidebar:
    st.title("📘 Settings")

    try:
        index = get_index()
    except Exception as exc:  # index missing or corrupt
        st.error(str(exc))
        st.stop()

    st.caption(
        f"**{index.meta.get('n_chunks', len(index.chunks))}** passages from "
        f"**{len(index.sources)}** documents"
    )
    if index.meta.get("built_at"):
        st.caption(f"Index built {index.meta['built_at']}")

    st.subheader("Search")
    top_k = st.slider("Passages per answer", 3, 12, config.TOP_K)
    use_rerank = st.toggle(
        "LLM reranking",
        value=bool(config.RERANK),
        help="Re-order retrieved passages by relevance before answering. "
        "More accurate, slightly slower.",
    )
    selected = st.multiselect(
        "Limit to documents",
        options=index.sources,
        default=[],
        help="Leave empty to search all documents.",
    )

    st.subheader("Models")
    st.caption(f"Chat: `{config.CHAT_MODEL}`")
    st.caption(f"Embeddings: `{config.EMBED_MODEL}`")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption(
        "Answers are generated from the indexed documents only. "
        "Always confirm critical decisions against the official PDF."
    )

# --------------------------------------------------------------------------- #
# Main pane
# --------------------------------------------------------------------------- #
st.title("CWDE Program Virtual Course Advisor")
st.info(ABOUT_NOTE)
st.caption(
    "Ask about policies, procedures, internships, study plans, course syllabi, "
    "and university web pages. Every answer cites the source it came from."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

tab_chat, tab_docs = st.tabs(["💬 Chat", "📚 Supported documents & pages"])

with tab_docs:
    st.markdown(
        "Everything the advisor can answer from. Titles only — open the source "
        "itself from the citations under an answer."
    )
    docs: dict[str, dict] = {}
    for c in index.chunks:
        docs.setdefault(c.source, {"title": c.title, "url": getattr(c, "url", "")})
    pdfs = {s: d for s, d in docs.items() if s.lower().endswith(".pdf")}
    syllabi = {s: d for s, d in docs.items() if s.lower().endswith(".docx")}
    web = {s: d for s, d in docs.items() if d["url"] or s.startswith("web/")}

    st.subheader(f"Policies, procedures & study plans ({len(pdfs)})")
    for d in sorted(pdfs.values(), key=lambda d: d["title"].lower()):
        st.markdown(f"- {d['title']}")

    st.subheader(f"Course syllabi ({len(syllabi)})")
    for d in sorted(syllabi.values(), key=lambda d: d["title"].lower()):
        st.markdown(f"- {d['title']}")

    st.subheader(f"University web pages ({len(web)})")
    for d in sorted(web.values(), key=lambda d: d["title"].lower()):
        if d["url"]:
            st.markdown(f"- [{d['title']}]({d['url']})")
        else:
            st.markdown(f"- {d['title']}")

with tab_chat:
    if not st.session_state.messages:
        st.markdown("**Try one of these:**")
        cols = st.columns(len(EXAMPLE_QUESTIONS[:3]))
        for col, q in zip(cols, EXAMPLE_QUESTIONS[:3]):
            if col.button(q, use_container_width=True):
                st.session_state.pending = q
                st.rerun()
        with st.expander("More examples"):
            for q in EXAMPLE_QUESTIONS[3:]:
                if st.button(q, key=f"ex-{q}", use_container_width=True):
                    st.session_state.pending = q
                    st.rerun()

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown(highlight_citations(msg["content"]), unsafe_allow_html=True)
                render_sources(msg.get("sources", []), key_prefix=f"m{i}")
            else:
                st.markdown(msg["content"])

    prompt = st.chat_input("Ask about the university documents…")
    if not prompt:
        # An example-question button was clicked on the previous run.
        prompt = st.session_state.pop("pending", None)

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        client = get_client(api_key)
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching the documents…"):
                    search_query = rewrite_query(client, prompt, history)
                    hits = retrieve(
                        client,
                        index,
                        search_query,
                        top_k=top_k,
                        rerank=use_rerank,
                        sources=selected or None,
                    )
                if search_query.strip().lower() != prompt.strip().lower():
                    st.caption(f"Searched for: _{search_query}_")

                text = st.write_stream(answer_stream(client, prompt, hits, history))
            except Exception as exc:
                text = f"Something went wrong while answering: `{exc}`"
                st.error(text)
                hits = []

            sources = [
                {
                    "title": h.chunk.title,
                    "source": h.chunk.source,
                    "page_start": h.chunk.page_start,
                    "page_end": h.chunk.page_end,
                    "section": h.chunk.section,
                    "text": h.chunk.text,
                    "retrievers": h.retrievers,
                    "url": h.chunk.url,
                }
                for h in hits
            ]
            render_sources(sources, key_prefix=f"m{len(st.session_state.messages)}")

        st.session_state.messages.append(
            {"role": "assistant", "content": text, "sources": sources}
        )
