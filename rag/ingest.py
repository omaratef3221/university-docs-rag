"""PDF -> cleaned, page-tracked, section-aware text chunks.

The source PDFs are university policy manuals and study plans. They share a few
quirks that plain text extraction handles badly:

* every page starts with blank lines and a bare page number;
* the manual opens with a table of contents whose entries are dotted leaders
  ("3.6 Academic Progress Policy ......... 33") that pollute retrieval;
* headings are numbered ("4.10 Graduate Assistants Policy") and are sometimes
  split across two lines, with the number alone on the first line;
* the study plans are almost entirely tables, which linear text extraction
  flattens column-major into unreadable noise ("3 0405 221 / Eng. Probability").

This module therefore reads each page as positioned blocks, renders detected
tables row-wise, strips the boilerplate, and emits chunks that carry the
document, page range, and section path needed to cite them.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf
import tiktoken

from . import config

_ENC = tiktoken.get_encoding("cl100k_base")

# "3.6 Academic Progress Policy ....... 33" — a table-of-contents line.
_TOC_LINE = re.compile(r"\.{4,}\s*\d+\s*$")
# A bare section number on its own line, e.g. "4.10" (heading split across lines).
_LONE_NUMBER = re.compile(r"^\d+(?:\.\d+)*\.?$")
# A candidate numbered heading, e.g. "4.10.1 Introduction and Purpose".
_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*)$")
_PAGE_NUMBER = re.compile(r"^\d{1,4}$")
_SENTENCE_END = (".", ",", ";", ":")


def n_tokens(text: str) -> int:
    return len(_ENC.encode(text))


@dataclass
class Chunk:
    """One retrievable passage plus everything needed to cite it."""

    id: int
    source: str  # PDF file name
    title: str  # human-readable document title
    page_start: int  # 1-based, as printed in a PDF viewer
    page_end: int
    section: str  # nearest enclosing numbered heading, "" if none
    text: str
    tokens: int = 0
    embed_text: str = field(default="", repr=False)

    def citation(self) -> str:
        pages = (
            f"p. {self.page_start}"
            if self.page_start == self.page_end
            else f"pp. {self.page_start}-{self.page_end}"
        )
        return f"{self.title}, {pages}"


def pretty_title(filename: str) -> str:
    """Turn a PDF file name into something readable for citations."""
    stem = Path(filename).stem
    stem = re.sub(r"[_-]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    # Split "InternshipTrainingPoliciesProcedures2425" into words.
    if stem.count(" ") <= 1:
        stem = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", stem)
    stem = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", stem)  # "plan2023" -> "plan 2023"
    return re.sub(r"\s+", " ", stem).strip()


def _is_heading(line: str) -> str | None:
    """Return the normalised heading if `line` is a section heading, else None.

    Numbered list items ("2. Papers written in English Language which are ...")
    match the same pattern as headings, so length and terminal punctuation are
    used to tell the two apart.
    """
    if "|" in line:  # rendered table row, never a heading
        return None
    m = _HEADING.match(line)
    if not m:
        return None
    number, title = m.group(1), m.group(2).strip()
    if title.endswith(_SENTENCE_END):
        return None
    words = title.split()
    multilevel = "." in number
    max_chars, max_words = (90, 12) if multilevel else (60, 9)
    if len(title) > max_chars or len(words) > max_words:
        return None
    return f"{number} {title}"


def _render_table(table) -> str:
    """Render a detected table as one line per row, pipe-separated."""
    rows: list[str] = []
    for row in table.extract():
        cells = [(c or "").replace("\n", " ").strip() for c in row]
        cells = [c for c in cells if c]
        # Extraction sometimes splits a leading capital into its own cell
        # ("A" + "rabic Language"); glue those back together.
        merged: list[str] = []
        for cell in cells:
            if merged and len(merged[-1]) == 1 and merged[-1].isalpha() and cell[:1].islower():
                merged[-1] += cell
            else:
                merged.append(cell)
        if merged:
            rows.append(" | ".join(merged))
    return "\n".join(rows)


def _page_lines(page) -> list[str]:
    """Extract one page as ordered lines, with tables rendered row-wise."""
    try:
        tables = [t for t in page.find_tables().tables if t.row_count >= 2 and t.col_count >= 2]
    except Exception:
        tables = []
    table_rects = [pymupdf.Rect(t.bbox) for t in tables]

    items: list[tuple[float, str]] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:  # not a text block
            continue
        rect = pymupdf.Rect(block["bbox"])
        area = rect.get_area()
        # Skip prose blocks that are really the innards of a detected table.
        if area > 0 and any((rect & tr).get_area() > 0.5 * area for tr in table_rects):
            continue
        text = "\n".join(
            "".join(span["text"] for span in line["spans"]) for line in block["lines"]
        )
        if text.strip():
            items.append((rect.y0, text))
    for table, rect in zip(tables, table_rects):
        rendered = _render_table(table)
        if rendered:
            items.append((rect.y0, rendered))
    items.sort(key=lambda item: item[0])

    # Blank line between blocks so paragraph boundaries survive.
    lines = [ln.strip() for _, block in items for ln in (block.splitlines() + [""])]

    # Drop the running page number that sits above the body text.
    while lines and not lines[0]:
        lines.pop(0)
    if lines and _PAGE_NUMBER.match(lines[0]):
        lines.pop(0)

    kept = [ln for ln in lines if not (ln and _TOC_LINE.search(ln))]

    # Reattach headings whose number was extracted onto its own line.
    merged: list[str] = []
    i = 0
    while i < len(kept):
        cur = kept[i]
        if _LONE_NUMBER.match(cur):
            nxt = next((j for j in range(i + 1, len(kept)) if kept[j]), None)
            if nxt is not None and not _LONE_NUMBER.match(kept[nxt]):
                merged.append(f"{cur} {kept[nxt]}")
                i = nxt + 1
                continue
        merged.append(cur)
        i += 1

    # Collapse runs of blank lines.
    collapsed: list[str] = []
    for ln in merged:
        if not ln and (not collapsed or not collapsed[-1]):
            continue
        collapsed.append(ln)
    while collapsed and not collapsed[-1]:
        collapsed.pop()
    return collapsed


def _is_toc_page(lines: list[str], raw: str) -> bool:
    """True for front-matter contents pages, which we skip entirely."""
    return raw.count("....") >= 5 and len(lines) < 12


@dataclass
class _Para:
    text: str
    page: int
    section: str


def _paragraphs(pdf_path: Path) -> list[_Para]:
    """Flatten a PDF into paragraphs tagged with page number and section."""
    doc = pymupdf.open(pdf_path)
    paras: list[_Para] = []
    section = ""
    try:
        for page_index, page in enumerate(doc):
            lines = _page_lines(page)
            if _is_toc_page(lines, page.get_text()):
                continue
            page_no = page_index + 1
            buf: list[str] = []

            def flush() -> None:
                if buf:
                    text = " ".join(buf).strip()
                    if text:
                        paras.append(_Para(text, page_no, section))
                    buf.clear()

            for ln in lines:
                if not ln:
                    flush()
                    continue
                heading = _is_heading(ln)
                if heading:
                    flush()
                    section = heading
                    paras.append(_Para(ln, page_no, section))
                    continue
                if "|" in ln:
                    # Keep each table row as its own paragraph so rows are never
                    # glued into a single unreadable line.
                    flush()
                    paras.append(_Para(ln, page_no, section))
                    continue
                buf.append(ln)
            flush()
    finally:
        doc.close()
    return paras


def _split_long(para: _Para) -> list[_Para]:
    """Break a paragraph that alone exceeds the chunk budget."""
    tokens = _ENC.encode(para.text)
    if len(tokens) <= config.CHUNK_TOKENS:
        return [para]
    step = config.CHUNK_TOKENS - config.CHUNK_OVERLAP_TOKENS
    pieces: list[_Para] = []
    for start in range(0, len(tokens), step):
        piece = _ENC.decode(tokens[start : start + config.CHUNK_TOKENS]).strip()
        if piece:
            pieces.append(_Para(piece, para.page, para.section))
        if start + config.CHUNK_TOKENS >= len(tokens):
            break
    return pieces


def chunk_document(pdf_path: Path, start_id: int) -> list[Chunk]:
    """Chunk one PDF into overlapping, section-aware passages."""
    source = pdf_path.name
    title = pretty_title(source)
    paras: list[_Para] = []
    for para in _paragraphs(pdf_path):
        paras.extend(_split_long(para))

    chunks: list[Chunk] = []
    buf: list[_Para] = []
    buf_tokens = 0
    next_id = start_id

    def emit() -> None:
        nonlocal buf, buf_tokens, next_id
        if not buf:
            return
        text = "\n".join(p.text for p in buf).strip()
        if n_tokens(text) < config.MIN_CHUNK_TOKENS:
            return
        section = next((p.section for p in buf if p.section), "")
        chunk = Chunk(
            id=next_id,
            source=source,
            title=title,
            page_start=min(p.page for p in buf),
            page_end=max(p.page for p in buf),
            section=section,
            text=text,
            tokens=n_tokens(text),
        )
        # Embedding the document/section header alongside the body measurably
        # improves retrieval on questions phrased in the document's own terms.
        header = f"Document: {title}"
        if section:
            header += f"\nSection: {section}"
        chunk.embed_text = f"{header}\n\n{text}"
        chunks.append(chunk)
        next_id += 1

    for para in paras:
        ptok = n_tokens(para.text)
        if buf and buf_tokens + ptok > config.CHUNK_TOKENS:
            emit()
            # Carry the tail of the previous chunk forward as overlap.
            overlap: list[_Para] = []
            total = 0
            for prev in reversed(buf):
                t = n_tokens(prev.text)
                if total + t > config.CHUNK_OVERLAP_TOKENS:
                    break
                overlap.insert(0, prev)
                total += t
            buf = overlap
            buf_tokens = total
        buf.append(para)
        buf_tokens += ptok
    emit()
    return chunks


def build_chunks(docs_dir: Path | None = None) -> list[Chunk]:
    """Chunk every PDF in the documents directory."""
    docs_dir = docs_dir or config.DOCS_DIR
    pdfs = sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {docs_dir}")
    chunks: list[Chunk] = []
    for pdf in pdfs:
        doc_chunks = chunk_document(pdf, start_id=len(chunks))
        print(f"  {pdf.name}: {len(doc_chunks)} chunks")
        chunks.extend(doc_chunks)
    return chunks


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    """Persist chunks. `embed_text` is derived, so it is not written to disk."""
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            record = {k: v for k, v in asdict(c).items() if k != "embed_text"}
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as fh:
        return [Chunk(**json.loads(line)) for line in fh if line.strip()]
