"""Query rewriting and grounded answer generation with inline citations."""

from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI

from . import config
from .retrieve import Hit

SYSTEM_PROMPT = """\
You are a precise assistant answering questions about University of Sharjah policy \
and study-plan documents. You answer ONLY from the numbered sources provided.

Rules:
1. Ground every factual statement in the sources. Cite with bracketed numbers that \
match the source list, e.g. "Students may postpone for up to two semesters [2]."
2. Cite the specific source for each claim. Use several citations when a claim draws \
on more than one source.
3. If the sources do not contain the answer, say so plainly and state what the sources \
do cover. Never guess, and never fall back on general knowledge about other universities.
4. Quote exact figures, deadlines, GPA thresholds, credit hours, and course codes as \
written. Do not round or paraphrase numbers.
5. Prefer a short direct answer first, then supporting detail. Use bullets or a table \
when listing requirements, courses, or steps.
6. If the sources conflict (for example two academic years of a study plan), say so and \
attribute each version to its document.
7. Answer in the language the user asked in.
"""

_REWRITE_PROMPT = """\
Rewrite the user's latest message into a standalone search query for a document \
retrieval system, resolving pronouns and references from the conversation.

Keep the institutional vocabulary the documents would use. Return only the query text.

Conversation so far:
{history}

Latest message: {question}

Standalone search query:"""


def rewrite_query(client: OpenAI, question: str, history: list[dict]) -> str:
    """Turn a follow-up like "what about the deadline?" into a standalone query."""
    if not history:
        return question
    recent = history[-6:]
    rendered = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in recent)
    try:
        resp = client.chat.completions.create(
            model=config.UTILITY_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": _REWRITE_PROMPT.format(history=rendered, question=question),
                }
            ],
            reasoning_effort="low",
            max_completion_tokens=2000,
        )
        rewritten = (resp.choices[0].message.content or "").strip()
        return rewritten or question
    except Exception:
        return question


def format_context(hits: list[Hit]) -> str:
    """Render retrieved passages as a numbered source list for the prompt."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        c = hit.chunk
        header = f"[{i}] {c.title}"
        pages = c.pages()  # "" for pageless sources (syllabi, web pages)
        if pages:
            header += f" — {pages}"
        if c.section:
            header += f" — Section {c.section}"
        blocks.append(f"{header}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def answer_stream(
    client: OpenAI,
    question: str,
    hits: list[Hit],
    history: list[dict] | None = None,
) -> Iterator[str]:
    """Stream a grounded answer to `question` from the retrieved passages."""
    if not hits:
        yield (
            "I could not find anything relevant to that in the indexed documents. "
            "Try rephrasing, or ask about admissions, registration, examinations, "
            "grading, internships, promotion, or the Chemical & Water Desalination "
            "Engineering study plans."
        )
        return

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in (history or [])[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append(
        {
            "role": "user",
            "content": (
                f"Sources:\n\n{format_context(hits)}\n\n"
                f"---\n\nQuestion: {question}\n\n"
                "Answer using only the sources above, with bracketed citations."
            ),
        }
    )

    stream = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=messages,
        stream=True,
        temperature=0.1,
        max_completion_tokens=4000,
    )
    for event in stream:
        if event.choices and event.choices[0].delta.content:
            yield event.choices[0].delta.content


def answer(
    client: OpenAI,
    question: str,
    hits: list[Hit],
    history: list[dict] | None = None,
) -> str:
    """Non-streaming convenience wrapper, used by the CLI and tests."""
    return "".join(answer_stream(client, question, hits, history))
