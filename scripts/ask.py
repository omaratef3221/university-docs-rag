#!/usr/bin/env python
"""Ask the index a question from the terminal — handy for smoke-testing.

    python scripts/ask.py "What is the minimum GPA required to graduate?"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI  # noqa: E402

from rag import config  # noqa: E402
from rag.generate import answer_stream  # noqa: E402
from rag.retrieve import retrieve  # noqa: E402
from rag.store import load_index  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    question = " ".join(argv[1:])

    api_key = config.get_api_key()
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set.")
        return 1

    client = OpenAI(api_key=api_key)
    index = load_index()
    hits = retrieve(client, index, question)

    for token in answer_stream(client, question, hits):
        print(token, end="", flush=True)
    print("\n\nSources")
    for i, hit in enumerate(hits, start=1):
        section = f" — {hit.chunk.section}" if hit.chunk.section else ""
        print(f"  [{i}] {hit.chunk.citation()}{section}  ({hit.retrievers})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
