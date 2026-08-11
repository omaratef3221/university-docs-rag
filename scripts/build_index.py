#!/usr/bin/env python
"""Build the hybrid search index from the PDFs in Docs/.

    python scripts/build_index.py

Re-run this whenever the documents change, then commit the regenerated index/
directory so the deployed app picks it up.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import config  # noqa: E402
from rag.store import build_index  # noqa: E402


def main() -> int:
    if not config.get_api_key():
        print("ERROR: OPENAI_API_KEY is not set (put it in .env or export it).")
        return 1

    started = time.time()

    def progress(done: int, total: int) -> None:
        pct = 100 * done / total
        print(f"\r  embedded {done}/{total} chunks ({pct:.0f}%)", end="", flush=True)

    index = build_index(progress=progress)
    print()
    print(
        f"Done in {time.time() - started:.1f}s — "
        f"{index.meta['n_chunks']} chunks, dim {index.meta['dim']}, "
        f"{len(index.meta['documents'])} documents."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
