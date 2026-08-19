#!/usr/bin/env python
"""Scrape sharjah.ac.ae pages into Docs/web/ as markdown for indexing.

    python scripts/scrape_site.py                  # priority crawl, default cap
    python scripts/scrape_site.py --max-pages 500  # raise the cap
    python scripts/scrape_site.py --only-seeds     # just the must-have pages

The university sitemap lists ~14,600 URLs (half of them Arabic mirrors).
Indexing all of them is impossible with this repository's committed-index
architecture: the FAISS file would be several hundred MB, which GitHub refuses
(100 MB per-file limit) and a Streamlit Community Cloud instance cannot hold in
memory. The crawl is therefore **priority-ordered and capped**: the seed pages
and the sections most relevant to student advising are fetched first, and the
cap cuts the tail. Raise `--max-pages` deliberately, watching the size of
`index/faiss.index`.

Each page becomes `Docs/web/<slug>.md` with a small front-matter header
(url / title / fetched) that `rag/ingest.py` understands. Re-running refreshes
existing files in place.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup, Tag

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import config  # noqa: E402

BASE = "https://www.sharjah.ac.ae"
SITEMAP = f"{BASE}/sitemap.xml"
OUT_DIR = config.DOCS_DIR / "web"

# Pages the assistant must always have, fetched first regardless of the cap.
SEED_URLS = [
    f"{BASE}/Academics/Degree/Undergraduate/Chemical-and-Water-Desalination-Engineering",
]

# Sitemap sections in priority order — most useful for student advising first.
# Everything else in the sitemap is lower priority but still eligible.
SECTION_PRIORITY = [
    "Admissions",
    "Academics/Academic-Calendar",
    "Academics/Degree",
    "Student-Life",
    "Academics/Eng",
    "Services",
    "Discover-UoS",
    "Academics",
]

# Never fetch: Arabic mirrors, news/event streams, staff profile pages, and
# site plumbing. They dilute retrieval without answering advising questions.
EXCLUDE = re.compile(
    r"/ar(/|$)"
    r"|/News(/|$)|/Global-News(/|$)|/Events(/|$)|/Conferences(/|$)"
    r"|/Faculty-And-Staff(/|$)"
    r"|/404-Not-found|/Search$|DELETE-TEST",
    re.IGNORECASE,
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept-Language": "en",
}

# Site chrome stripped before text extraction (shared header/footer/widgets).
_STRIP_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "form", "button", "svg", "iframe", "aside"]
_STRIP_SELECTORS = [
    "[class*=cookie]", "[id*=cookie]", "[class*=feedback]", "[id*=feedback]",
    "[class*=breadcrumb]", "[class*=social]", "[class*=share]", "[class*=rating]",
    "[class*=skip]", "[aria-hidden=true]", "[class*=search]", "[class*=menu]",
    "[class*=banner]", "[class*=popup]", "[class*=modal]",
]
# Chrome phrases that survive element stripping on some templates.
_NOISE_LINES = re.compile(
    r"^(Accessibility|Sign Language|Feedback|Submit|Apply Online"
    r"|How would you rate your experience.*|\+?\d[\d\s()+-]{6,}"
    r"|Follow us.*|©.*|All Rights Reserved.*)$",
    re.IGNORECASE,
)


def slugify(url: str) -> str:
    path = urlparse(url).path.strip("/") or "home"
    slug = re.sub(r"[^A-Za-z0-9]+", "-", path).strip("-")
    return slug[:120] or "home"


def compose_title(url: str, title: str) -> str:
    """Contextualize generic page titles: "Undergraduate" -> "Admissions — Undergraduate"."""
    segments = [s for s in urlparse(url).path.strip("/").split("/") if s]
    if not title:
        title = segments[-1].replace("-", " ") if segments else "Home"
    if len(segments) >= 2 and len(title) < 30:
        section = segments[0].replace("-", " ")
        if section.lower() not in title.lower():
            title = f"{section} — {title}"
    return title


def _table_lines(table: Tag) -> list[str]:
    lines = []
    for tr in table.find_all("tr"):
        cells = [" ".join(td.get_text(" ", strip=True).split()) for td in tr.find_all(["td", "th"])]
        merged: list[str] = []
        for c in cells:
            if c and (not merged or c != merged[-1]):
                merged.append(c)
        if merged:
            lines.append(" | ".join(merged))
    return lines


def extract_page(html: str) -> tuple[str, list[str]]:
    """HTML -> (title, markdown lines). Returns ("", []) for empty pages."""
    soup = BeautifulSoup(html, "lxml")
    title = ""
    if soup.title:
        title = soup.title.get_text(" ", strip=True)
        # Sitecore appends nothing here, but normalise separators just in case.
        title = re.split(r"\s*[|–—]\s*", title)[0].strip() or title

    root = soup.find("main") or soup.body
    if root is None:
        return title, []
    for tag in root.find_all(_STRIP_TAGS):
        tag.decompose()
    for sel in _STRIP_SELECTORS:
        try:
            for el in root.select(sel):
                el.decompose()
        except Exception:
            pass

    lines: list[str] = []
    seen_tables: set[int] = set()
    for el in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table"]):
        if any(id(parent) in seen_tables for parent in el.parents):
            continue  # already rendered as part of a table
        if el.name == "table":
            seen_tables.add(id(el))
            rows = _table_lines(el)
            if rows:
                lines.extend([""] + rows + [""])
            continue
        if el.name == "li" and el.find(["p", "li", "table"]):
            continue  # container item; children will be visited
        text = " ".join(el.get_text(" ", strip=True).split())
        if not text or _NOISE_LINES.match(text):
            continue
        if el.name.startswith("h"):
            level = int(el.name[1])
            lines.extend(["", f"{'#' * level} {text}", ""])
        elif el.name == "li":
            lines.append(f"- {text}")
        else:
            lines.extend([text, ""])

    # Collapse duplicate consecutive lines and blank runs.
    out: list[str] = []
    for ln in lines:
        if ln and out and ln == out[-1]:
            continue
        if not ln and (not out or not out[-1]):
            continue
        out.append(ln)
    return title, out


def fetch(session: requests.Session, url: str) -> str | None:
    for attempt in range(3):
        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 200 and "text/html" in resp.headers.get("content-type", ""):
                return resp.text
            if resp.status_code in (404, 410):
                return None
        except requests.RequestException:
            pass
        time.sleep(2**attempt)
    return None


def sitemap_urls(session: requests.Session) -> list[str]:
    resp = session.get(SITEMAP, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    root = ElementTree.fromstring(resp.content)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    return [loc.text.strip() for loc in root.iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc") if loc.text]


def prioritise(urls: list[str]) -> list[str]:
    """Order sitemap URLs: seeds, then priority sections, then the rest."""
    def rank(url: str) -> tuple:
        path = urlparse(url).path.strip("/")
        for i, section in enumerate(SECTION_PRIORITY):
            if path == section or path.startswith(section + "/"):
                return (i, path.count("/"), path)
        return (len(SECTION_PRIORITY), path.count("/"), path)

    eligible = sorted(
        {u.rstrip("/") for u in urls if u.startswith(BASE) and not EXCLUDE.search(u)},
        key=rank,
    )
    ordered = list(SEED_URLS)
    ordered.extend(u for u in eligible if u not in set(SEED_URLS))
    return ordered


def scrape(urls: list[str], delay: float) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    written = 0
    for url in urls:
        html = fetch(session, url)
        if html is None:
            print(f"  SKIP (unreachable): {url}")
            continue
        title, lines = extract_page(html)
        body = "\n".join(lines).strip()
        if len(body) < 200:  # empty shells and soft 404s
            print(f"  SKIP (no content): {url}")
            continue
        slug = slugify(url)
        title = compose_title(url, title)
        front = (
            "---\n"
            f"url: {url}\n"
            f"title: {title}\n"
            f"fetched: {time.strftime('%Y-%m-%d')}\n"
            "---\n\n"
        )
        (OUT_DIR / f"{slug}.md").write_text(front + body + "\n", encoding="utf-8")
        written += 1
        print(f"  [{written}] {title}  <- {url}")
        time.sleep(delay)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-pages", type=int, default=350, help="crawl cap (default 350)")
    parser.add_argument("--delay", type=float, default=0.4, help="seconds between requests")
    parser.add_argument("--only-seeds", action="store_true", help="fetch only the seed pages")
    args = parser.parse_args()

    session = requests.Session()
    if args.only_seeds:
        urls = list(SEED_URLS)
    else:
        print("Reading sitemap…")
        urls = prioritise(sitemap_urls(session))
        print(f"  {len(urls)} eligible English pages; crawling the top {args.max_pages}.")
        urls = urls[: args.max_pages]

    written = scrape(urls, args.delay)
    print(f"\nWrote {written} pages to {OUT_DIR}.")
    print("Now rebuild the index:  python scripts/build_index.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
