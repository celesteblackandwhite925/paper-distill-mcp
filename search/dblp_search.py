#!/usr/bin/env python3
"""DBLP API client for paper-distill.

Queries https://dblp.org/search/publ/api for computer science papers.
Good coverage for LLM, AI, and ML papers.

CLI usage:
    python search/dblp_search.py "large language model" --max-results 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG = logging.getLogger("search.dblp")

DBLP_BASE = "https://dblp.org/search/publ/api"


def _normalise(hit: dict[str, Any]) -> dict[str, Any]:
    """Convert a DBLP hit to the unified paper schema."""
    info = hit.get("info", {})

    # Authors: can be string or list or dict with "author" key
    authors_raw = info.get("authors", {}).get("author", [])
    if isinstance(authors_raw, str):
        authors = [authors_raw]
    elif isinstance(authors_raw, dict):
        authors = [authors_raw.get("text", authors_raw.get("@text", ""))]
    elif isinstance(authors_raw, list):
        authors = []
        for a in authors_raw[:5]:
            if isinstance(a, str):
                authors.append(a)
            elif isinstance(a, dict):
                authors.append(a.get("text", a.get("@text", "")))
    else:
        authors = []

    year = None
    year_str = info.get("year")
    if year_str:
        try:
            year = int(year_str)
        except (ValueError, TypeError):
            pass

    doi = info.get("doi", "") or ""
    # DBLP doi often has prefix like "db/..."  - only keep real DOIs
    if doi and not doi.startswith("10."):
        doi = ""

    # URL/PDF
    ee = info.get("ee", "")
    if isinstance(ee, list):
        ee = ee[0] if ee else ""

    venue = info.get("venue", "")
    if isinstance(venue, list):
        venue = ", ".join(venue)

    title = info.get("title", "")
    # DBLP titles sometimes end with "."
    if title.endswith("."):
        title = title[:-1]

    return {
        "source": "dblp",
        "doi": doi,
        "pmid": "",
        "arxiv_id": "",
        "title": title,
        "authors": authors,
        "abstract": "",
        "journal": venue,
        "year": year,
        "citation_count": 0,
        "tldr": "",
        "open_access_url": ee,
        "topic_tags": [],
    }


async def search_dblp(
    query: str,
    *,
    max_results: int = 20,
) -> list[dict[str, Any]]:
    """Query DBLP and return normalised paper dicts."""
    params: dict[str, Any] = {
        "q": query,
        "format": "json",
        "h": min(max_results, 40),
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(DBLP_BASE, params=params)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            LOG.warning("DBLP error: %s", exc)
            return []

    result = data.get("result", {})
    hits_wrapper = result.get("hits", {})
    hits = hits_wrapper.get("hit", [])

    return [_normalise(h) for h in hits[:max_results]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search DBLP")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    papers = asyncio.run(search_dblp(args.query, max_results=args.max_results))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(papers, ensure_ascii=False, indent=2))
        LOG.info("Wrote %d papers to %s", len(papers), args.output)
    else:
        print(json.dumps(papers, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
