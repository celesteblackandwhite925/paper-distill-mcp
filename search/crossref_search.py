#!/usr/bin/env python3
"""CrossRef API client for paper-distill.

Queries https://api.crossref.org/works for papers with comprehensive DOI metadata.

CLI usage:
    python search/crossref_search.py "LLM reasoning" --max-results 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG = logging.getLogger("search.crossref")

CROSSREF_BASE = "https://api.crossref.org/works"


def _normalise(item: dict[str, Any]) -> dict[str, Any]:
    """Convert a CrossRef work to the unified paper schema."""
    authors = []
    for author in item.get("author", []):
        given = author.get("given", "")
        family = author.get("family", "")
        if family:
            authors.append(f"{given} {family}".strip())

    year = None
    for date_field in ("published-print", "published-online", "created"):
        date_parts = item.get(date_field, {}).get("date-parts", [[]])
        if date_parts and date_parts[0] and date_parts[0][0]:
            year = date_parts[0][0]
            break

    doi = item.get("DOI", "")
    container = item.get("container-title", [])
    journal = container[0] if container else ""
    issn_list = item.get("ISSN", [])

    abstract = item.get("abstract", "")
    if abstract:
        # CrossRef abstracts often have JATS XML tags
        import re
        abstract = re.sub(r"<[^>]+>", "", abstract)[:300]

    return {
        "source": "crossref",
        "doi": doi,
        "pmid": "",
        "arxiv_id": "",
        "title": (item.get("title") or [""])[0],
        "authors": authors[:5],
        "abstract": abstract,
        "journal": journal,
        "issn": issn_list,
        "year": year,
        "citation_count": item.get("is-referenced-by-count", 0),
        "tldr": "",
        "open_access_url": "",
        "topic_tags": [],
    }


async def search_crossref(
    query: str,
    *,
    max_results: int = 20,
    email: str | None = None,
) -> list[dict[str, Any]]:
    """Query CrossRef /works and return normalised paper dicts."""
    email = email or os.getenv("OPENALEX_EMAIL", "")
    headers = {}
    if email:
        headers["User-Agent"] = f"paper-distill/1.0 (mailto:{email})"

    params: dict[str, Any] = {
        "query": query,
        "rows": min(max_results, 50),
        "sort": "relevance",
        "order": "desc",
        "filter": "type:journal-article",
        "select": "DOI,title,author,published-print,published-online,created,"
                  "container-title,is-referenced-by-count,ISSN,abstract",
    }

    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        try:
            resp = await client.get(CROSSREF_BASE, params=params)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            LOG.warning("CrossRef error: %s", exc)
            return []

    items = data.get("message", {}).get("items", [])
    return [_normalise(item) for item in items[:max_results]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search CrossRef")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    papers = asyncio.run(search_crossref(args.query, max_results=args.max_results))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(papers, ensure_ascii=False, indent=2))
        LOG.info("Wrote %d papers to %s", len(papers), args.output)
    else:
        print(json.dumps(papers, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
