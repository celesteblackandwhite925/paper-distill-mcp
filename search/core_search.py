#!/usr/bin/env python3
"""CORE API client for paper-distill.

Queries https://api.core.ac.uk/v3/search/works — the world's largest
aggregator of open-access research papers (200M+ records).

Requires a free API key: https://core.ac.uk/services/api
Set CORE_API_KEY in .env.

CLI usage:
    python search/core_search.py "LLM reasoning" --max-results 20
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
LOG = logging.getLogger("search.core")

CORE_BASE = "https://api.core.ac.uk/v3/search/works"


def _normalise(result: dict[str, Any]) -> dict[str, Any]:
    """Convert a CORE result to the unified paper schema."""
    authors_raw = result.get("authors") or []
    authors = []
    for a in authors_raw[:5]:
        if isinstance(a, dict):
            authors.append(a.get("name", ""))
        elif isinstance(a, str):
            authors.append(a)

    year = None
    year_raw = result.get("yearPublished")
    if year_raw:
        try:
            year = int(year_raw)
        except (ValueError, TypeError):
            pass

    # Best available download URL
    download_url = result.get("downloadUrl") or ""
    if not download_url:
        urls = result.get("sourceFulltextUrls") or []
        if urls:
            download_url = urls[0]

    return {
        "source": "core",
        "doi": result.get("doi", "") or "",
        "pmid": "",
        "arxiv_id": "",
        "title": result.get("title", "") or "",
        "authors": authors,
        "abstract": (result.get("abstract") or "")[:300],
        "journal": result.get("publisher", "") or "",
        "year": year,
        "citation_count": result.get("citationCount", 0) or 0,
        "tldr": "",
        "open_access_url": download_url,
        "topic_tags": [],
    }


async def search_core(
    query: str,
    *,
    max_results: int = 20,
) -> list[dict[str, Any]]:
    """Query CORE API and return normalised paper dicts."""
    api_key = os.getenv("CORE_API_KEY", "")
    if not api_key:
        LOG.warning("CORE_API_KEY not set, skipping CORE search")
        return []

    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"q": query, "limit": min(max_results, 100)}

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(CORE_BASE, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            LOG.warning("CORE error: %s", exc)
            return []

    results = data.get("results", [])
    return [_normalise(r) for r in results[:max_results]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search CORE")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    papers = asyncio.run(search_core(args.query, max_results=args.max_results))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(papers, ensure_ascii=False, indent=2))
        LOG.info("Wrote %d papers to %s", len(papers), args.output)
    else:
        print(json.dumps(papers, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
