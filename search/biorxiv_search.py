#!/usr/bin/env python3
"""bioRxiv / medRxiv API client for paper-distill.

The bioRxiv API does not support query-based search — it returns all papers
in a date range. We fetch recent papers and filter locally by keyword match
in title/abstract.

CLI usage:
    python search/biorxiv_search.py "large language model" --days 30 --max-results 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG = logging.getLogger("search.biorxiv")

BIORXIV_BASE = "https://api.biorxiv.org/details"


def _normalise(paper: dict[str, Any], server: str) -> dict[str, Any]:
    """Convert a bioRxiv/medRxiv result to the unified paper schema."""
    authors_str = paper.get("authors", "")
    authors = [a.strip() for a in authors_str.split(";") if a.strip()][:5]

    year = None
    date_str = paper.get("date", "")
    if date_str:
        try:
            year = int(date_str[:4])
        except (ValueError, IndexError):
            pass

    doi = paper.get("doi", "")

    return {
        "source": server,
        "doi": doi,
        "pmid": "",
        "arxiv_id": "",
        "title": paper.get("title", ""),
        "authors": authors,
        "abstract": (paper.get("abstract") or "")[:300],
        "journal": server,
        "year": year,
        "citation_count": 0,
        "tldr": "",
        "open_access_url": f"https://doi.org/{doi}" if doi else "",
        "topic_tags": [],
        "category": paper.get("category", ""),
    }


def _matches_query(paper: dict[str, Any], keywords: list[str]) -> bool:
    """Check if any keyword appears in title or abstract (case-insensitive)."""
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    return any(kw.lower() in text for kw in keywords)


async def _fetch_server(
    client: httpx.AsyncClient,
    server: str,
    start_date: str,
    end_date: str,
    keywords: list[str],
    max_results: int,
) -> list[dict[str, Any]]:
    """Fetch from one server (biorxiv or medrxiv)."""
    url = f"{BIORXIV_BASE}/{server}/{start_date}/{end_date}/0"
    try:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        LOG.warning("%s error: %s", server, exc)
        return []

    collection = data.get("collection", [])
    papers = []
    for item in collection:
        if _matches_query(item, keywords):
            papers.append(_normalise(item, server))
            if len(papers) >= max_results:
                break
    return papers


async def search_biorxiv(
    query: str,
    *,
    days: int = 30,
    max_results: int = 20,
) -> list[dict[str, Any]]:
    """Search bioRxiv and medRxiv for recent papers matching query keywords."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    start_date = start.strftime("%Y-%m-%d")
    end_date = end.strftime("%Y-%m-%d")

    keywords = [k.strip() for k in query.split() if len(k.strip()) >= 3]
    if not keywords:
        keywords = [query.strip()]

    per_server = max_results // 2 + 1

    async with httpx.AsyncClient(timeout=60.0) as client:
        bio_task = _fetch_server(client, "biorxiv", start_date, end_date, keywords, per_server)
        med_task = _fetch_server(client, "medrxiv", start_date, end_date, keywords, per_server)
        bio_results, med_results = await asyncio.gather(bio_task, med_task, return_exceptions=True)

    papers: list[dict[str, Any]] = []
    for result in (bio_results, med_results):
        if isinstance(result, Exception):
            LOG.warning("bioRxiv/medRxiv error: %s", result)
        elif result:
            papers.extend(result)

    return papers[:max_results]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search bioRxiv/medRxiv")
    parser.add_argument("query", help="Search keywords")
    parser.add_argument("--days", type=int, default=30, help="Look back N days")
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    papers = asyncio.run(search_biorxiv(args.query, days=args.days, max_results=args.max_results))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(papers, ensure_ascii=False, indent=2))
        LOG.info("Wrote %d papers to %s", len(papers), args.output)
    else:
        print(json.dumps(papers, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
