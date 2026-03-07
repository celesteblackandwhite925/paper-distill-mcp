#!/usr/bin/env python3
"""Europe PMC API client for paper-distill.

Queries https://www.ebi.ac.uk/europepmc/webservices/rest/search — a PubMed
superset that also covers preprints and patent citations.

CLI usage:
    python search/europepmc_search.py "LLM reasoning" --max-results 20
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
LOG = logging.getLogger("search.europepmc")

EUROPEPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def _normalise(result: dict[str, Any]) -> dict[str, Any]:
    """Convert a Europe PMC result to the unified paper schema."""
    author_string = result.get("authorString", "")
    authors = [a.strip() for a in author_string.split(",") if a.strip()][:5] if author_string else []

    year = None
    pub_year = result.get("pubYear")
    if pub_year:
        try:
            year = int(pub_year)
        except (ValueError, TypeError):
            pass

    return {
        "source": "europepmc",
        "doi": result.get("doi", "") or "",
        "pmid": result.get("pmid", "") or "",
        "arxiv_id": "",
        "title": result.get("title", ""),
        "authors": authors,
        "abstract": (result.get("abstractText") or "")[:300],
        "journal": result.get("journalTitle", "") or "",
        "year": year,
        "citation_count": result.get("citedByCount", 0) or 0,
        "tldr": "",
        "open_access_url": "",
        "topic_tags": [],
    }


async def search_europepmc(
    query: str,
    *,
    max_results: int = 20,
) -> list[dict[str, Any]]:
    """Query Europe PMC and return normalised paper dicts."""
    from urllib.parse import quote

    page_size = min(max_results, 25)
    # Europe PMC API requires + for spaces (not %20 or %2B)
    q_encoded = quote(query, safe="").replace("%20", "+")
    url = (
        f"{EUROPEPMC_BASE}?query={q_encoded}"
        f"&format=json&pageSize={page_size}"
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            LOG.warning("Europe PMC error: %s", exc)
            return []

    results = data.get("resultList", {}).get("result", [])
    return [_normalise(r) for r in results[:max_results]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Europe PMC")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    papers = asyncio.run(search_europepmc(args.query, max_results=args.max_results))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(papers, ensure_ascii=False, indent=2))
        LOG.info("Wrote %d papers to %s", len(papers), args.output)
    else:
        print(json.dumps(papers, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
