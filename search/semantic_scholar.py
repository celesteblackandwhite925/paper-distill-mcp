#!/usr/bin/env python3
"""Semantic Scholar API client for paper-distill.

Queries https://api.semanticscholar.org/graph/v1/paper/search for recent
papers matching user interests.  Respects 1 req/sec rate limit.

CLI usage:
    python search/semantic_scholar.py \
        --interests-file data/interests.jsonl \
        --output data/tmp_s2.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG = logging.getLogger("search.semantic_scholar")

S2_BASE = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = (
    "title,abstract,tldr,citationCount,year,authors,"
    "venue,externalIds,openAccessPdf"
)

# ---------------------------------------------------------------------------
# Interest helpers (shared pattern)
# ---------------------------------------------------------------------------

def _load_latest_interests(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        raise ValueError(f"interests file is empty: {path}")
    return json.loads(lines[-1])


def _extract_keywords(interests: dict[str, Any]) -> list[str]:
    keywords: list[str] = list(interests.get("keywords", []))
    topics = interests.get("topics", {})
    if isinstance(topics, dict):
        for topic_data in topics.values():
            if isinstance(topic_data, dict):
                keywords.extend(topic_data.get("keywords", []))
            elif isinstance(topic_data, list):
                keywords.extend(topic_data)
    return list(dict.fromkeys(keywords))


# ---------------------------------------------------------------------------
# Normalise
# ---------------------------------------------------------------------------

def _normalise(paper: dict[str, Any]) -> dict[str, Any]:
    """Convert a Semantic Scholar paper to the unified schema."""
    ext_ids = paper.get("externalIds") or {}
    doi = ext_ids.get("DOI", "")
    pmid = ext_ids.get("PubMed", "")
    arxiv_id = ext_ids.get("ArXiv", "")

    authors = [
        a.get("name", "")
        for a in (paper.get("authors") or [])
        if a.get("name")
    ]

    tldr_obj = paper.get("tldr") or {}
    tldr = tldr_obj.get("text", "") if isinstance(tldr_obj, dict) else ""

    oa_pdf = paper.get("openAccessPdf") or {}
    oa_url = oa_pdf.get("url", "") if isinstance(oa_pdf, dict) else ""

    return {
        "source": "s2",
        "doi": doi,
        "pmid": str(pmid) if pmid else "",
        "arxiv_id": arxiv_id,
        "title": paper.get("title", ""),
        "authors": authors,
        "abstract": paper.get("abstract", "") or "",
        "journal": paper.get("venue", "") or "",
        "year": paper.get("year"),
        "citation_count": paper.get("citationCount", 0),
        "tldr": tldr,
        "open_access_url": oa_url,
        "mesh_terms": [],
        "categories": [],
        "topic_tags": [],
    }


# ---------------------------------------------------------------------------
# Async search with rate limiting
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Simple 1-request-per-second limiter."""

    def __init__(self, interval: float = 1.0) -> None:
        self._interval = interval
        self._last: float = 0.0

    async def wait(self) -> None:
        now = time.monotonic()
        diff = self._interval - (now - self._last)
        if diff > 0:
            await asyncio.sleep(diff)
        self._last = time.monotonic()


async def search_semantic_scholar(
    keywords: list[str],
    *,
    year_range: str | None = None,
    max_results: int = 20,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Query Semantic Scholar paper search and return normalised papers."""

    api_key = api_key or os.getenv("S2_API_KEY", "")
    headers: dict[str, str] = {}
    if api_key:
        headers["x-api-key"] = api_key

    limiter = _RateLimiter(1.0)
    all_papers: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        for kw in keywords:
            await limiter.wait()

            params: dict[str, Any] = {
                "query": kw,
                "fields": S2_FIELDS,
                "limit": min(max_results, 100),
            }
            if year_range:
                params["year"] = year_range

            LOG.info("S2 query: query=%s", kw)
            try:
                resp = await client.get(S2_BASE, params=params)
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError as exc:
                LOG.warning("S2 HTTP %s for keyword '%s': %s",
                            exc.response.status_code, kw, exc)
                continue
            except httpx.RequestError as exc:
                LOG.warning("S2 request error for keyword '%s': %s", kw, exc)
                continue

            for paper in data.get("data", []):
                pid = paper.get("paperId", "")
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                all_papers.append(_normalise(paper))

            if len(all_papers) >= max_results:
                break

    return all_papers[:max_results]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search Semantic Scholar for papers",
    )
    parser.add_argument(
        "--interests-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "interests.jsonl",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=PROJECT_ROOT / "data" / "tmp_s2.json",
    )
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument(
        "--year-range",
        type=str,
        default="2021-2026",
        help="Year range, e.g. 2021-2026",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    )

    interests = _load_latest_interests(args.interests_file)
    keywords = _extract_keywords(interests)
    if not keywords:
        LOG.error("No keywords found in %s", args.interests_file)
        sys.exit(1)

    LOG.info("Keywords (%d): %s", len(keywords), keywords[:10])

    papers = asyncio.run(
        search_semantic_scholar(
            keywords,
            year_range=args.year_range,
            max_results=args.max_results,
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(papers, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOG.info("Wrote %d papers to %s", len(papers), args.output)


if __name__ == "__main__":
    main()
