#!/usr/bin/env python3
"""Papers With Code API client for paper-distill.

Queries https://paperswithcode.com/api/v1/papers/ for trending papers and
tools with code implementations.

CLI usage:
    python search/papers_with_code.py --output data/tmp_pwc.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG = logging.getLogger("search.papers_with_code")

PWC_BASE = "https://paperswithcode.com/api/v1"

# ---------------------------------------------------------------------------
# Interest helpers
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
    """Convert a Papers With Code paper to the unified schema."""

    # PWC paper object fields
    title = paper.get("title", "")
    abstract = paper.get("abstract", "") or ""
    arxiv_id = paper.get("arxiv_id", "") or ""
    url_pdf = paper.get("url_pdf", "") or ""
    published = paper.get("published", "") or ""

    # Parse year from published date (format: "2025-02-20")
    year = None
    if published and len(published) >= 4:
        try:
            year = int(published[:4])
        except ValueError:
            pass

    # Authors: PWC returns a list of dicts or a list of strings
    raw_authors = paper.get("authors", []) or []
    authors: list[str] = []
    for a in raw_authors:
        if isinstance(a, dict):
            authors.append(a.get("name", ""))
        elif isinstance(a, str):
            authors.append(a)

    return {
        "source": "pwc",
        "doi": "",
        "pmid": "",
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "journal": "",
        "year": year,
        "citation_count": 0,
        "tldr": "",
        "open_access_url": url_pdf,
        "mesh_terms": [],
        "categories": [],
        "topic_tags": [],
    }


# ---------------------------------------------------------------------------
# Async search
# ---------------------------------------------------------------------------

async def _fetch_trending(
    client: httpx.AsyncClient,
    *,
    max_results: int = 20,
) -> list[dict[str, Any]]:
    """Fetch trending / latest papers from PWC."""
    params: dict[str, Any] = {
        "ordering": "-published",
        "items_per_page": min(max_results, 50),
        "page": 1,
    }

    LOG.info("PWC trending query")
    try:
        resp = await client.get(f"{PWC_BASE}/papers/", params=params)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        LOG.warning("PWC HTTP %s: %s", exc.response.status_code, exc)
        return []
    except httpx.RequestError as exc:
        LOG.warning("PWC request error: %s", exc)
        return []

    results = data.get("results", [])
    return [_normalise(p) for p in results]


async def _search_by_keyword(
    client: httpx.AsyncClient,
    keyword: str,
    *,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """Search PWC papers by keyword."""
    params: dict[str, Any] = {
        "q": keyword,
        "ordering": "-published",
        "items_per_page": min(max_results, 50),
        "page": 1,
    }

    LOG.info("PWC search: q=%s", keyword)
    try:
        resp = await client.get(f"{PWC_BASE}/search/", params=params)
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        LOG.warning("PWC HTTP %s for keyword '%s': %s",
                    exc.response.status_code, keyword, exc)
        return []
    except httpx.RequestError as exc:
        LOG.warning("PWC request error for keyword '%s': %s", keyword, exc)
        return []

    results = data.get("results", [])
    return [_normalise(p) for p in results]


async def search_papers_with_code(
    keywords: list[str] | None = None,
    *,
    max_results: int = 20,
) -> list[dict[str, Any]]:
    """Query PWC for trending papers and optionally search by keywords."""

    all_papers: list[dict[str, Any]] = []
    seen_titles: set[str] = set()

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1) Trending papers
        trending = await _fetch_trending(client, max_results=max_results)
        for p in trending:
            key = p["title"].lower().strip()
            if key not in seen_titles:
                seen_titles.add(key)
                all_papers.append(p)

        # 2) Keyword-based search
        if keywords:
            for kw in keywords:
                if len(all_papers) >= max_results:
                    break
                results = await _search_by_keyword(
                    client, kw, max_results=10,
                )
                for p in results:
                    key = p["title"].lower().strip()
                    if key not in seen_titles:
                        seen_titles.add(key)
                        all_papers.append(p)

    return all_papers[:max_results]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch papers from Papers With Code",
    )
    parser.add_argument(
        "--interests-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "interests.jsonl",
        help="Path to interests.jsonl (optional, for keyword search)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=PROJECT_ROOT / "data" / "tmp_pwc.json",
    )
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument(
        "--trending-only",
        action="store_true",
        help="Only fetch trending, skip keyword search",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    )

    keywords: list[str] | None = None
    if not args.trending_only and args.interests_file.exists():
        try:
            interests = _load_latest_interests(args.interests_file)
            keywords = _extract_keywords(interests)
            LOG.info("Keywords (%d): %s", len(keywords), keywords[:10])
        except Exception:
            LOG.warning("Could not load interests; falling back to trending only")

    papers = asyncio.run(
        search_papers_with_code(keywords, max_results=args.max_results)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(papers, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOG.info("Wrote %d papers to %s", len(papers), args.output)


if __name__ == "__main__":
    main()
