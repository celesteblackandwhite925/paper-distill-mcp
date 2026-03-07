#!/usr/bin/env python3
"""arXiv API client for paper-distill.

Uses the ``arxiv`` Python package to search for recent papers (last 7 days)
matching user interests in relevant arXiv categories.

CLI usage:
    python search/arxiv_search.py \
        --interests-file data/interests.jsonl \
        --output data/tmp_arxiv.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import arxiv
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG = logging.getLogger("search.arxiv")

# Relevant arXiv categories for the project's interests
DEFAULT_CATEGORIES = [
    "cs.AI",
    "cs.CL",   # Computation and Language (NLP / LLM)
    "cs.LG",   # Machine Learning
    "cs.CV",   # Computer Vision
    "q-bio.QM",  # Quantitative Methods
    "q-bio.GN",  # Genomics
    "q-fin.ST",  # Statistical Finance
    "q-fin.CP",  # Computational Finance
    "stat.ML",   # Machine Learning (stat)
]

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

def _normalise(result: arxiv.Result) -> dict[str, Any]:
    """Convert an arxiv.Result to the unified paper schema."""
    arxiv_id = result.entry_id.split("/abs/")[-1] if result.entry_id else ""
    # Strip version suffix (e.g. v1)
    if arxiv_id and "v" in arxiv_id:
        arxiv_id_base = arxiv_id.rsplit("v", 1)
        if arxiv_id_base[1].isdigit():
            arxiv_id = arxiv_id_base[0]

    authors = [str(a) for a in (result.authors or [])]

    pdf_url = result.pdf_url or ""

    published = result.published
    year = published.year if published else None

    doi = result.doi or ""

    return {
        "source": "arxiv",
        "doi": doi,
        "pmid": "",
        "arxiv_id": arxiv_id,
        "title": result.title or "",
        "authors": authors,
        "abstract": result.summary or "",
        "journal": "",
        "year": year,
        "citation_count": 0,
        "tldr": "",
        "open_access_url": pdf_url,
        "mesh_terms": [],
        "categories": list(result.categories) if result.categories else [],
        "topic_tags": [],
    }


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_arxiv(
    keywords: list[str],
    *,
    categories: list[str] | None = None,
    days_back: int = 7,
    max_results: int = 20,
) -> list[dict[str, Any]]:
    """Search arXiv for recent papers matching keywords."""

    categories = categories or DEFAULT_CATEGORIES
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

    all_papers: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for kw in keywords:
        # Build query: keyword AND (cat1 OR cat2 OR ...)
        cat_query = " OR ".join(f"cat:{c}" for c in categories)
        query = f'all:"{kw}" AND ({cat_query})'

        LOG.info("arXiv query: %s", query)

        client = arxiv.Client(
            page_size=50,
            delay_seconds=3.0,
            num_retries=3,
        )
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        try:
            for result in client.results(search):
                # Filter to recent papers
                if result.published and result.published.replace(tzinfo=timezone.utc) < cutoff:
                    continue

                arxiv_id = result.entry_id.split("/abs/")[-1] if result.entry_id else ""
                if arxiv_id in seen_ids:
                    continue
                seen_ids.add(arxiv_id)

                all_papers.append(_normalise(result))

                if len(all_papers) >= max_results:
                    break
        except Exception:
            LOG.exception("arXiv search failed for keyword '%s'", kw)
            continue

        if len(all_papers) >= max_results:
            break

    LOG.info("arXiv returned %d papers", len(all_papers))
    return all_papers[:max_results]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search arXiv for recent papers",
    )
    parser.add_argument(
        "--interests-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "interests.jsonl",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=PROJECT_ROOT / "data" / "tmp_arxiv.json",
    )
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument(
        "--days-back", type=int, default=7,
        help="How many days back to search (default: 7)",
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

    papers = search_arxiv(
        keywords,
        days_back=args.days_back,
        max_results=args.max_results,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(papers, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOG.info("Wrote %d papers to %s", len(papers), args.output)


if __name__ == "__main__":
    main()
