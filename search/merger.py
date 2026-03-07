#!/usr/bin/env python3
"""Merge and deduplicate search results from multiple sources.

Reads JSON files produced by the individual search modules, deduplicates
by DOI (primary), PMID, or fuzzy title match (>0.85 via rapidfuzz), and
merges metadata using source-specific preferences:
  - Semantic Scholar: preferred for TLDR
  - OpenAlex: preferred for citation_count
  - PubMed: preferred for MeSH terms

CLI usage:
    python search/merger.py \
        --inputs data/tmp_openalex.json data/tmp_s2.json data/tmp_pubmed.json \
        --output data/tmp_merged.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG = logging.getLogger("search.merger")

# Title similarity threshold for fuzzy dedup
TITLE_SIMILARITY_THRESHOLD = 0.85

# Source priority for each field (first = highest priority)
FIELD_SOURCE_PRIORITY: dict[str, list[str]] = {
    "tldr": ["s2", "deepseek", "openalex", "pubmed", "arxiv", "pwc"],
    "citation_count": ["openalex", "s2", "pubmed", "arxiv", "pwc", "deepseek"],
    "mesh_terms": ["pubmed", "openalex", "s2", "arxiv", "pwc", "deepseek"],
    "abstract": ["pubmed", "s2", "openalex", "arxiv", "pwc", "deepseek"],
    "open_access_url": ["arxiv", "openalex", "s2", "pwc", "pubmed", "deepseek"],
    "categories": ["arxiv", "pwc", "s2", "openalex", "pubmed", "deepseek"],
    "authors": ["pubmed", "s2", "openalex", "arxiv", "pwc", "deepseek"],
    "journal": ["pubmed", "openalex", "s2", "arxiv", "pwc", "deepseek"],
}


# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------

def _load_papers(path: Path) -> list[dict[str, Any]]:
    """Load papers from a JSON file. Handles both flat arrays and
    the deepseek format {search_queries: [...], papers: [...]}.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, FileNotFoundError, OSError) as exc:
        LOG.warning("Could not load %s: %s", path, exc)
        return []

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # DeepSeek format
        return data.get("papers", [])
    return []


# ---------------------------------------------------------------------------
# Dedup helpers
# ---------------------------------------------------------------------------

def _normalise_title(title: str) -> str:
    """Lowercase, strip punctuation for comparison."""
    return title.lower().strip().rstrip(".")


def _find_duplicate(
    paper: dict[str, Any],
    merged: list[dict[str, Any]],
    doi_index: dict[str, int],
    pmid_index: dict[str, int],
    title_index: dict[str, int],
) -> int | None:
    """Return the index of a duplicate in merged, or None."""

    # 1) DOI match
    doi = paper.get("doi", "").strip()
    if doi and doi in doi_index:
        return doi_index[doi]

    # 2) PMID match
    pmid = str(paper.get("pmid", "")).strip()
    if pmid and pmid in pmid_index:
        return pmid_index[pmid]

    # 3) Fuzzy title match
    title = _normalise_title(paper.get("title", ""))
    if not title:
        return None

    for norm_title, idx in title_index.items():
        ratio = fuzz.ratio(title, norm_title) / 100.0
        if ratio >= TITLE_SIMILARITY_THRESHOLD:
            return idx

    return None


def _merge_field(
    field: str,
    existing: dict[str, Any],
    new_paper: dict[str, Any],
) -> Any:
    """Decide which value to keep for a field based on source priority."""
    existing_val = existing.get(field)
    new_val = new_paper.get(field)

    # For list fields: merge (union)
    if field in ("mesh_terms", "categories", "topic_tags"):
        combined = list(existing_val or [])
        for item in (new_val or []):
            if item not in combined:
                combined.append(item)
        return combined

    # For scalar fields: prefer non-empty from higher-priority source
    priority = FIELD_SOURCE_PRIORITY.get(field)
    if not priority:
        # No priority defined; keep existing if non-empty, else new
        if existing_val:
            return existing_val
        return new_val

    existing_source = existing.get("source", "")
    new_source = new_paper.get("source", "")

    # Determine which source ranks higher
    def _rank(src: str) -> int:
        try:
            return priority.index(src)
        except ValueError:
            return 999

    existing_rank = _rank(existing_source)
    new_rank = _rank(new_source)

    # If new value is non-empty and from a higher (lower number) priority source
    if _is_nonempty(new_val) and new_rank < existing_rank:
        return new_val
    if _is_nonempty(existing_val):
        return existing_val
    if _is_nonempty(new_val):
        return new_val
    return existing_val


def _is_nonempty(val: Any) -> bool:
    """Check if a value is non-empty (not None, not '', not 0, not [])."""
    if val is None:
        return False
    if isinstance(val, str) and not val.strip():
        return False
    if isinstance(val, list) and len(val) == 0:
        return False
    if isinstance(val, (int, float)) and val == 0:
        return False
    return True


def _merge_papers(
    existing: dict[str, Any],
    new_paper: dict[str, Any],
) -> dict[str, Any]:
    """Merge new_paper into existing, preferring best-source values."""
    result = dict(existing)

    # Track all sources
    sources: set[str] = set()
    if existing.get("source"):
        sources.update(existing["source"].split(","))
    if new_paper.get("source"):
        sources.add(new_paper["source"])
    result["source"] = ",".join(sorted(sources))

    # Merge identifiers (keep any non-empty)
    for id_field in ("doi", "pmid", "arxiv_id"):
        if not result.get(id_field) and new_paper.get(id_field):
            result[id_field] = new_paper[id_field]

    # Merge content fields with priority
    for field in FIELD_SOURCE_PRIORITY:
        result[field] = _merge_field(field, existing, new_paper)

    # Simple fields: keep non-empty
    for field in ("title", "year"):
        if not _is_nonempty(result.get(field)) and _is_nonempty(new_paper.get(field)):
            result[field] = new_paper[field]

    return result


# ---------------------------------------------------------------------------
# Main merge function
# ---------------------------------------------------------------------------

def merge_results(
    input_files: list[Path],
) -> list[dict[str, Any]]:
    """Load, deduplicate, and merge papers from multiple input files."""

    merged: list[dict[str, Any]] = []
    doi_index: dict[str, int] = {}
    pmid_index: dict[str, int] = {}
    title_index: dict[str, int] = {}

    total_loaded = 0

    for fpath in input_files:
        papers = _load_papers(fpath)
        LOG.info("Loaded %d papers from %s", len(papers), fpath.name)
        total_loaded += len(papers)

        for paper in papers:
            dup_idx = _find_duplicate(
                paper, merged, doi_index, pmid_index, title_index,
            )

            if dup_idx is not None:
                # Merge into existing
                merged[dup_idx] = _merge_papers(merged[dup_idx], paper)
                LOG.debug(
                    "Merged duplicate: '%s' (source=%s)",
                    paper.get("title", "")[:60],
                    paper.get("source", ""),
                )
            else:
                # Add new paper
                idx = len(merged)
                merged.append(paper)

                # Update indices
                doi = paper.get("doi", "").strip()
                if doi:
                    doi_index[doi] = idx

                pmid = str(paper.get("pmid", "")).strip()
                if pmid:
                    pmid_index[pmid] = idx

                title = _normalise_title(paper.get("title", ""))
                if title:
                    title_index[title] = idx

    LOG.info(
        "Merge complete: %d input papers -> %d unique papers (%d duplicates removed)",
        total_loaded,
        len(merged),
        total_loaded - len(merged),
    )
    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge and deduplicate search results",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="Input JSON files to merge",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=PROJECT_ROOT / "data" / "tmp_merged.json",
    )
    parser.add_argument(
        "--sort-by",
        choices=["citation_count", "year", "title"],
        default="citation_count",
        help="Sort merged results by this field (descending for numeric)",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    )

    # Validate inputs
    missing = [p for p in args.inputs if not p.exists()]
    if missing:
        LOG.warning("Missing input files (will be skipped): %s",
                     [str(p) for p in missing])

    existing_inputs = [p for p in args.inputs if p.exists()]
    if not existing_inputs:
        LOG.error("No input files found")
        sys.exit(1)

    # Merge
    merged = merge_results(existing_inputs)

    # Sort
    if args.sort_by in ("citation_count", "year"):
        merged.sort(key=lambda p: p.get(args.sort_by, 0) or 0, reverse=True)
    elif args.sort_by == "title":
        merged.sort(key=lambda p: p.get("title", "").lower())

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOG.info("Wrote %d merged papers to %s", len(merged), args.output)


if __name__ == "__main__":
    main()
