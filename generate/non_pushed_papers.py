"""
Process non-pushed papers mentioned in user's research document.

For each paper found in the user's md document that isn't in today's push:
  1. Write a paper reading card to Obsidian
  2. Add to Zotero under the appropriate collection

CLI usage:
  python generate/non_pushed_papers.py \
    --topics data/tmp_topics.json \
    --date   2026-02-25
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from generate.obsidian_note import generate_citekey, write_paper_notes

logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("paper-distill.non-pushed")


# Area -> Zotero collection name mapping
AREA_COLLECTION_MAP: dict[str, str] = {
    "LLM": "LLM & AI",
    "RAG": "LLM & AI",
    "Agents": "LLM & AI",
    "Code Generation": "LLM & AI",
    "Multimodal": "Multimodal",
    "Diffusion": "Generative Models",
    "RL": "Reinforcement Learning",
}


def collect_papers_from_topics(topics_data: dict) -> list[dict]:
    """Extract all papers from topics_data, deduped by citekey."""
    seen: set[str] = set()
    papers: list[dict] = []

    for topic in topics_data.get("topics", []):
        area = topic.get("area", "")
        for paper in topic.get("papers", []):
            if not paper.get("citekey"):
                paper["citekey"] = generate_citekey(paper)
            citekey = paper["citekey"]
            if citekey not in seen:
                seen.add(citekey)
                # Carry area info for Zotero collection mapping
                if not paper.get("_area"):
                    paper["_area"] = area
                papers.append(paper)

    return papers


def write_obsidian_cards(
    papers: list[dict],
    date: str,
    project_root: Path,
) -> list[Path]:
    """Write paper reading cards to Obsidian."""
    output_dir = project_root / "obsidian" / "Paper Distill" / "论文阅读"
    return write_paper_notes(papers, date, output_dir)


async def add_to_zotero(
    papers: list[dict],
    project_root: Path,
) -> list[dict]:
    """Add papers to Zotero under area-specific collections."""
    try:
        from integrations.zotero_api import add_papers_to_zotero_by_data
        return await add_papers_to_zotero_by_data(project_root, papers)
    except ImportError:
        logger.warning("Zotero integration not available, skipping")
        return []
    except Exception as e:
        logger.error("Zotero error: %s", e)
        return []


def process_non_pushed_papers(
    topics_data: dict,
    date: str,
    project_root: Path,
) -> dict[str, int]:
    """
    Main entry: write Obsidian cards for non-pushed papers.
    Returns stats dict.
    """
    papers = collect_papers_from_topics(topics_data)

    if not papers:
        logger.info("No papers found in topics data")
        return {"obsidian_cards": 0}

    paths = write_obsidian_cards(papers, date, project_root)

    return {
        "obsidian_cards": len(paths),
        "papers_found": len(papers),
    }


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process non-pushed papers from research document."
    )
    parser.add_argument("--topics", required=True, help="Path to topics JSON")
    parser.add_argument("--date", required=True, help="Date string (YYYY-MM-DD)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)

    args = parse_args(argv)
    project_root = Path(__file__).resolve().parent.parent

    topics_path = Path(args.topics)
    if not topics_path.is_absolute():
        topics_path = project_root / topics_path

    if not topics_path.exists():
        logger.error("Topics file not found: %s", topics_path)
        sys.exit(1)

    topics_data = json.loads(topics_path.read_text(encoding="utf-8"))
    stats = process_non_pushed_papers(topics_data, args.date, project_root)

    logger.info("Non-pushed papers processing complete: %s", stats)


if __name__ == "__main__":
    main()
