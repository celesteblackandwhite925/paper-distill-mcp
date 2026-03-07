#!/usr/bin/env python3
"""OpenClaw skill CLI entry point for paper-distill.

No FastMCP dependency — pure CLI that outputs markdown to stdout.

Usage:
    python3 -m paper_digest.openclaw_runner --action daily --config ~/.openclaw/skills/paper-distill/config.json
    python3 -m paper_digest.openclaw_runner --action search --query "LLM reasoning" --top 5
    python3 -m paper_digest.openclaw_runner --action sync-prefs --config ~/.openclaw/skills/paper-distill/config.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Project root for sibling package imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from search.query_all import search_all
from curate.ranker import rank_papers
from curate.filter import filter_papers, load_jsonl

LOG = logging.getLogger("openclaw_runner")

DEFAULT_CONFIG_DIR = Path.home() / ".openclaw" / "skills" / "paper-distill"
DEFAULT_CONFIG = DEFAULT_CONFIG_DIR / "config.json"
DEFAULT_DATA_DIR = DEFAULT_CONFIG_DIR / "data"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    if not path.exists():
        LOG.error("Config not found: %s", path)
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def config_to_topic_prefs(config: dict) -> dict:
    """Convert OpenClaw config.json → topic_prefs.json format."""
    topics = {}
    for key, t in config.get("topics", {}).items():
        topics[key] = {
            "weight": t.get("weight", 1.0),
            "blocked": False,
            "label": t.get("label", key),
            "keywords": t.get("keywords", []),
        }

    pc = config.get("paper_count", {})
    max_total = pc.get("value", 6)

    return {
        "topics": topics,
        "max_per_topic": max(1, max_total // max(len(topics), 1)),
        "max_total": max_total,
    }


def get_data_dir(config: dict) -> Path:
    """Return the data directory, creating it if needed."""
    data_dir = DEFAULT_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_history(data_dir: Path) -> tuple[list[dict], set[str]]:
    """Load papers.jsonl history and extract DOI set."""
    history_path = data_dir / "papers.jsonl"
    history = load_jsonl(history_path)
    dois = {h.get("doi", "").lower() for h in history if h.get("doi")}
    return history, dois


def apply_paper_count(papers: list[dict], config: dict) -> list[dict]:
    """Truncate papers list according to paper_count config."""
    pc = config.get("paper_count", {})
    mode = pc.get("mode", "at_most")
    value = pc.get("value", 6)

    if mode == "at_most":
        return papers[:value]
    elif mode == "exactly":
        return papers[:value]
    elif mode == "at_least":
        # Return at least `value` papers, or all if fewer
        return papers[:max(value, len(papers))]
    return papers[:value]


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_digest_markdown(papers: list[dict], date_str: str) -> str:
    """Format papers as markdown digest for OpenClaw agent output."""
    lines = [f"📬 Paper Distill {date_str} | {len(papers)} papers", ""]

    for i, p in enumerate(papers, 1):
        title = p.get("title", "Untitled")
        year = p.get("year", "")
        citations = p.get("citation_count", 0)
        authors = p.get("authors", [])
        if isinstance(authors, list):
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += " et al."
        else:
            author_str = str(authors)

        doi = p.get("doi", "")
        tldr = p.get("tldr", "") or p.get("abstract", "")[:150]
        if tldr and len(tldr) > 150:
            tldr = tldr[:147] + "..."

        cite_str = f", {citations}x cited" if citations else ""
        year_str = f" ({year}{cite_str})" if year else ""

        lines.append(f"{i}. **{title}**{year_str}")
        if author_str:
            lines.append(f"   {author_str}")
        if tldr:
            lines.append(f"   > {tldr}")
        if doi:
            lines.append(f"   🔗 https://doi.org/{doi}")
        lines.append("")

    return "\n".join(lines)


def format_search_markdown(papers: list[dict], query: str) -> str:
    """Format ad-hoc search results."""
    lines = [f"🔍 Search: \"{query}\" | {len(papers)} results", ""]
    for i, p in enumerate(papers, 1):
        title = p.get("title", "Untitled")
        year = p.get("year", "")
        citations = p.get("citation_count", 0)
        doi = p.get("doi", "")
        tldr = p.get("tldr", "") or p.get("abstract", "")[:120]
        if tldr and len(tldr) > 120:
            tldr = tldr[:117] + "..."

        cite_str = f", {citations}x cited" if citations else ""
        lines.append(f"{i}. **{title}** ({year}{cite_str})")
        if tldr:
            lines.append(f"   > {tldr}")
        if doi:
            lines.append(f"   🔗 https://doi.org/{doi}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

async def action_daily(config: dict) -> str:
    """Run daily digest: search all topics → rank → filter → format."""
    prefs = config_to_topic_prefs(config)
    data_dir = get_data_dir(config)
    history, history_dois = get_history(data_dir)

    # Search all topics
    all_papers: list[dict] = []
    for key, topic in prefs["topics"].items():
        if topic.get("blocked"):
            continue
        query = " ".join(topic["keywords"])
        if not query.strip():
            continue
        LOG.info("Searching topic '%s': %s", topic["label"], query)
        results = await search_all(query, top=20)
        # Tag papers with topic
        for p in results:
            existing_tags = p.get("topic_tags", [])
            if key not in existing_tags:
                p.setdefault("topic_tags", []).append(key)
        all_papers.extend(results)

    if not all_papers:
        return "📭 No papers found for today's digest."

    # Deduplicate across topics (by DOI)
    seen_dois: dict[str, int] = {}
    deduped: list[dict] = []
    for p in all_papers:
        doi = (p.get("doi") or "").lower()
        if doi and doi in seen_dois:
            # Merge topic tags
            idx = seen_dois[doi]
            existing_tags = set(deduped[idx].get("topic_tags", []))
            existing_tags.update(p.get("topic_tags", []))
            deduped[idx]["topic_tags"] = list(existing_tags)
            continue
        if doi:
            seen_dois[doi] = len(deduped)
        deduped.append(p)

    LOG.info("Total %d papers after cross-topic dedup", len(deduped))

    # Rank
    ranked = rank_papers(deduped, prefs, history)

    # Filter already-pushed
    filtered = filter_papers(ranked, history_dois)
    LOG.info("After history filter: %d papers", len(filtered))

    # Apply paper_count
    final = apply_paper_count(filtered, config)

    # Append to local papers.jsonl
    if final:
        jsonl_path = data_dir / "papers.jsonl"
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with open(jsonl_path, "a", encoding="utf-8") as f:
            for p in final:
                record = {
                    "doi": p.get("doi", ""),
                    "title": p.get("title", ""),
                    "authors": p.get("authors", []),
                    "year": p.get("year"),
                    "citation_count": p.get("citation_count", 0),
                    "source": p.get("source", ""),
                    "topic_tags": p.get("topic_tags", []),
                    "push_date": today,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return format_digest_markdown(final, date_str)


async def action_search(query: str, top: int = 5) -> str:
    """Ad-hoc search, no config needed."""
    results = await search_all(query, top=top)
    return format_search_markdown(results[:top], query)


def action_sync_prefs(config: dict) -> str:
    """Sync config.json topics → topic_prefs.json for ranker compatibility."""
    prefs = config_to_topic_prefs(config)
    data_dir = get_data_dir(config)
    prefs_path = data_dir / "topic_prefs.json"
    prefs_path.write_text(json.dumps(prefs, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"✅ Synced {len(prefs['topics'])} topics to {prefs_path}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Paper Distill OpenClaw Runner")
    parser.add_argument("--action", required=True, choices=["daily", "search", "sync-prefs"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    if args.action == "daily":
        config = load_config(args.config)
        output = asyncio.run(action_daily(config))
    elif args.action == "search":
        if not args.query:
            parser.error("--query required for search action")
        output = asyncio.run(action_search(args.query, args.top))
    elif args.action == "sync-prefs":
        config = load_config(args.config)
        output = action_sync_prefs(config)
    else:
        parser.error(f"Unknown action: {args.action}")

    print(output)


if __name__ == "__main__":
    main()
