#!/usr/bin/env python3
"""DeepSeek-powered search for paper-distill.

Uses the OpenAI-compatible API (DeepSeek) to:
  (a) Generate search queries from user summary + interests + ideas
  (b) Ask the model to recommend 5 papers + 2 AI news items

CLI usage:
    python search/deepseek.py \
        --summary-file data/tmp_summary_2026-02-25.txt \
        --output data/tmp_deepseek.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG = logging.getLogger("search.deepseek")

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# ---------------------------------------------------------------------------
# Helpers
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


def _load_ideas(path: Path) -> list[dict[str, Any]]:
    """Load ideas.jsonl if it exists."""
    if not path.exists():
        return []
    ideas: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        line = line.strip()
        if line:
            try:
                ideas.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return ideas


def _load_summary(path: Path) -> str:
    """Load the daily summary text file."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# DeepSeek interaction
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    return (
        "You are an academic research assistant. Your task is to recommend "
        "recent impactful papers and AI news based on the user's research "
        "interests and current work. Always respond with valid JSON."
    )


def _build_user_prompt(
    summary: str,
    keywords: list[str],
    ideas: list[dict[str, Any]],
) -> str:
    today = date.today().isoformat()

    ideas_text = ""
    if ideas:
        ideas_lines = []
        for idea in ideas[:5]:
            title = idea.get("title", idea.get("idea", ""))
            ideas_lines.append(f"  - {title}")
        ideas_text = "\n".join(ideas_lines)

    return f"""Today is {today}.

## My Research Summary
{summary or "(no summary available)"}

## My Keywords / Interests
{', '.join(keywords)}

## My Recent Ideas
{ideas_text or "(none)"}

---

Based on the above, please:

1. Generate 3-5 specific search queries that would find the most relevant recent papers for my interests.
2. Recommend exactly 5 recent papers (published 2024-2026) that are highly relevant.
3. Recommend exactly 2 recent AI news items or breakthroughs.

Return your response as a JSON object with this exact structure:
{{
  "search_queries": ["query1", "query2", ...],
  "papers": [
    {{
      "title": "...",
      "authors": ["Author A", "Author B"],
      "abstract": "Brief description or abstract",
      "doi": "10.xxx/... or empty string if unknown",
      "arxiv_id": "2502.xxxxx or empty string if unknown",
      "year": 2025,
      "journal": "venue or journal name",
      "why_relevant": "One sentence on why this is relevant"
    }}
  ],
  "ai_news": [
    {{
      "title": "...",
      "summary": "2-3 sentence summary",
      "url": "source URL if known, else empty string"
    }}
  ]
}}

Important: Only recommend papers you are confident actually exist. Include DOI
or arXiv ID when possible. Do NOT fabricate paper details."""


def _parse_response(text: str) -> dict[str, Any]:
    """Extract JSON from the model response, handling markdown fences."""
    # Try to find JSON in code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        text = match.group(1)

    # Strip leading/trailing whitespace
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        LOG.warning("Failed to parse DeepSeek response as JSON")
        return {"papers": [], "ai_news": [], "search_queries": []}


def _normalise_paper(paper: dict[str, Any]) -> dict[str, Any]:
    """Convert a DeepSeek-recommended paper to the unified schema."""
    return {
        "source": "deepseek",
        "doi": paper.get("doi", ""),
        "pmid": "",
        "arxiv_id": paper.get("arxiv_id", ""),
        "title": paper.get("title", ""),
        "authors": paper.get("authors", []),
        "abstract": paper.get("abstract", ""),
        "journal": paper.get("journal", ""),
        "year": paper.get("year"),
        "citation_count": 0,
        "tldr": paper.get("why_relevant", ""),
        "open_access_url": "",
        "mesh_terms": [],
        "categories": [],
        "topic_tags": [],
    }


def _normalise_news(news: dict[str, Any]) -> dict[str, Any]:
    """Convert an AI news item to the unified schema (loosely)."""
    return {
        "source": "deepseek",
        "doi": "",
        "pmid": "",
        "arxiv_id": "",
        "title": news.get("title", ""),
        "authors": [],
        "abstract": news.get("summary", ""),
        "journal": "",
        "year": date.today().year,
        "citation_count": 0,
        "tldr": news.get("summary", ""),
        "open_access_url": news.get("url", ""),
        "mesh_terms": [],
        "categories": ["ai-news"],
        "topic_tags": ["ai-news"],
    }


def search_deepseek(
    summary: str,
    keywords: list[str],
    ideas: list[dict[str, Any]],
    *,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Call DeepSeek to get paper recommendations and AI news.

    Returns a dict with keys: papers, ai_news, search_queries.
    """
    api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        LOG.error("DEEPSEEK_API_KEY not set")
        return {"papers": [], "ai_news": [], "search_queries": []}

    client = OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
    )

    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(summary, keywords, ideas)

    LOG.info("Calling DeepSeek %s ...", DEEPSEEK_MODEL)
    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=4096,
        )
    except Exception:
        LOG.exception("DeepSeek API call failed")
        return {"papers": [], "ai_news": [], "search_queries": []}

    content = response.choices[0].message.content or ""
    LOG.debug("DeepSeek raw response length: %d", len(content))

    parsed = _parse_response(content)
    return parsed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search via DeepSeek for paper recommendations",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Path to daily summary text file",
    )
    parser.add_argument(
        "--interests-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "interests.jsonl",
    )
    parser.add_argument(
        "--ideas-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "ideas.jsonl",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=PROJECT_ROOT / "data" / "tmp_deepseek.json",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    )

    # Load interests
    interests = _load_latest_interests(args.interests_file)
    keywords = _extract_keywords(interests)
    if not keywords:
        LOG.error("No keywords found in %s", args.interests_file)
        sys.exit(1)

    LOG.info("Keywords (%d): %s", len(keywords), keywords[:10])

    # Load summary
    summary = ""
    if args.summary_file and args.summary_file.exists():
        summary = _load_summary(args.summary_file)
        LOG.info("Loaded summary from %s (%d chars)", args.summary_file, len(summary))
    else:
        # Use the summary from interests as fallback
        summary = interests.get("summary", "")

    # Load ideas
    ideas = _load_ideas(args.ideas_file)
    LOG.info("Loaded %d ideas", len(ideas))

    # Call DeepSeek
    result = search_deepseek(summary, keywords, ideas)

    # Normalise output
    papers = [_normalise_paper(p) for p in result.get("papers", [])]
    news = [_normalise_news(n) for n in result.get("ai_news", [])]
    all_items = papers + news

    # Also save search queries for potential downstream use
    output_data = {
        "search_queries": result.get("search_queries", []),
        "papers": all_items,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOG.info(
        "Wrote %d papers + %d news items to %s",
        len(papers), len(news), args.output,
    )


if __name__ == "__main__":
    main()
