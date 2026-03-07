"""
Format Telegram/Discord messages for the Paper Distill daily push.

Handles:
  - Individual paper cards (Markdown)
  - Full daily push message
  - Format exactly matching the user's expected digest layout.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("paper-distill.telegram")

def format_paper_card(idx: int, paper: dict) -> str:
    title = paper.get("title", "Untitled")
    year  = paper.get("year", "")
    doi   = paper.get("doi", "")

    # Authors
    authors_raw = paper.get("authors", "")
    if isinstance(authors_raw, list):
        if len(authors_raw) > 2:
            parts = authors_raw[0].split()
            author_str = f"{parts[-1] if parts else authors_raw[0]} et al."
        elif len(authors_raw) == 2:
            author_str = f"{authors_raw[0]} & {authors_raw[1]}"
        else:
            author_str = authors_raw[0] if authors_raw else ""
    else:
        author_str = authors_raw

    lines = [f"*{idx}. {title}*"]
    
    if author_str or year:
        author_line = author_str
        if year:
            author_line = f"{author_line} ({year})" if author_line else str(year)
        lines.append(author_line)

    # Journal info
    journal_parts = []
    if paper.get("journal"):
        journal_parts.append(paper["journal"])
    if paper.get("jcr_quartile"):
        q = paper["jcr_quartile"]
        if paper.get("cas_zone"):
            q += f" (中科院 {paper['cas_zone']} 区)"
        journal_parts.append(q)
    if paper.get("impact_factor"):
        journal_parts.append(f"IF {paper['impact_factor']}")
    if journal_parts:
        lines.append(f"📖 {' | '.join(journal_parts)}")

    # Topic tags
    topic_tags = paper.get("topic_tags", [])
    if topic_tags:
        tags = " ".join(f"#{t}" for t in topic_tags[:3])
        lines.append(tags)

    tldr = paper.get("tldr") or paper.get("one_liner") or ""
    if tldr:
        lines.append(f"💡 {tldr}")

    if doi:
        lines.append(f"[📄 原文](https://doi.org/{doi})")

    return "\n".join(lines)

def format_daily_push(date: str, papers: list[dict], site_url: str = "") -> str:
    header = f"📬 **论文推送 | {date} | {len(papers)} 篇**\n"
    separator = "━━━━━━━━━━━━━━━━━━━━"

    cards = [format_paper_card(i, p) for i, p in enumerate(papers, 1)]
    
    parts = [header, separator, ""]
    parts.extend([c + "\n" for c in cards])
    parts.append(separator)

    if site_url:
        parts.append(f"🔗 [完整网页版阅读]({site_url}/digest/{date})")

    return "\n".join(parts)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", required=True)
    parser.add_argument("--date", required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    papers_path = Path(args.papers)
    if not papers_path.is_absolute():
        papers_path = project_root / papers_path

    papers = json.loads(papers_path.read_text(encoding="utf-8"))
    site_url = os.getenv("SITE_URL", "")

    full_message = format_daily_push(args.date, papers, site_url)
    print(full_message)

if __name__ == "__main__":
    main()
