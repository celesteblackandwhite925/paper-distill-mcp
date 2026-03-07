"""Review prompt builder and result parser.

This module does NOT call any AI model directly. It prepares prompts
that the OpenClaw agent uses for review, and parses the results.

Two-stage review:
  Initial selection (初选): AI can PUSH / OVERFLOW / DISCARD each paper
  Final selection (终选): AI can only PUSH / OVERFLOW (no discard)
"""
from __future__ import annotations

import json
import logging
import re

LOG = logging.getLogger("paper_digest.reviewer")


def prepare_initial_review_prompt(
    candidates: list[dict],
    user_prefs: dict,
    history_dois: set[str],
    custom_focus: str = "",
    picks_per_reviewer: int = 3,
    is_final_scan: bool = False,
) -> str:
    """Build initial review prompt. AI picks top papers and can discard irrelevant ones.

    Args:
        candidates: papers to review (today's scan batch)
        picks_per_reviewer: each reviewer picks this many
        is_final_scan: if True, this is the last scan day (review all remaining)
    """
    # Filter out already-pushed
    filtered = []
    for c in candidates:
        doi = (c.get("doi") or "").lower()
        if doi and doi in history_dois:
            continue
        filtered.append(c)

    if not filtered:
        return "NO_CANDIDATES: All candidates have been previously pushed."

    lines = [
        f"# Paper Review - Initial Selection",
        "",
        f"You have {len(filtered)} candidate papers to review.",
        f"Select your top {picks_per_reviewer} most relevant papers.",
        "",
        "For EACH candidate you must decide:",
        f"- **PUSH**: Select for today's digest (pick {picks_per_reviewer})",
        "- **OVERFLOW**: Good paper but not top pick. Keep for next day.",
        "- **DISCARD**: Not relevant enough. Remove from pool permanently.",
        "",
    ]

    if is_final_scan:
        lines.insert(3, "**This is the final scan. All remaining papers must be decided on.**")
        lines.insert(4, "")

    # User context
    topics = user_prefs.get("topics", {})
    if topics:
        lines.append("## User's Research Interests")
        for key, t in topics.items():
            kw = ", ".join(t.get("keywords", []))
            lines.append(f"- **{t.get('label', key)}** (weight {t.get('weight', 1.0)}): {kw}")
        lines.append("")

    if custom_focus:
        lines.append("## Custom Screening Criteria")
        lines.append(custom_focus)
        lines.append("")

    # Candidates
    lines.append("## Candidates")
    lines.append("")

    for i, paper in enumerate(filtered, 1):
        _append_paper_block(lines, i, paper)

    lines.extend([
        "## Response Format",
        "",
        "Reply with a JSON array. For EACH candidate, specify action:",
        "```json",
        "[",
        '  {"index": 1, "action": "push", "reason": "...", "tldr": "..."},',
        '  {"index": 2, "action": "overflow", "reason": "decent but not top"},',
        '  {"index": 3, "action": "discard", "reason": "not relevant"},',
        "  ...",
        "]",
        "```",
        "",
        f"You MUST have exactly {picks_per_reviewer} papers with action=push.",
        "Every candidate must have a decision (push/overflow/discard).",
    ])

    return "\n".join(lines)


def prepare_final_review_prompt(
    reviewer1_picks: list[dict],
    reviewer2_picks: list[dict] | None,
    max_push: int = 6,
) -> str:
    """Build final review prompt. Merge picks from two reviewers.

    In final review, AI can only PUSH or OVERFLOW. No discarding.
    """
    lines = [
        f"# Final Review - Select up to {max_push} papers for today's push",
        "",
    ]

    if reviewer2_picks is not None:
        # Dual review mode
        lines.append("Two reviewers independently selected papers. Make the final call.")
        lines.append("")
        lines.append("## Reviewer 1 Picks")
        for i, p in enumerate(reviewer1_picks, 1):
            lines.append(f"{i}. **{p.get('title', '')}** — {p.get('review_reason', p.get('reason', ''))}")
            if p.get("tldr"):
                lines.append(f"   > {p['tldr']}")
        lines.append("")

        lines.append("## Reviewer 2 Picks")
        for i, p in enumerate(reviewer2_picks, 1):
            lines.append(f"{i}. **{p.get('title', '')}** — {p.get('review_reason', p.get('reason', ''))}")
            if p.get("tldr"):
                lines.append(f"   > {p['tldr']}")
        lines.append("")

        # Overlaps
        r1_dois = {(p.get("doi") or "").lower() for p in reviewer1_picks}
        r2_dois = {(p.get("doi") or "").lower() for p in reviewer2_picks}
        overlap = r1_dois & r2_dois - {""}
        if overlap:
            lines.append(f"**{len(overlap)} paper(s) selected by BOTH reviewers — strongly prioritize.**")
            lines.append("")
    else:
        # Single review mode — just confirm picks
        lines.append("## Selected Papers")
        for i, p in enumerate(reviewer1_picks, 1):
            lines.append(f"{i}. **{p.get('title', '')}** — {p.get('review_reason', p.get('reason', ''))}")
            if p.get("tldr"):
                lines.append(f"   > {p['tldr']}")
        lines.append("")

    lines.extend([
        "## Instructions",
        "",
        f"Select up to {max_push} papers for today's push.",
        "Remaining papers will OVERFLOW to tomorrow (NOT discarded).",
        "",
        "Reply in JSON:",
        "```json",
        '[{"index": 1, "action": "push", "tldr": "..."}, {"index": 2, "action": "overflow"}, ...]',
        "```",
    ])

    return "\n".join(lines)


def _append_paper_block(lines: list[str], idx: int, paper: dict) -> None:
    """Append a formatted paper block to lines."""
    title = paper.get("title", "Untitled")
    year = paper.get("year", "?")
    cites = paper.get("citation_count", 0)
    doi = paper.get("doi", "")
    journal = paper.get("journal", "")
    authors = paper.get("authors", [])
    author_str = ", ".join(authors[:3]) if isinstance(authors, list) else str(authors)
    if isinstance(authors, list) and len(authors) > 3:
        author_str += " et al."
    sources = paper.get("source", "")
    is_overflow = paper.get("pool_status") == "overflow"

    lines.append(f"### [{idx}] {title}")
    meta_parts = [f"{year}"]
    if cites:
        meta_parts.append(f"{cites}x cited")
    if journal:
        meta_parts.append(journal)
    if sources:
        meta_parts.append(f"[{sources}]")
    if is_overflow:
        lines.append(f"**OVERFLOW from previous day — prioritize**")
    lines.append(f"{author_str} | {' | '.join(meta_parts)}")
    if doi:
        lines.append(f"DOI: {doi}")

    summary = paper.get("summary")
    if summary and isinstance(summary, dict):
        for field, value in summary.items():
            if value and field != "custom":
                label = field.replace("_", " ").title()
                lines.append(f"- **{label}**: {value}")
        if summary.get("custom"):
            lines.append(f"- **Custom Focus**: {summary['custom']}")
    elif paper.get("abstract"):
        lines.append(f"> {paper['abstract'][:200]}")
    elif paper.get("tldr"):
        lines.append(f"> {paper['tldr']}")

    lines.append("")


def parse_initial_review(review_text: str, candidates: list[dict]) -> dict:
    """Parse initial review. Returns dict with push/overflow/discard lists.

    Returns:
        {"push": [papers], "overflow": [papers], "discard": [papers]}
    """
    json_match = re.search(r'\[.*\]', review_text, re.DOTALL)
    if not json_match:
        LOG.warning("Could not parse JSON from review result")
        return {"push": [], "overflow": [], "discard": []}

    try:
        decisions = json.loads(json_match.group())
    except json.JSONDecodeError:
        LOG.warning("Invalid JSON in review result")
        return {"push": [], "overflow": [], "discard": []}

    result: dict[str, list[dict]] = {"push": [], "overflow": [], "discard": []}

    for dec in decisions:
        idx = dec.get("index", 0) - 1
        action = dec.get("action", "").lower()
        if 0 <= idx < len(candidates) and action in result:
            paper = dict(candidates[idx])
            paper["review_reason"] = dec.get("reason", "")
            if dec.get("tldr"):
                paper["tldr"] = dec["tldr"]
            result[action].append(paper)

    LOG.info("Initial review: %d push, %d overflow, %d discard",
             len(result["push"]), len(result["overflow"]), len(result["discard"]))
    return result


def parse_final_review(review_text: str, all_picks: list[dict]) -> dict:
    """Parse final review. Returns dict with push/overflow lists (no discard)."""
    json_match = re.search(r'\[.*\]', review_text, re.DOTALL)
    if not json_match:
        LOG.warning("Could not parse JSON from final review")
        return {"push": [], "overflow": []}

    try:
        decisions = json.loads(json_match.group())
    except json.JSONDecodeError:
        LOG.warning("Invalid JSON in final review")
        return {"push": [], "overflow": []}

    result: dict[str, list[dict]] = {"push": [], "overflow": []}

    for dec in decisions:
        idx = dec.get("index", 0) - 1
        action = dec.get("action", "push").lower()
        if action not in ("push", "overflow"):
            action = "overflow"  # Final review cannot discard
        if 0 <= idx < len(all_picks):
            paper = dict(all_picks[idx])
            if dec.get("tldr"):
                paper["tldr"] = dec["tldr"]
            result[action].append(paper)

    LOG.info("Final review: %d push, %d overflow", len(result["push"]), len(result["overflow"]))
    return result
