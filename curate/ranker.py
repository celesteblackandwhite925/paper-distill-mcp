"""
Paper scoring and ranking engine.

Scoring formula:
  score = relevance(0.55) + recency(0.20) + impact(0.15) + novelty(0.10)

Each component is normalized to [0, 1] then multiplied by its weight.

CLI usage:
  python curate/ranker.py \
    --papers  data/tmp_merged.json \
    --prefs   data/topic_prefs.json \
    --history data/papers.jsonl \
    --output  data/tmp_ranked.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("paper-distill.ranker")

# ---------------------------------------------------------------------------
# Weight defaults (overridable via config)
# ---------------------------------------------------------------------------
W_RELEVANCE = 0.55
W_RECENCY   = 0.20
W_IMPACT    = 0.15
W_NOVELTY   = 0.10

# Recency decay parameters
RECENCY_FULL   = 30    # days: score = 1.0
RECENCY_RECENT = 180   # days: score = 0.8
RECENCY_LAMBDA = 0.005 # exponential decay rate beyond 6 months

# Impact normalisation – rough log-scale ceiling per broad field
FIELD_CITATION_CAPS: dict[str, float] = {
    "medicine":    500,
    "biology":     300,
    "cs":          200,
    "finance":     100,
    "default":     200,
}


# ── helpers ────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> set[str]:
    """Lowercase split + strip punctuation."""
    return set(re.findall(r"[a-z0-9]{2,}", text.lower()))


def _load_jsonl(path: Path) -> list[dict]:
    """Load a .jsonl file; return [] if missing or empty."""
    if not path.exists():
        return []
    entries: list[dict] = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def _guess_field(paper: dict) -> str:
    """Heuristic to bucket a paper into a broad field."""
    topics = " ".join(paper.get("topic_tags", []))
    title  = paper.get("title", "").lower()
    journal = paper.get("journal", "").lower()
    blob = f"{topics} {title} {journal}"

    if any(kw in blob for kw in ("medicine", "clinical", "blood", "leukemia",
                                  "hematol", "cancer", "oncol", "lancet")):
        return "medicine"
    if any(kw in blob for kw in ("biology", "cell", "protein", "peptide",
                                  "genom", "nucleic")):
        return "biology"
    if any(kw in blob for kw in ("trading", "financ", "quant", "stock",
                                  "portfolio")):
        return "finance"
    return "cs"


# ── scoring components ─────────────────────────────────────────────────────

def score_relevance(paper: dict, prefs: dict) -> float:
    """
    Keyword overlap between the paper and the user's topic preferences.

    For each topic the user has defined we compute a Jaccard-like overlap
    between the topic keywords and the paper text, weighted by topic weight.
    The maximum weighted overlap across all (non-blocked) topics is returned.
    """
    paper_tokens = _tokenise(
        " ".join([
            paper.get("title", ""),
            paper.get("abstract", ""),
            " ".join(paper.get("topic_tags", [])),
        ])
    )

    if not paper_tokens:
        return 0.0

    best = 0.0
    topics = prefs.get("topics", {})

    for _key, info in topics.items():
        if info.get("blocked", False):
            continue
        kw_tokens: set[str] = set()
        for kw in info.get("keywords", []):
            kw_tokens |= _tokenise(kw)

        if not kw_tokens:
            continue

        overlap = len(paper_tokens & kw_tokens)
        # Normalise by number of keyword tokens (recall-oriented)
        score = overlap / len(kw_tokens) if kw_tokens else 0.0
        score = min(score, 1.0)
        weight = info.get("weight", 0.5)
        best = max(best, score * weight)

    return min(best, 1.0)


def score_recency(paper: dict, now: datetime | None = None) -> float:
    """
    Time-based score:
      - last 30 days  -> 1.0
      - last 6 months -> 0.8
      - older         -> exponential decay from 0.8
    """
    if now is None:
        now = datetime.now(timezone.utc)

    date_str = paper.get("published_date") or paper.get("date") or ""
    if not date_str:
        return 0.5  # unknown date -> neutral

    try:
        pub = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        # Try YYYY-MM-DD
        try:
            pub = datetime.strptime(date_str[:10], "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except (ValueError, TypeError):
            return 0.5

    days_old = max((now - pub).days, 0)

    if days_old <= RECENCY_FULL:
        return 1.0
    if days_old <= RECENCY_RECENT:
        return 0.8
    # Exponential decay from 0.8
    return 0.8 * math.exp(-RECENCY_LAMBDA * (days_old - RECENCY_RECENT))


def score_impact(paper: dict) -> float:
    """
    log(citations + 1) normalised by field-specific cap.
    """
    citations = paper.get("citation_count", 0) or 0
    field = _guess_field(paper)
    cap = FIELD_CITATION_CAPS.get(field, FIELD_CITATION_CAPS["default"])

    raw = math.log(citations + 1)
    ceiling = math.log(cap + 1)
    return min(raw / ceiling, 1.0)


def score_novelty(paper: dict, seen_dois: set[str]) -> float:
    """
    1.0 if DOI not in history (papers.jsonl), 0.0 if already pushed.
    """
    doi = (paper.get("doi") or "").strip().lower()
    if not doi:
        # No DOI -> treat as novel (benefit of the doubt)
        return 1.0
    return 0.0 if doi in seen_dois else 1.0


# ── main ranking ───────────────────────────────────────────────────────────

def rank_papers(
    papers: list[dict],
    prefs: dict,
    history: list[dict],
    *,
    w_relevance: float = W_RELEVANCE,
    w_recency: float = W_RECENCY,
    w_impact: float = W_IMPACT,
    w_novelty: float = W_NOVELTY,
) -> list[dict]:
    """Score every paper and return the list sorted descending by score."""
    seen_dois: set[str] = set()
    for entry in history:
        doi = (entry.get("doi") or "").strip().lower()
        if doi:
            seen_dois.add(doi)

    now = datetime.now(timezone.utc)

    scored: list[dict] = []
    for paper in papers:
        rel   = score_relevance(paper, prefs)
        rec   = score_recency(paper, now)
        imp   = score_impact(paper)
        nov   = score_novelty(paper, seen_dois)

        total = (
            w_relevance * rel
            + w_recency * rec
            + w_impact  * imp
            + w_novelty * nov
        )

        paper_copy = dict(paper)
        paper_copy["_scores"] = {
            "relevance": round(rel, 4),
            "recency":   round(rec, 4),
            "impact":    round(imp, 4),
            "novelty":   round(nov, 4),
            "total":     round(total, 4),
        }
        scored.append(paper_copy)

    scored.sort(key=lambda p: p["_scores"]["total"], reverse=True)
    return scored


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score and rank candidate papers."
    )
    parser.add_argument(
        "--papers", required=True,
        help="Path to merged candidates JSON file",
    )
    parser.add_argument(
        "--prefs", required=True,
        help="Path to topic_prefs.json",
    )
    parser.add_argument(
        "--history", required=True,
        help="Path to papers.jsonl (previously pushed papers)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to write ranked JSON output",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)

    args = parse_args(argv)
    project_root = Path(__file__).resolve().parent.parent

    # Resolve paths relative to project root when not absolute
    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else project_root / path

    papers_path  = _resolve(args.papers)
    prefs_path   = _resolve(args.prefs)
    history_path = _resolve(args.history)
    output_path  = _resolve(args.output)

    # Load data
    if not papers_path.exists():
        logger.error("Papers file not found: %s", papers_path)
        sys.exit(1)
    if not prefs_path.exists():
        logger.error("Preferences file not found: %s", prefs_path)
        sys.exit(1)

    papers  = json.loads(papers_path.read_text(encoding="utf-8"))
    prefs   = json.loads(prefs_path.read_text(encoding="utf-8"))
    history = _load_jsonl(history_path)

    logger.info(
        "Ranking %d papers (history=%d, topics=%d)",
        len(papers), len(history), len(prefs.get("topics", {})),
    )

    ranked = rank_papers(papers, prefs, history)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(ranked, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "Wrote %d ranked papers to %s (top score=%.4f)",
        len(ranked),
        output_path,
        ranked[0]["_scores"]["total"] if ranked else 0,
    )


if __name__ == "__main__":
    main()
