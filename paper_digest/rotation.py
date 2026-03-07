"""Topic rotation scheduling.

When a user has many research topics (>3), we rotate which topics
get reviewed each day to ensure diverse coverage within a pool cycle.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

LOG = logging.getLogger("paper_digest.rotation")

ROTATION_FILE = "rotation_state.json"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _load_state(data_dir: Path) -> dict:
    path = data_dir / ROTATION_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(data_dir: Path, state: dict) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / ROTATION_FILE
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def plan_rotation(topic_keys: list[str], batch_size: int = 3) -> list[list[str]]:
    """Split topics into rotation batches.

    If topics <= batch_size, single batch with all topics.
    """
    if len(topic_keys) <= batch_size:
        return [topic_keys]
    batches = []
    for i in range(0, len(topic_keys), batch_size):
        batches.append(topic_keys[i:i + batch_size])
    return batches


def get_today_topics(data_dir: Path, topics: dict, batch_size: int = 3) -> list[str]:
    """Get which topics to review today.

    If all topics fit in one day, return all.
    Otherwise, rotate through batches, advancing each day.
    """
    topic_keys = sorted(topics.keys())

    if len(topic_keys) <= batch_size:
        return topic_keys

    state = _load_state(data_dir)
    today = _today()

    # Check if state is stale or topics changed
    cycle_topics = state.get("cycle_topics", [])
    if sorted(cycle_topics) != sorted(topic_keys) or not state.get("current_batch"):
        # Reset rotation with new topic list
        batches = plan_rotation(topic_keys, batch_size)
        state = {
            "last_rotated": today,
            "current_batch": batches[0],
            "remaining_batches": batches[1:],
            "cycle_topics": topic_keys,
        }
        _save_state(data_dir, state)
        LOG.info("Rotation reset: %d topics in %d batches", len(topic_keys), len(batches))
        return state["current_batch"]

    # If already rotated today, return current batch
    if state.get("last_rotated") == today:
        return state["current_batch"]

    # Advance to next batch
    remaining = state.get("remaining_batches", [])
    if remaining:
        state["current_batch"] = remaining[0]
        state["remaining_batches"] = remaining[1:]
    else:
        # Cycle complete, restart
        batches = plan_rotation(topic_keys, batch_size)
        state["current_batch"] = batches[0]
        state["remaining_batches"] = batches[1:]

    state["last_rotated"] = today
    _save_state(data_dir, state)
    LOG.info("Rotated to topics: %s", state["current_batch"])
    return state["current_batch"]


def force_topic(data_dir: Path, topic_key: str) -> None:
    """Force a specific topic into today's batch (for new topics mid-cycle)."""
    state = _load_state(data_dir)
    current = state.get("current_batch", [])
    if topic_key not in current:
        current.append(topic_key)
        state["current_batch"] = current

    # Also add to cycle_topics if not present
    cycle = state.get("cycle_topics", [])
    if topic_key not in cycle:
        cycle.append(topic_key)
        state["cycle_topics"] = cycle

    _save_state(data_dir, state)
    LOG.info("Forced topic '%s' into today's batch: %s", topic_key, current)


def get_rotation_status(data_dir: Path, topics: dict) -> dict:
    """Return current rotation status for display."""
    state = _load_state(data_dir)
    topic_keys = sorted(topics.keys())
    batches = plan_rotation(topic_keys)
    return {
        "total_topics": len(topic_keys),
        "batch_size": 3,
        "total_batches": len(batches),
        "current_batch": state.get("current_batch", topic_keys[:3]),
        "last_rotated": state.get("last_rotated", ""),
        "needs_rotation": len(topic_keys) > 3,
    }
