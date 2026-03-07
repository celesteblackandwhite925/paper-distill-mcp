"""Configuration loader for Paper Distill MCP Server.

Priority: environment variables > config.yaml > defaults.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


_DEFAULT_DATA_DIR = str(Path.home() / ".paper-distill")

_DEFAULTS = {
    "data_dir": _DEFAULT_DATA_DIR,
    "openalex_email": "",
    "deepseek_api_key": "",
    "zotero_library_id": "",
    "zotero_api_key": "",
    "telegram_bot_token": "",
    "telegram_chat_id": "",
    "discord_webhook_url": "",
    "feishu_webhook_url": "",
    "wecom_webhook_url": "",
    "site_url": "",
}

_ENV_MAP = {
    "data_dir": "PAPER_DISTILL_DATA_DIR",
    "openalex_email": "OPENALEX_EMAIL",
    "deepseek_api_key": "DEEPSEEK_API_KEY",
    "zotero_library_id": "ZOTERO_LIBRARY_ID",
    "zotero_api_key": "ZOTERO_API_KEY",
    "telegram_bot_token": "TELEGRAM_BOT_TOKEN",
    "telegram_chat_id": "TELEGRAM_CHAT_ID",
    "discord_webhook_url": "DISCORD_WEBHOOK_URL",
    "feishu_webhook_url": "FEISHU_WEBHOOK_URL",
    "wecom_webhook_url": "WECOM_WEBHOOK_URL",
    "site_url": "SITE_URL",
}


class Config:
    """Immutable configuration object."""

    def __init__(self, values: dict[str, str]):
        self._v = values

    def __getattr__(self, name: str) -> str:
        if name.startswith("_"):
            return super().__getattribute__(name)
        try:
            return self._v[name]
        except KeyError:
            raise AttributeError(f"No config key: {name}")

    @property
    def project_root(self) -> Path:
        return Path(self._v["data_dir"])


_DEFAULT_TOPIC_PREFS = {
    "topics": {
        "example-topic": {
            "weight": 1.0,
            "blocked": False,
            "label": "Example Topic",
            "keywords": ["keyword1", "keyword2", "keyword3"],
        }
    },
    "max_per_topic": 2,
    "max_total": 5,
}


def _ensure_data_dir(data_dir: str) -> None:
    """Create data directory and seed default files if first run."""
    import json

    path = Path(data_dir)
    data_path = path / "data"
    data_path.mkdir(parents=True, exist_ok=True)

    # Seed topic_prefs.json if missing
    prefs = data_path / "topic_prefs.json"
    if not prefs.exists():
        prefs.write_text(
            json.dumps(_DEFAULT_TOPIC_PREFS, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def load_config() -> Config:
    """Load config with priority: env > config.yaml > defaults."""
    values = dict(_DEFAULTS)

    # Try loading config.yaml
    data_dir = os.environ.get("PAPER_DISTILL_DATA_DIR", _DEFAULTS["data_dir"])
    yaml_path = Path(data_dir) / "config" / "config.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f) or {}
        for key in values:
            if key in yaml_data:
                values[key] = str(yaml_data[key])

    # Load .env from data dir
    env_path = Path(data_dir) / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # Environment variables override everything
    for key, env_name in _ENV_MAP.items():
        env_val = os.environ.get(env_name)
        if env_val:
            values[key] = env_val

    # Auto-create data dir on first use
    _ensure_data_dir(values["data_dir"])

    return Config(values)
