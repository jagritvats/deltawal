"""YAML config loading with dotted-path access."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


class Config(dict):
    def get_path(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


def load_config(path: str | Path) -> Config:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with p.open("r", encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))


def repo_path(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def load_dotenv(path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines from a gitignored .env into os.environ (existing vars win)."""
    import os
    p = repo_path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def make_cloud_client(cfg: "Config", cache, session_id: str = "ses_epochmem"):
    """Cloud client for `opencode-go/*` models; raises via the guard unless --allow-cloud."""
    import os
    from .llm import OpenAICompatClient
    load_dotenv()
    c = cfg.get("cloud", {}) or {}
    key = os.environ.get(c.get("api_key_env", "OPENCODE_API_KEY"))
    if not key:
        raise RuntimeError("cloud API key env var not set")
    return OpenAICompatClient(c.get("base_url", "https://opencode.ai/inference/go/openai/v1"), key, cache,
                              seed=cfg["seed"], extra_body=c.get("extra_body") or {}, session_id=session_id)
