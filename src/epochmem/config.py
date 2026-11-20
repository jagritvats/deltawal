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


def provider_cfg(cfg: "Config", provider: str) -> dict:
    """Provider settings from `cloud.providers.<name>`; base URL / key come from env vars."""
    import os
    load_dotenv()
    providers = (cfg.get("cloud") or {}).get("providers") or {}
    if provider not in providers:
        raise RuntimeError(f"unknown cloud provider {provider!r}; configure cloud.providers.{provider}")
    pc = dict(providers[provider])
    base = pc.get("base_url") or os.environ.get(pc.get("base_url_env", ""), "")
    key = os.environ.get(pc.get("api_key_env", ""), "")
    if not base or not key:
        raise RuntimeError(f"provider {provider!r}: set {pc.get('base_url_env') or 'base_url'} and {pc.get('api_key_env')}")
    pc.update(base_url=base, api_key=key)
    return pc


def make_cloud_client(cfg: "Config", cache, session_id: str = "ses_epochmem", provider: str = "opencode-go"):
    """Guard-checked OpenAI-compatible client for one provider (raises unless --allow-cloud)."""
    from .llm import OpenAICompatClient
    pc = provider_cfg(cfg, provider)
    return OpenAICompatClient(pc["base_url"], pc["api_key"], cache, seed=cfg["seed"],
                              extra_body=pc.get("extra_body") or {}, session_id=session_id,
                              send_session_header=bool(pc.get("session_header", False)),
                              max_concurrency=int(pc.get("max_concurrency", 2)),
                              prices=pc.get("prices"), budget_usd=pc.get("budget_usd"),
                              extra_body_by_model=pc.get("extra_body_by_model"))
