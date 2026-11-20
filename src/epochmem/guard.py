"""Endpoint guard: any non-localhost model endpoint requires an explicit opt-in.

Every model/embedding HTTP call goes through `check_url`. Milestone 1 runs with
`allow_cloud=False`, so a misconfigured endpoint fails loudly instead of spending money.
"""
from __future__ import annotations

from urllib.parse import urlparse

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
_allow_cloud = False


class CloudCallBlocked(RuntimeError):
    pass


def set_allow_cloud(value: bool) -> None:
    global _allow_cloud
    _allow_cloud = bool(value)


def allow_cloud() -> bool:
    return _allow_cloud


def is_local(url: str) -> bool:
    host = urlparse(url).hostname or ""
    return host in LOCAL_HOSTS


def check_url(url: str) -> None:
    if not is_local(url) and not _allow_cloud:
        raise CloudCallBlocked(
            f"Refusing non-localhost endpoint {urlparse(url).hostname!r}: pass --allow-cloud to enable."
        )
