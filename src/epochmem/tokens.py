"""Model-agnostic prompt token counting (o200k_base) so numbers compare across systems/models.

The server-reported prompt_eval_count is also logged, but it under-counts when the
server reuses a cached KV prefix, so it is not used for the headline metric.
"""
from __future__ import annotations

import functools

import tiktoken


@functools.lru_cache(maxsize=1)
def _enc():
    return tiktoken.get_encoding("o200k_base")


def count_text(text: str) -> int:
    return len(_enc().encode(text))


def count_messages(messages: list[dict]) -> int:
    return sum(count_text(m.get("content", "")) + 4 for m in messages)
