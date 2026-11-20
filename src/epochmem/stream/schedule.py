"""Deterministic fault schedules: commit reordering (p_ooo) and the lag barrier (k).

Straggler model: each message is independently delayed with probability p_ooo to a
uniformly random later commit slot. The barrier then lets exactly n-k commits land before
the query; the last k in *commit* order are still in flight. No sleeps anywhere: staleness
is a pure function of (n, k, p_ooo, seed, instance id).
"""
from __future__ import annotations

import random
from dataclasses import dataclass


def commit_order(n: int, p_ooo: float, seed_key: str) -> list[int]:
    """Return message indices (0-based, produce order) in the order the worker commits them."""
    rng = random.Random(seed_key)
    slots = []
    for i in range(n):
        delayed = p_ooo > 0 and i < n - 1 and rng.random() < p_ooo
        d = rng.randint(1, n - 1 - i) if delayed else 0
        slots.append((i + d, 1 if delayed else 0, i))
    return [i for _, _, i in sorted(slots)]


def lagged_commit_order(n: int, lag: int, p_ooo: float, seed_key: str) -> list[int]:
    """Commit order with decoupled fault axes.

    The latest `lag` sessions (produce order) are ALWAYS the in-flight tail: they come last, in
    produce order, so barrier(order, lag) never commits them regardless of p_ooo. p_ooo only
    permutes commit order among the n-lag earlier sessions. Stragglers are drawn once per
    (instance, p_ooo) over the full stream and filtered per lag, so the prefix's relative order is
    identical across lag levels.
    """
    lag = max(0, min(lag, n))
    full = commit_order(n, p_ooo, seed_key)
    return [i for i in full if i < n - lag] + list(range(n - lag, n))


@dataclass(frozen=True)
class BarrierState:
    applied: tuple[int, ...]     # indices committed before the query, in commit order
    inflight: tuple[int, ...]    # indices still in flight at query time


def barrier(order: list[int], lag: int) -> BarrierState:
    lag = max(0, min(lag, len(order)))
    cut = len(order) - lag
    return BarrierState(applied=tuple(order[:cut]), inflight=tuple(order[cut:]))


def low_watermark(applied_epochs: set[int]) -> int:
    wm = 0
    while wm + 1 in applied_epochs:
        wm += 1
    return wm
