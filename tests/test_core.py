"""Pure-Python unit tests: guard, cache, schedule/barrier, stream cutoff, sampler, OOCR."""
from __future__ import annotations

import pytest

from epochmem.cache import DiskCache, cache_key
from epochmem.data.base import Instance, Session, Turn
from epochmem.data.sampler import stratified_sample
from epochmem.data.stream import prepare
from epochmem.guard import CloudCallBlocked, check_url, set_allow_cloud
from epochmem.memory.extract import Fact, normalize_key
from epochmem.stream.schedule import barrier, commit_order, low_watermark
from epochmem.systems.base import compute_oocr


def test_guard_blocks_and_allows():
    set_allow_cloud(False)
    check_url("http://localhost:11434/api/chat")
    check_url("http://127.0.0.1:8000")
    with pytest.raises(CloudCallBlocked):
        check_url("https://api.example.com/v1")
    set_allow_cloud(True)
    check_url("https://api.example.com/v1")
    set_allow_cloud(False)


def test_cache_roundtrip(tmp_path):
    c = DiskCache(tmp_path / "c.sqlite")
    k = cache_key({"model": "m", "messages": [{"role": "user", "content": "x"}], "options": {"seed": 42}})
    assert c.get(k) is None
    c.put(k, {"message": {"content": "y"}})
    assert c.get(k) == {"message": {"content": "y"}}
    assert cache_key({"b": 1, "a": 2}) == cache_key({"a": 2, "b": 1})
    assert cache_key({"a": 1}) != cache_key({"a": 2})


def test_commit_order_deterministic_and_permutation():
    a = commit_order(30, 0.3, "42:q1")
    b = commit_order(30, 0.3, "42:q1")
    assert a == b and sorted(a) == list(range(30))
    assert commit_order(30, 0.0, "42:q1") == list(range(30))
    assert a != list(range(30))


def test_barrier_and_watermark():
    order = [0, 2, 1, 4, 3]
    b = barrier(order, 2)
    assert b.applied == (0, 2, 1) and b.inflight == (4, 3)
    assert low_watermark({1, 2, 3}) == 3
    assert low_watermark({1, 3, 4}) == 1
    assert low_watermark(set()) == 0
    assert barrier(order, 0).inflight == ()


def _inst(qid, qtype, ev_positions, n=6, abs_=False):
    sessions = tuple(Session(f"s{i}", f"2023/01/{i + 1:02d} (Mon) 10:00", (Turn("user", f"t{i}"),)) for i in range(n))
    return Instance(qid=qid, qtype=qtype, question="q", answer="a", question_date="2023/02/01 (Wed) 10:00",
                    sessions=sessions, evidence_session_ids=tuple(f"s{i}" for i in ev_positions), is_abstention=abs_)


def test_cutoff_modes():
    p = prepare(_inst("k1", "knowledge-update", [1, 3]), "post_update")
    assert p.n == 4 and p.cutoff_mode_applied == "post_update" and p.evidence_epochs() == [2, 4]
    assert prepare(_inst("k1", "knowledge-update", [1, 3]), "full_haystack").n == 6
    pa = prepare(_inst("x_abs", "single-session-user", [2], abs_=True), "post_update")
    assert pa.n == 6 and pa.cutoff_mode_applied == "full_haystack"
    pn = prepare(_inst("k2", "knowledge-update", []), "post_update")
    assert pn.n == 6 and pn.cutoff_mode_applied == "full_haystack"
    with pytest.raises(ValueError):
        prepare(_inst("k1", "knowledge-update", [1]), "bogus")


def test_stratified_sample_deterministic():
    pool = [_inst(f"k{i}", "knowledge-update", [1]) for i in range(20)] + \
           [_inst(f"a{i}_abs", "temporal-reasoning", [1], abs_=True) for i in range(10)]
    s1 = stratified_sample(pool, {"knowledge-update": 5, "abstention": 3}, 42)
    s2 = stratified_sample(pool, {"knowledge-update": 5, "abstention": 3}, 42)
    assert [i.qid for i in s1] == [i.qid for i in s2]
    assert sum(i.is_abstention for i in s1) == 3


def test_oocr_definition():
    f = lambda k, e: Fact(k, "x", e, "unknown", "d")
    applied = [f("a", 1), f("a", 3), f("b", 2), f("c", 1), f("c", 2)]
    assert compute_oocr(applied, {"a": 3, "b": 2, "c": 2})[:2] == (2, 0)
    assert compute_oocr(applied, {"a": 1, "b": 2, "c": 2})[:2] == (2, 1)


def test_normalize_key():
    assert normalize_key(" User.Car Model ") == "user.car_model"
    assert normalize_key("..") == "user.misc"



def test_lagged_commit_order_never_commits_the_tail():
    from epochmem.stream.schedule import lagged_commit_order
    for n in (5, 14, 40):
        for lag in (0, 1, 3, 5):
            for p in (0.0, 0.3, 0.9):
                for seed in range(25):
                    order = lagged_commit_order(n, lag, p, f"{seed}:q")
                    b = barrier(order, lag)
                    tail = set(range(max(0, n - lag), n))
                    assert sorted(order) == list(range(n))
                    assert set(b.inflight) == tail and not (set(b.applied) & tail)
                    assert set(b.applied) == set(range(n - min(lag, n)))
                    if lag >= 1:
                        assert n - 1 not in b.applied           # final session never committed at lag>=1


def test_lagged_order_prefix_consistent_across_lags_and_lag0_unchanged():
    from epochmem.stream.schedule import lagged_commit_order
    full = commit_order(30, 0.3, "42:q")
    assert lagged_commit_order(30, 0, 0.3, "42:q") == full
    for lag in (1, 3, 5):
        pre = lagged_commit_order(30, lag, 0.3, "42:q")[: 30 - lag]
        assert pre == [i for i in full if i < 30 - lag]
    assert lagged_commit_order(30, 3, 0.0, "x") == list(range(30))


def test_provider_routing_and_caps():
    from epochmem.config import Config
    from epochmem.llm import provider_of
    from epochmem.runner import enforce_provider_caps
    assert provider_of("gemma4:12b-it-qat") is None
    assert provider_of("opencode-go/glm-5.3") == "opencode-go"
    assert provider_of("openai-compat/some-model") == "openai-compat"
    cfg = Config({"models": {"reader": "opencode-go/glm-5.3", "judge": "gemma4:12b", "extractor": "gemma4:12b"},
                  "cloud": {"providers": {"opencode-go": {"max_instances": 20}, "openai-compat": {}}}})
    enforce_provider_caps(cfg, 20)
    with pytest.raises(RuntimeError):
        enforce_provider_caps(cfg, 138)
    cfg["models"]["reader"] = "openai-compat/some-model"
    enforce_provider_caps(cfg, 138)
