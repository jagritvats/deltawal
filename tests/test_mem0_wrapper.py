"""Network-free tests for the Mem0 backend: routing through the (fake) router, date pinning, input
truncation, history accounting (ADD -> UPDATE in place), parse-failure counting, tag rendering and
the KU-past evidence record. Uses the real mem0ai 1.0.11 pipeline with an embedded Qdrant in tmp."""
from __future__ import annotations

import hashlib
import importlib.util
import json

import numpy as np
import pytest

from epochmem.backends.mem0_wrapper import (Mem0Backend, assemble_lines, iso_date, mem0_parse_ok,
                                            reader_messages, session_messages)
from epochmem.llm import ChatResult
from epochmem.systems.base import READER_SYSTEM
from epochmem.systems.deltawal import PRECEDENCE, gate_wal

DIM = 16


class FakeRouter:
    """Stands in for ModelRouter: records every call; deterministic JSON replies."""

    def __init__(self):
        self.chats, self.embeds = [], []
        self.last_facts: list[str] = []
        self.garbage_next = False

    def chat(self, model, messages, max_tokens=512, **kw):
        self.chats.append((model, messages))
        content = "\n".join(m["content"] for m in messages)
        if self.garbage_next:
            self.garbage_next = False
            return ChatResult(content="sorry, no json here", cached=False, prompt_eval_count=10, eval_count=3)
        if "smart memory manager" in content:
            new = self.last_facts[0]
            if "'id': '0'" in content:
                out = {"memory": [{"id": "0", "text": new, "event": "UPDATE", "old_memory": "old"}]}
            else:
                out = {"memory": [{"id": "0", "text": new, "event": "ADD"}]}
        else:
            user_lines = [ln[len("user: "):] for ln in messages[-1]["content"].splitlines() if ln.startswith("user: ")]
            self.last_facts = [user_lines[0][:80]] if user_lines else []
            out = {"facts": self.last_facts}
        return ChatResult(content=json.dumps(out), cached=False, prompt_eval_count=100, eval_count=20, duration_s=0.01)

    def embed(self, model, texts, **kw):
        self.embeds.append(list(texts))
        out = []
        for t in texts:
            # topic-ish embedding: memories about the Apex goal land close together
            base = np.ones(DIM) if "Apex" in t else np.zeros(DIM)
            h = np.frombuffer(hashlib.sha256(t.encode()).digest()[:DIM], dtype=np.uint8) / 2550.0
            out.append((base + h).tolist())
        return out


@pytest.fixture
def be(tmp_path):
    r = FakeRouter()
    b = Mem0Backend(r, "opencode-zen/gpt-6-luna", "qwen3-embedding:0.6b", "Instruct: q\nQuery: ", DIM, tmp_path)
    yield b
    b.close_all()


def turns(user, asst="ok"):
    return [{"role": "user", "content": user}, {"role": "assistant", "content": asst}]


def test_add_update_history_and_search(be):
    r = be.router
    s1 = be.add_session("q_A", turns("My Apex Legends goal is level 50.", "x" * 1000), 1, "2023/05/20 (Sat) 02:21")
    assert s1["events"] == {"ADD": 1, "UPDATE": 0, "DELETE": 0}
    # date pinned to the session date, wall-clock date gone; assistant turn truncated like FactExtractor
    extract_prompt = "\n".join(m["content"] for m in r.chats[0][1])
    assert "Today's date is 2023-05-20." in extract_prompt
    assert "x" * 300 + " [...]" in extract_prompt and "x" * 301 not in extract_prompt
    assert "SESSION DATE: 2023/05/20 (Sat) 02:21" in extract_prompt
    s2 = be.add_session("q_A", turns("My Apex Legends goal is now level 100."), 2, "2023/06/01 (Thu) 10:00")
    assert s2["events"]["UPDATE"] == 1
    h = s2["history"][0]
    assert h["event"] == "UPDATE" and "level 50" in h["old"] and "level 100" in h["new"]
    hc = be.history_counts("q_A")
    assert hc["totals"] == {"ADD": 1, "UPDATE": 1, "DELETE": 0}
    assert hc["n_memories_ever"] == 1 and hc["n_updated_memories"] == 1
    assert list(hc["per_memory"].values()) == [{"ADD": 1, "UPDATE": 1, "DELETE": 0}]
    hits = be.search("q_A", "What is my Apex goal?", 5)
    assert len(hits) == 1 and "level 100" in hits[0]["memory"]
    assert hits[0]["source_epoch"] == 2 and hits[0]["session_date"].startswith("2023/06/01")   # UPDATE overwrote metadata
    # every internal call went through the router; search queries carry the instruction prefix
    assert be.stats.calls == len(r.chats) == 4 and be.stats.parse_failures == 0
    assert any(t.startswith("Instruct: q\nQuery: What is my Apex goal?") for batch in r.embeds for t in batch)


def test_snapshot_isolates_stages(be):
    be.add_session("q_A", turns("My Apex Legends goal is level 50."), 1, "2023/05/20 (Sat) 02:21")
    be.snapshot("q_A", "q_B")
    be.add_session("q_B", turns("My Apex Legends goal is now level 100."), 2, "2023/06/01 (Thu) 10:00")
    assert "level 50" in be.search("q_A", "Apex goal", 5)[0]["memory"]
    assert "level 100" in be.search("q_B", "Apex goal", 5)[0]["memory"]
    assert be.history_counts("q_A")["totals"]["UPDATE"] == 0
    assert be.history_counts("q_B")["totals"]["UPDATE"] == 1


def test_parse_failure_is_counted(be):
    be.router.garbage_next = True
    out = be.add_session("q_A", turns("I like tea."), 1, "2023/05/20 (Sat) 02:21")
    assert out["internal"]["parse_failures"] == 1 and out["internal"]["by_kind"]["extract"] == [1, 1]
    assert out["events"] == {"ADD": 0, "UPDATE": 0, "DELETE": 0}


def test_parse_ok_and_helpers():
    assert mem0_parse_ok('```json\n{"facts": ["a"]}\n```', "facts")
    assert mem0_parse_ok('Here you go: {"memory": []}', "memory")
    assert not mem0_parse_ok('{"facts": "a"}', "facts") and not mem0_parse_ok("", "facts")
    assert iso_date("2023/05/20 (Sat) 02:21") == "2023-05-20" and iso_date("unknown") is None
    msgs = session_messages([{"role": "user", "content": "u" * 2500}], "D")
    assert msgs[0] == {"role": "system", "content": "SESSION DATE: D"} and msgs[1]["content"] == "u" * 2000 + " [...]"


def test_tag_rendering_and_wal_gating():
    mems = [{"memory": "Goal is level 100", "source_epoch": 14, "session_date": "2023/06/01"},
            {"memory": "Plays Apex", "source_epoch": 3, "session_date": "2023/05/01"}]
    wal = [{"epoch": 15, "turn": 0, "role": "user", "content": "I updated my goal to level 150", "date": "2023/07/01",
            "embedding": np.array([1, 0, 0], dtype=np.float32)},
           {"epoch": 15, "turn": 1, "role": "assistant", "content": "a" * 900, "date": "2023/07/01",
            "embedding": np.array([0, 1, 0], dtype=np.float32)}]
    gated = gate_wal(np.array([1, 0.1, 0], dtype=np.float32), wal, 0.35, 8)
    assert [(t["epoch"], t["turn"]) for t in gated] == [(15, 0)]
    lines = assemble_lines(mems, "OLDER MEMORY", gated, 2000, 600)
    assert lines == ["[OLDER MEMORY | epoch=3 | 2023/05/01] Plays Apex",
                     "[OLDER MEMORY | epoch=14 | 2023/06/01] Goal is level 100",
                     "[NEWER MESSAGE | epoch=15 | 2023/07/01] [user] I updated my goal to level 150"]
    msgs = reader_messages(lines, "Current date: X\nQuestion: Q\nAnswer:", READER_SYSTEM, PRECEDENCE)
    assert msgs[0]["content"] == READER_SYSTEM + "\n\n" + PRECEDENCE
    assert msgs[1]["content"].startswith("Memory:\n[OLDER MEMORY | epoch=3") and msgs[1]["content"].endswith("Answer:")
    assert assemble_lines([], "MEMORY") == [] and "(none)" in reader_messages([], "Q", READER_SYSTEM, None)[1]["content"]
    assert assemble_lines([mems[0]], "MEMORY") == ["[MEMORY | epoch=14 | 2023/06/01] Goal is level 100"]


def test_ku_past_record_flags_in_place_update():
    spec = importlib.util.spec_from_file_location("run_mem0", "scripts/run_mem0.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class Prep:
        class inst:
            qid, question, answer = "q", "What was my previous goal?", "level 100"
        stream = [type("S", (), {"epoch": e})() for e in range(1, 4)]

        @staticmethod
        def evidence_epochs():
            return [1, 3]

        @staticmethod
        def superseded_evidence(n):
            return []

    meta_a = {"history_totals": {"ADD": 1}, "sessions": [
        {"history": [{"memory_id": "m1", "event": "ADD", "old": None, "new": "goal 100", "epoch": 1}]}]}
    meta_b = {"history_totals": {"ADD": 1, "UPDATE": 1}, "sessions": [
        {"history": [{"memory_id": "m1", "event": "UPDATE", "old": "goal 100", "new": "goal 150", "epoch": 3}]}]}
    rec = mod.ku_past_record(Prep, meta_a, meta_b)
    assert rec["old_value_updated_in_place"] and not rec["old_value_deleted"]
    assert rec["events_by_evidence_epoch"] == {1: {"ADD": 1, "UPDATE": 0, "DELETE": 0},
                                               3: {"ADD": 0, "UPDATE": 1, "DELETE": 0}}
    ow = rec["older_evidence_memory_overwritten"][0]
    assert ow["evidence_write_epoch"] == 1 and ow["gold_in_old"] and not ow["gold_in_new"]
    assert rec["gold_old_value_in_live_store"] is False and rec["gold_old_value_only_in_history"] is True
    # a memory created at a non-evidence epoch but UPDATEd at an earlier evidence epoch also counts
    meta_a2 = {"history_totals": {}, "sessions": [{"history": [
        {"memory_id": "m2", "event": "ADD", "old": None, "new": "plays games", "epoch": 0},
        {"memory_id": "m2", "event": "UPDATE", "old": "plays games", "new": "plays games; goal level 100", "epoch": 1}]}]}
    meta_b2 = {"history_totals": {}, "sessions": [{"history": [
        {"memory_id": "m2", "event": "UPDATE", "old": "plays games; goal level 100",
         "new": "plays games; goal level 100, later level 150", "epoch": 3}]}]}
    rec2 = mod.ku_past_record(Prep, meta_a2, meta_b2)
    assert rec2["old_value_updated_in_place"] and rec2["older_evidence_memory_overwritten"][0]["old_words_retained"] == 1.0
    assert rec2["gold_old_value_in_live_store"] is True and rec2["gold_old_value_only_in_history"] is False
    assert mod.gold_in_text("Yes", "yes") is None and mod.gold_in_text("Premier Silver", "was premier silver status")
