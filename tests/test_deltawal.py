"""Unit tests for DeltaWAL relevance gating (wal_sim_threshold) and shadow masking."""
from __future__ import annotations

import numpy as np

from epochmem.systems.deltawal import gate_wal, shadow_mask


def v(*xs):
    return np.asarray(xs, dtype=np.float32)


def turn(epoch, ti, emb):
    return {"epoch": epoch, "turn": ti, "role": "user", "content": f"t{epoch}.{ti}", "date": "d", "embedding": emb}


def fact(key, epoch, emb):
    return {"entity_key": key, "source_epoch": epoch, "fact_text": key, "session_date": "d", "embedding": emb}


Q = v(1, 0, 0)


def test_gate_threshold_filters_and_orders_chronologically():
    turns = [turn(5, 0, v(0, 1, 0)),          # sim 0
             turn(6, 1, v(1, 1, 0)),          # sim ~0.707
             turn(4, 2, v(1, 0.1, 0))]        # sim ~0.995
    out = gate_wal(Q, turns, threshold=0.5, top_m=8)
    assert [(t["epoch"], t["turn"]) for t in out] == [(4, 2), (6, 1)]
    assert all(t["sim"] >= 0.5 for t in out)


def test_gate_nothing_above_threshold_injects_zero():
    assert gate_wal(Q, [turn(5, 0, v(0, 1, 0)), turn(6, 0, v(0, 0, 1))], threshold=0.35, top_m=8) == []
    assert gate_wal(Q, [], threshold=0.0, top_m=8) == []


def test_gate_top_m_keeps_most_similar():
    turns = [turn(1, 0, v(1, 0.9, 0)), turn(2, 0, v(1, 0.1, 0)), turn(3, 0, v(1, 0.5, 0))]
    out = gate_wal(Q, turns, threshold=0.0, top_m=2)
    assert {t["epoch"] for t in out} == {2, 3}


def test_gate_skips_turns_without_embeddings():
    assert gate_wal(Q, [turn(1, 0, None), turn(2, 0, v(1, 0, 0))], 0.5, 8)[0]["epoch"] == 2


def test_shadow_masks_fact_shadowed_by_newer_turn():
    facts = [fact("user.mortgage.preapproval", 3, v(1, 0.05, 0)), fact("user.dog.name", 5, v(0, 0, 1))]
    turns = [turn(37, 0, v(1, 0, 0))]
    kept, sup = shadow_mask(facts, turns, threshold=0.8)
    assert [f["entity_key"] for f in sup] == ["user.mortgage.preapproval"]
    assert sup[0]["shadowed_by"] == (37, 0) and sup[0]["shadow_sim"] >= 0.8
    assert [f["entity_key"] for f in kept] == ["user.dog.name"]


def test_shadow_never_masks_with_older_or_equal_turn():
    facts = [fact("k", 10, v(1, 0, 0))]
    assert shadow_mask(facts, [turn(9, 0, v(1, 0, 0)), turn(10, 1, v(1, 0, 0))], 0.5) == (facts, [])


def test_shadow_respects_threshold_and_empty_inputs():
    facts = [fact("k", 1, v(1, 1, 0))]                    # sim with turn ~0.707
    assert shadow_mask(facts, [turn(2, 0, v(1, 0, 0))], 0.75)[1] == []
    assert len(shadow_mask(facts, [turn(2, 0, v(1, 0, 0))], 0.70)[1]) == 1
    assert shadow_mask(facts, [], 0.1) == (facts, [])
    assert shadow_mask([], [turn(2, 0, v(1, 0, 0))], 0.1) == ([], [])


# ---------------- Shadow v2 (sentence-level) ----------------
from epochmem.memory.wal_embed import embed_records, split_sentences  # noqa: E402
from epochmem.systems.deltawal import shadow_mask_v2  # noqa: E402


def test_split_sentences():
    assert split_sentences("I moved. My dog is Rex! Is it ok?\nYes it is\nOk") == \
        ["I moved.", "My dog is Rex!", "Is it ok?", "Yes it is"]          # "Ok" < min_chars, dropped
    assert split_sentences("") == []
    assert len(split_sentences(". ".join(["word word"] * 50), max_sents=5)) == 5


def test_shadow_v2_catches_clause_that_turn_level_misses():
    # turn embedding is dominated by unrelated content; one sentence matches the stale fact
    f = fact("user.mortgage.preapproval", 3, v(1, 0, 0))
    t = {**turn(37, 0, v(0.3, 1, 0)), "sent_embeddings": np.stack([v(0, 1, 0), v(1, 0.1, 0)])}
    assert shadow_mask([f], [t], 0.6)[1] == []                       # turn-level: cos ~0.29 -> kept
    kept, sup = shadow_mask_v2([f], [t], 0.6)
    assert kept == [] and sup[0]["shadowed_by"] == (37, 0, 1) and sup[0]["shadow_sim"] > 0.99


def test_shadow_v2_only_newer_turns_and_threshold():
    f = fact("k", 10, v(1, 0, 0))
    older = {**turn(9, 0, v(1, 0, 0)), "sent_embeddings": np.stack([v(1, 0, 0)])}
    assert shadow_mask_v2([f], [older], 0.5)[1] == []
    newer = {**turn(11, 0, v(1, 1, 0)), "sent_embeddings": np.stack([v(1, 1, 0)])}   # cos ~0.707
    assert shadow_mask_v2([f], [newer], 0.75)[1] == [] and len(shadow_mask_v2([f], [newer], 0.70)[1]) == 1


def test_shadow_v2_falls_back_to_turn_embedding():
    f = fact("k", 1, v(1, 0, 0))
    t = turn(2, 0, v(1, 0, 0))            # no sent_embeddings
    assert len(shadow_mask_v2([f], [t], 0.9)[1]) == 1


def test_embed_records_shapes():
    def fake_embed(texts):
        return [[float(len(t)), 1.0, 0.0] for t in texts]
    recs = [{"epoch": 1, "turn": 0, "role": "user", "content": "A b c. D e f!"},
            {"epoch": 1, "turn": 1, "role": "assistant", "content": "x" * 900}]
    cfg = {"raw_turn_user_chars": 2000, "raw_turn_assistant_chars": 600}
    items = embed_records(fake_embed, recs, cfg, dim=3)
    assert len(items) == 2 and recs[0]["sent_embeddings"].shape == (2, 3)
    assert recs[1]["embedding"][0] == len("x" * 600 + " [...]")      # same truncation as the prompt
