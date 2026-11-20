"""MemConflict loader on a tiny synthetic record (no network, no real data)."""
from __future__ import annotations

import json

import pytest

from epochmem.data.memconflict import MemConflictDataset, passes_quality_filter
from epochmem.data.stream import prepare


def _q(qid, ctype, question="q?", answer="a."):
    return {"question_id": qid, "question": question, "answer": answer, "conflict_type": ctype,
            "ability_target": "x", "difficulty": "hard"}


def _user(name_line: str) -> dict:
    s0 = {
        "Session_ID": 0, "Date": "2022-01-03", "Session_Type": "initial_reveal",
        "Revealed_Attributes": {"Residence": "Lyon, France", "Social_Relationships": {"Social_Status": {"Current_State": "Normal"}}},
        "Static_Conflict_Information": [
            {"Conflict_ID": "SC_001", "Role": "Point_A", "Target_Field_Path": "Birthplace", "Value": "Nice"},
            {"Conflict_ID": "SC_002", "Role": "Point_A", "Target_Field_Path": "Name", "Value": "Alex Doe"}],
        "Conditional_Conflict_Information": [
            {"Conflict_ID": "CC_001", "Rule_ID": "CC_001_R1", "Role": "Point_A", "Preference_Type": "Food_Preference",
             "Item": "Soup", "Condition": "Cold evenings"}],
        "Session_Questions": [],
        "Session_Dialogue": {
            "dialogue_turn_2": [{"role": "user", "content": "I live in Lyon and was born in Nice."},
                                {"role": "assistant", "content": "Noted."}],
            "dialogue_turn_1": [{"role": "user", "content": name_line},
                                {"assistant": "content", "content": "Hi!"}],          # malformed, repaired
            "dialogue_turn_10": [{"role": "user", "content": "I like soup on cold evenings."},
                                 {"role": "assistant"}],                                # no content, dropped
        },
    }
    s1 = {
        "Session_ID": 1, "Date": "2022-02-01", "Session_Type": "update",
        "Updated_Attributes": [{"Attribute": "Residence", "Before": "Lyon, France", "After": "Oslo, Norway"}],
        "Static_Conflict_Information": [
            {"Conflict_ID": "SC_001", "Role": "Point_B", "Target_Field_Path": "Birthplace", "Value": "Rome"},
            {"Conflict_ID": "SC_002", "Role": "Point_B", "Target_Field_Path": "Name", "Value": "Sam Roe"},
            {"Conflict_ID": "SC_009", "Role": "Distractor", "Target_Field_Path": "Name", "Value": "Kim"}],
        "Conditional_Conflict_Information": [
            {"Conflict_ID": "CC_001", "Rule_ID": "CC_001_R2", "Role": "Point_B", "Preference_Type": "Food_Preference",
             "Item": "Salad", "Condition": "Hot days"}],
        "Session_Questions": [_q("Q_001", "dynamic_conflict", answer="Yes."),
                              _q("Q_002", "dynamic_conflict", answer="From Lyon to Oslo."),
                              _q("Q_003", "static_conflict", answer="Nice."),
                              _q("Q_004", "static_conflict", answer="Alex Doe."),
                              _q("Q_005", "conditional_conflict", answer="Cold evenings.")],
        "Session_Dialogue": {"dialogue_turn_1": [{"role": "user", "content": "I moved to Oslo. I was born in Rome."},
                                                 {"role": "assistant", "content": "Ok."}]},
    }
    s2 = {"Session_ID": 2, "Date": "2022-03-01", "Session_Type": "chitchat", "Session_Questions": [],
          "Static_Conflict_Information": [], "Conditional_Conflict_Information": [],
          "Session_Dialogue": {"dialogue_turn_1": [{"role": "user", "content": "later"}]}}
    return {"ID": "0123abcd-0000-0000-0000-000000000000", "metadata": {"persona_seed": "p"},
            "Full_Session_Chain": [s0, s1, s2]}


def _write(tmp_path, user) -> MemConflictDataset:
    p = tmp_path / "mc.jsonl"
    p.write_text(json.dumps(user) + "\n", encoding="utf-8")
    return MemConflictDataset(p)


def test_memconflict_mapping(tmp_path):
    insts = _write(tmp_path, _user("Hi, I'm Alex Doe.")).load()
    assert [i.qid for i in insts] == [f"mc-0123abcd-s01-Q_00{n}" for n in range(1, 6)]
    assert [i.qtype for i in insts] == ["memconflict-dynamic"] * 2 + ["memconflict-static"] * 2 + ["memconflict-conditional"]
    for i in insts:
        assert [s.session_id for s in i.sessions] == ["0123abcd-S00", "0123abcd-S01"]   # future session excluded
        assert i.question_date == "2022-02-01" and not i.is_abstention
        assert i.evidence_session_ids == ("0123abcd-S00", "0123abcd-S01")
        assert i.meta["aligned"]
    s0 = insts[0].sessions[0]
    assert [(t.role, t.content) for t in s0.turns] == [
        ("user", "Hi, I'm Alex Doe."), ("assistant", "Hi!"),
        ("user", "I live in Lyon and was born in Nice."), ("assistant", "Noted."),
        ("user", "I like soup on cold evenings.")]                                         # numeric turn order
    assert insts[0].meta["conflict"]["update"]["After"] == "Oslo, Norway"
    assert insts[2].meta["evidence_source"]["source_point"]["Value"] == "Nice"
    assert insts[4].meta["evidence_source"]["source_point"]["Item"] == "Soup"
    prep = prepare(insts[0], "post_update")
    assert prep.n == 2 and prep.evidence_epochs() == [1, 2] and prep.cutoff_mode_applied == "post_update"
    assert all(passes_quality_filter(i) for i in insts)


def test_memconflict_name_filter_and_qids(tmp_path):
    ds = _write(tmp_path, _user("Alex Doe is my neighbour."))                  # name rendered as a third party
    insts = ds.load()
    keep = {i.qid: passes_quality_filter(i) for i in insts}
    assert keep["mc-0123abcd-s01-Q_004"] is False and sum(keep.values()) == 4
    sel = MemConflictDataset(ds.path, qids=["mc-0123abcd-s01-Q_005", "mc-0123abcd-s01-Q_001"]).load()
    assert [i.qid for i in sel] == ["mc-0123abcd-s01-Q_005", "mc-0123abcd-s01-Q_001"]
    with pytest.raises(ValueError):
        MemConflictDataset(ds.path, qids=["mc-missing"]).load()


def test_memconflict_rejects_unordered_dates(tmp_path):
    u = _user("I'm Alex Doe.")
    u["Full_Session_Chain"][1]["Date"] = "2021-12-31"
    with pytest.raises(ValueError):
        _write(tmp_path, u).load()
