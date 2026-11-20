"""Paper assets: tables agree with the numbers manifest, CIs match the report scripts, history-hook versions."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
MANIFEST = ROOT / "paper" / "numbers_manifest.json"
TABLES = ROOT / "paper" / "tables"


def _manifest():
    if not MANIFEST.exists():
        pytest.skip("paper assets not built (run reproduce_assets.sh)")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["metrics"]


def _ci(s):
    return f"{s['mean']:.1f}\\,{{\\scriptsize[{s['lo']:.0f}, {s['hi']:.0f}]}}"


def test_boot_ci_identical_to_report_scripts():
    from build_paper_assets import boot_ci
    from report_heldout import boot_ci as ref
    rng = random.Random(0)
    for n in (5, 13, 60, 138):
        xs = [float(rng.random() < 0.6) for _ in range(n)]
        assert boot_ci(xs) == ref(xs)


def test_cluster_boot_equals_plain_boot_with_one_row_per_question():
    from build_paper_assets import boot_ci, cluster_boot
    rows = [{"qid": f"q{i:03d}", "correct": i % 3 != 0} for i in range(40)]
    assert cluster_boot(rows, lambda r: r["correct"]) == boot_ci([float(r["correct"]) for r in rows])


def test_every_table_cell_is_in_the_manifest():
    m = _manifest()
    checks = {"tab1_main_138.tex": ("tab1/", ("overall", "KU", "temporal", "abstention")),
              "tab3_regression_60.tex": ("tab3/", ("All", "multi-session", "single-session-user",
                                                   "single-session-assistant", "single-session-preference")),
              "tab5_memconflict_90.tex": ("tab5/", ("All", "Dynamic", "Static", "Conditional", "static_false_belief"))}
    for fname, (prefix, keys) in checks.items():
        tex = (TABLES / fname).read_text(encoding="utf-8")
        entries = {k: v for k, v in m.items() if k.startswith(prefix)}
        assert entries, prefix
        for key, st in entries.items():
            for k in keys:
                assert _ci(st[k]) in tex, (fname, key, k)
    tex = (TABLES / "tab2_history_ablation.tex").read_text(encoding="utf-8")
    comb = m["tab2a/destructive_cases_combined"]
    assert f"{comb['mutable']['correct']}/{comb['mutable']['n']} $\\to$ {comb['versioned']['correct']}/" in tex


def test_table1_has_no_stand_in_cells():
    m = _manifest()
    for reader in ("gemma4", "glm-5.3", "gpt-6-luna"):
        for lag in (0, 1, 3):
            assert m[f"tab1/{reader}/deltawal_versioned/lag{lag}"]["overall"]["n"] == 138
    assert "B1_sync_versioned" not in (TABLES / "tab1_main_138.tex").read_text(encoding="utf-8")


def test_history_hook_versions_follow_event_log():
    from run_mem0_history_hook import versions
    events = [{"memory_id": "m1", "event": "ADD", "new": "lives in A", "epoch": 3},
              {"memory_id": "m2", "event": "ADD", "new": "other", "epoch": 4},
              {"memory_id": "m1", "event": "UPDATE", "new": "lives in B", "epoch": 9}]
    hist = [{"event": "ADD", "new_memory": "lives in A"}, {"event": "UPDATE", "new_memory": "lives in B"}]
    assert versions("m1", hist, events) == [(3, "lives in A"), (9, "lives in B")]
    assert versions("m1", hist + [{"event": "DELETE", "new_memory": None}], events)[-1] == (9, "lives in B")
