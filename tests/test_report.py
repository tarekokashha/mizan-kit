"""Tests for ledger.report: DatasetReport, audit_frame, summarise, and the
CSV/JSONL sinks that carry the provisional findings disclosure.

RULING 2: audit_frame(df, fps) returns one counts dict for the whole
dataframe, the same shape and meaning as ledger.audit.audit_frame. It is
not renamed to audit_dataframe and does not return a list, so every test
below wraps it in a list before handing it to summarise(), exactly the
way tests/test_audit.py calls the v0 functions.
"""
import math

import numpy as np

from ledger.audit import audit_frame as v0_audit_frame
from ledger.audit import summarise as v0_summarise
from ledger.report import (
    DatasetReport, audit_frame, summarise, write_csv, append_jsonl,
    read_jsonl, PROVISIONAL_HEADER,
)
from ledger.synth import DEFECTS, make_episodes


def _rep(defect):
    df = make_episodes(defect=defect, seed=0)
    return summarise(f"synthetic/{defect or 'clean'}",
                     {"codebase_version": "demo", "fps": 30}, [audit_frame(df, 30.0)])


def test_clean_has_no_flags():
    assert _rep("").flags == ""


def test_each_defect_is_named():
    assert "action_equals_state" in _rep("identity").flags
    assert "bad_dt" in _rep("drops").flags
    assert "stuck_state" in _rep("stuck").flags
    assert "negative_lag" in _rep("swapped").flags
    assert "duplicate_episodes" in _rep("duplicate").flags


def test_confirmed_defaults_empty():
    assert _rep("identity").confirmed == ""


def test_audit_frame_returns_a_single_counts_dict():
    # RULING 2: audit_frame(df, fps) -> dict, matching ledger.audit
    # exactly, not a list of per-episode dicts. A refactor that quietly
    # changed the return shape could still slip past the flag-string
    # tests above if callers wrapped it differently, so the return type
    # and the episode count inside it are pinned directly.
    df = make_episodes(defect="", seed=0)
    result = audit_frame(df, 30.0)
    assert isinstance(result, dict)
    assert result["episodes"] == df["episode_index"].nunique()
    assert result["frames"] == len(df)


def test_csv_carries_the_provisional_header(tmp_path):
    p = tmp_path / "r.csv"
    write_csv([_rep("identity")], p)
    first = p.read_text().splitlines()[0]
    assert first.startswith("#")
    assert "provisional" in first.lower()
    assert "defective" not in p.read_text().lower()


def test_jsonl_round_trips(tmp_path):
    p = tmp_path / "l.jsonl"
    append_jsonl(_rep(""), p)
    append_jsonl(_rep("stuck"), p)
    rows = read_jsonl(p)
    assert len(rows) == 2
    assert rows[1]["flags"] == "stuck_state"


def test_error_report_is_recorded_not_raised():
    r = DatasetReport(repo="acme/gated", error="403 gated")
    assert r.flags == "" and r.error == "403 gated"


def test_provisional_header_text_never_says_defective():
    # Global constraint: the word "defective" must never appear as a
    # label applied to a dataset in any output. PROVISIONAL_HEADER still
    # has to say that no dataset is treated as having a confirmed
    # problem until a human opens it, just without that word.
    assert "defective" not in PROVISIONAL_HEADER.lower()
    assert "provisional" in PROVISIONAL_HEADER.lower()
    assert "confirmed" in PROVISIONAL_HEADER.lower()


def _assert_field_equal(v0_value, new_value, label):
    if isinstance(v0_value, float) or isinstance(new_value, float):
        v0_f, new_f = float(v0_value), float(new_value)
        if math.isnan(v0_f) and math.isnan(new_f):
            return
        assert abs(v0_f - new_f) <= 1e-12, f"{label}: v0={v0_value!r} new={new_value!r}"
    else:
        assert v0_value == new_value, f"{label}: v0={v0_value!r} new={new_value!r}"


def test_report_reproduces_v0_exactly_for_every_defect():
    """The refactor must change nothing observable.

    RULING 6: report.py pools numerators and denominators across
    episodes exactly as ledger.audit.summarise does, rather than
    averaging per-episode fractions, which would silently give a
    different answer whenever episodes have unequal length (the "drops"
    defect and real Hub data both produce that). This test is the
    guarantee that the refactor did not change that: for every defect
    ledger.synth knows how to inject, the untouched v0 path
    (ledger.audit.summarise + ledger.audit.audit_frame) and the new path
    (ledger.report.summarise + ledger.report.audit_frame) are run over
    the identical dataframe, and the resulting flags string and every
    numeric field must match, NaN treated as equal to NaN. If this ever
    fails, the fix belongs in report.py, not in this test.
    """
    numeric_fields = (
        "fps", "episodes_sampled", "frames_sampled", "ts_nonmonotonic_eps",
        "frac_bad_dt", "dt_jitter_ratio", "frame_gap_eps", "stuck_state_frac",
        "identity_frac", "lag_frames", "r_lag0", "r_best", "dup_episode_frac",
    )
    for defect in DEFECTS:
        df = make_episodes(defect=defect, seed=0)
        v0 = v0_summarise("t", {"fps": 30}, [v0_audit_frame(df, 30.0)])
        new = summarise("t", {"fps": 30}, [audit_frame(df, 30.0)])
        assert new.flags == v0.flags, f"defect={defect!r} flags differ: v0={v0.flags!r} new={new.flags!r}"
        assert new.codebase == v0.codebase, f"defect={defect!r} codebase"
        assert new.note == v0.note, f"defect={defect!r} note"
        for field in numeric_fields:
            _assert_field_equal(getattr(v0, field), getattr(new, field), f"defect={defect!r} field={field}")


def test_multi_part_pooling_matches_v0_with_unequal_parts():
    """Multi part pooling must match v0 exactly when parts differ in
    episode count and frame count, not just when a single dict is
    wrapped in a length-1 list.

    Every other summarise() call in this file passes exactly one part,
    so nothing else here would catch a regression in how report.py
    pools MORE THAN ONE part. Four parts are built with different
    n_eps (2, 2, 1, 3), different T (150, 200, 100, 130), and the
    second part uses defect="drops" so its two episodes end up unequal
    length within that part (one dropped, one not), on top of the
    frame count already differing part to part.

    Three parts inject lag=1 and one injects lag=6. v0 pools
    lag_frames as the median over every per-episode lag concatenated
    across all four parts: with these seeds that is
    [1, 1, 1, 1, 1, 6, 6, 6], median 1.0. A naive implementation that
    instead averaged each part's own median (1.0, 1.0, 1.0, 6.0) would
    report 2.25. The assertion below computes that wrong, naive answer
    from the same parts and checks it actually diverges from v0's
    pooled lag_frames, so this test cannot pass without teeth: if a
    future seed or shape change ever made the two coincide, this would
    fail loudly rather than silently stop testing anything.
    """
    specs = [
        dict(n_eps=2, T=150, n_joints=8, lag=1, defect="", seed=101),
        dict(n_eps=2, T=200, n_joints=8, lag=1, defect="drops", seed=102),
        dict(n_eps=1, T=100, n_joints=8, lag=1, defect="", seed=103),
        dict(n_eps=3, T=130, n_joints=8, lag=6, defect="", seed=104),
    ]
    dfs = [make_episodes(fps=30.0, **spec) for spec in specs]
    assert [int(df["episode_index"].nunique()) for df in dfs] == [2, 2, 1, 3], \
        "part shapes must carry different episode counts"

    v0_parts = [v0_audit_frame(df, 30.0) for df in dfs]
    new_parts = [audit_frame(df, 30.0) for df in dfs]
    assert len({p["frames"] for p in v0_parts}) == len(v0_parts), \
        "part shapes must carry different frame counts"

    v0 = v0_summarise("t", {"fps": 30}, v0_parts)
    new = summarise("t", {"fps": 30}, new_parts)

    # Self check: a naive mean of each part's own median must actually
    # diverge from v0's correct pooled median, or these part shapes do
    # not exercise the bug this test targets.
    per_part_medians = [np.median(p["lags"]) for p in v0_parts if p["lags"]]
    naive_lag_frames = float(np.mean(per_part_medians))
    assert abs(naive_lag_frames - v0.lag_frames) > 0.5, (
        "chosen part shapes do not create a pooling divergence: "
        f"naive={naive_lag_frames} pooled(v0)={v0.lag_frames}")

    assert new.flags == v0.flags, f"flags differ: v0={v0.flags!r} new={new.flags!r}"
    numeric_fields = (
        "fps", "episodes_sampled", "frames_sampled", "ts_nonmonotonic_eps",
        "frac_bad_dt", "dt_jitter_ratio", "frame_gap_eps", "stuck_state_frac",
        "identity_frac", "lag_frames", "r_lag0", "r_best", "dup_episode_frac",
    )
    for field in numeric_fields:
        _assert_field_equal(getattr(v0, field), getattr(new, field), f"field={field}")
