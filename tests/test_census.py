"""Tests for ledger.census: seeded sampling, the resume ledger, and the
two tier runner.

No test here touches the network. LocalSource and small hand-built
fixtures stand in for the Hub throughout, and build_frame is exercised
with a tiny fake client instead of ledger.hubclient.HubClient.
"""
import json

import pytest

from ledger.census import (
    CensusConfig, build_frame, draw_sample, load_done,
    run_deep_tier, run_metadata_tier, prevalence,
)
from ledger.report import DatasetReport
from ledger.sources import LocalSource
from ledger.synth import make_episodes, write_v30_fixture


def test_sample_is_deterministic_and_order_independent():
    frame = [f"o/d{i}" for i in range(500)]
    a = draw_sample(frame, 50, seed=1)
    b = draw_sample(list(reversed(frame)), 50, seed=1)
    assert a == b
    assert len(set(a)) == 50


def test_sample_smaller_than_request_returns_all():
    assert len(draw_sample(["a", "b"], 50, seed=1)) == 2


def test_sample_different_seeds_can_differ():
    frame = [f"o/d{i}" for i in range(500)]
    a = draw_sample(frame, 50, seed=1)
    b = draw_sample(frame, 50, seed=2)
    assert a != b


def test_resume_skips_completed(tmp_path):
    p = tmp_path / "ledger.jsonl"
    p.write_text(json.dumps({"repo": "acme/x", "revision": "abc"}) + "\n")
    assert load_done(p) == {"acme/x@abc"}


def test_load_done_missing_file_is_empty(tmp_path):
    assert load_done(tmp_path / "nope.jsonl") == set()


def test_load_done_skips_a_truncated_final_line(tmp_path):
    # A crash mid write leaves exactly this: a complete line followed by
    # a partial one with no closing brace or trailing newline. That
    # must be skipped, not raised on, so a resumed run only loses the
    # one dataset that was being written when the crash happened.
    p = tmp_path / "ledger.jsonl"
    good = json.dumps({"repo": "acme/x", "revision": "abc"})
    truncated = '{"repo": "acme/y", "revi'
    p.write_text(good + "\n" + truncated)
    assert load_done(p) == {"acme/x@abc"}


def test_deep_tier_writes_one_record_per_dataset(tmp_path):
    root = tmp_path / "repos"
    for name in ["acme/a", "acme/b"]:
        write_v30_fixture(make_episodes(n_eps=2, T=40, seed=1), root, name)
    out = tmp_path / "deep.jsonl"
    cfg = CensusConfig(out_dir=tmp_path, files_per_dataset=1)
    n = run_deep_tier(["acme/a", "acme/b"], LocalSource(root), cfg, out)
    assert n == 2
    assert len(out.read_text().strip().splitlines()) == 2


def test_deep_tier_survives_a_broken_dataset(tmp_path):
    root = tmp_path / "repos"
    write_v30_fixture(make_episodes(n_eps=2, T=40, seed=1), root, "acme/ok")
    out = tmp_path / "deep.jsonl"
    cfg = CensusConfig(out_dir=tmp_path, files_per_dataset=1)
    n = run_deep_tier(["acme/missing", "acme/ok"], LocalSource(root), cfg, out)
    assert n == 2  # both recorded, one carrying an error
    rows = [json.loads(l) for l in out.read_text().strip().splitlines()]
    assert any(r["error"] for r in rows)
    assert any(not r["error"] for r in rows)


def test_deep_tier_flushes_every_dataset_not_only_at_the_end(tmp_path):
    # A crash after the first dataset must not cost the first dataset's
    # record. Read the file back after only part of the run to prove
    # each append lands on disk immediately rather than being batched.
    root = tmp_path / "repos"
    for name in ["acme/a", "acme/b", "acme/c"]:
        write_v30_fixture(make_episodes(n_eps=1, T=40, seed=1), root, name)
    out = tmp_path / "deep.jsonl"
    cfg = CensusConfig(out_dir=tmp_path, files_per_dataset=1)
    source = LocalSource(root)
    seen_after_first = []

    real_frames = source.frames

    def spying_frames(repo, path):
        if repo == "acme/b":
            # by the time the second dataset is being read, the first
            # dataset's record must already be on disk
            seen_after_first.append(out.read_text().strip().splitlines())
        return real_frames(repo, path)

    source.frames = spying_frames
    n = run_deep_tier(["acme/a", "acme/b", "acme/c"], source, cfg, out)
    assert n == 3
    assert len(seen_after_first) == 1
    assert len(seen_after_first[0]) == 1  # exactly acme/a's record, flushed already


def test_deep_tier_resume_skips_repos_already_in_done(tmp_path):
    root = tmp_path / "repos"
    for name in ["acme/a", "acme/b"]:
        write_v30_fixture(make_episodes(n_eps=1, T=40, seed=1), root, name)
    out = tmp_path / "deep.jsonl"
    cfg = CensusConfig(out_dir=tmp_path, files_per_dataset=1)
    done = {"acme/a@"}
    n = run_deep_tier(["acme/a", "acme/b"], LocalSource(root), cfg, out, done=done)
    assert n == 1
    rows = [json.loads(l) for l in out.read_text().strip().splitlines()]
    assert [r["repo"] for r in rows] == ["acme/b"]


def test_metadata_tier_writes_one_record_per_dataset(tmp_path):
    root = tmp_path / "repos"
    for name in ["acme/a", "acme/b"]:
        write_v30_fixture(make_episodes(n_eps=1, T=40, seed=1), root, name)
    out = tmp_path / "meta.jsonl"
    n = run_metadata_tier(["acme/a", "acme/b"], LocalSource(root), out)
    assert n == 2
    rows = [json.loads(l) for l in out.read_text().strip().splitlines()]
    assert [r["repo"] for r in rows] == ["acme/a", "acme/b"]
    assert all(r["codebase"] == "v3.0" for r in rows)
    assert all(not r["error"] for r in rows)


def test_metadata_tier_records_error_for_missing_info(tmp_path):
    root = tmp_path / "repos"
    out = tmp_path / "meta.jsonl"
    n = run_metadata_tier(["acme/missing"], LocalSource(root), out)
    assert n == 1
    rows = [json.loads(l) for l in out.read_text().strip().splitlines()]
    assert rows[0]["error"]


def test_metadata_tier_resume_skips_completed(tmp_path):
    root = tmp_path / "repos"
    for name in ["acme/a", "acme/b"]:
        write_v30_fixture(make_episodes(n_eps=1, T=40, seed=1), root, name)
    out = tmp_path / "meta.jsonl"
    n = run_metadata_tier(["acme/a", "acme/b"], LocalSource(root), out, done={"acme/a@"})
    assert n == 1
    rows = [json.loads(l) for l in out.read_text().strip().splitlines()]
    assert [r["repo"] for r in rows] == ["acme/b"]


class _FakeClient:
    def __init__(self, ids):
        self._ids = ids

    def iter_datasets(self):
        for i in self._ids:
            yield {"id": i}


def test_build_frame_returns_every_repo_id():
    client = _FakeClient(["a/1", "a/2", "a/3"])
    assert build_frame(client) == ["a/1", "a/2", "a/3"]


def test_build_frame_respects_max_datasets():
    client = _FakeClient(["a/1", "a/2", "a/3"])
    assert build_frame(client, max_datasets=2) == ["a/1", "a/2"]


def test_prevalence_carries_a_confidence_interval():
    reps = [DatasetReport(repo=f"o/d{i}", flags="stuck_state" if i < 20 else "")
            for i in range(100)]
    rate, lo, hi = prevalence(reps, "stuck_state")
    assert rate == 0.2
    assert lo < 0.2 < hi
    assert hi - lo > 0.0


def test_prevalence_matches_flags_joined_by_pipe():
    reps = [
        DatasetReport(repo="o/a", flags="bad_dt|stuck_state"),
        DatasetReport(repo="o/b", flags="bad_dt"),
        DatasetReport(repo="o/c", flags=""),
    ]
    rate, lo, hi = prevalence(reps, "stuck_state")
    assert rate == pytest.approx(1 / 3)


def test_prevalence_empty_reports_is_nan_not_a_crash():
    import math
    rate, lo, hi = prevalence([], "stuck_state")
    assert math.isnan(rate) and math.isnan(lo) and math.isnan(hi)
