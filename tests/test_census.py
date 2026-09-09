"""Tests for ledger.census: seeded sampling, the resume ledger, and the
two tier runner.

No test here touches the network. LocalSource and small hand-built
fixtures stand in for the Hub throughout, and build_frame is exercised
with a tiny fake client instead of ledger.hubclient.HubClient.
"""

import json

import pytest

from ledger.census import (
    CensusConfig,
    build_frame,
    draw_sample,
    load_done,
    prevalence,
    run_census,
    run_deep_tier,
    run_metadata_tier,
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


def test_load_done_excludes_rate_limited_records_so_resume_retries_them(tmp_path):
    # CRITICAL 2: an exhausted-backoff 429 is recorded with an error, but
    # it is not genuinely done the way a successful record or a real
    # 401/403/404 is. --resume must see it as still outstanding.
    p = tmp_path / "ledger.jsonl"
    rows = [
        {"repo": "acme/ok", "revision": "r1", "error": ""},
        {"repo": "acme/missing", "revision": "", "error": "no info.json"},
        {
            "repo": "acme/busy",
            "revision": "",
            "error": "RateLimited: gave up on https://x after 5 attempts: 429",
        },
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    done = load_done(p)
    assert "acme/ok@r1" in done
    assert "acme/missing@" in done
    assert "acme/busy@" not in done


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
    rows = [json.loads(line) for line in out.read_text().strip().splitlines()]
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
    rows = [json.loads(line) for line in out.read_text().strip().splitlines()]
    assert [r["repo"] for r in rows] == ["acme/b"]


class _NoExistsLocalSource:
    """LocalSource minus its exists() method: standing in for a
    SampleSource that cannot check whether a path is present (IMPORTANT
    3's fallback case), while still serving real fixture data so the
    audit itself succeeds.
    """

    def __init__(self, root):
        self._local = LocalSource(root)

    def info(self, repo):
        return self._local.info(repo)

    def parquet_paths(self, repo, info, limit=None):
        return self._local.parquet_paths(repo, info, limit)

    def frames(self, repo, path):
        return self._local.frames(repo, path)


def test_deep_tier_notes_a_partial_sample_when_the_source_cannot_check_existence(tmp_path):
    # IMPORTANT 3: --files all (files_per_dataset=None) on a packed
    # dataset, through a source that cannot answer whether a path
    # exists, must not silently claim full coverage. The report's note
    # says so.
    root = tmp_path / "repos"
    write_v30_fixture(make_episodes(n_eps=1, T=40, seed=1), root, "acme/a")
    out = tmp_path / "deep.jsonl"
    cfg = CensusConfig(out_dir=tmp_path, files_per_dataset=None)
    n = run_deep_tier(["acme/a"], _NoExistsLocalSource(root), cfg, out)
    assert n == 1
    rows = [json.loads(line) for line in out.read_text().strip().splitlines()]
    assert not rows[0]["error"]
    assert "partial" in rows[0]["note"]


def test_deep_tier_does_not_note_partial_when_files_per_dataset_is_bounded(tmp_path):
    root = tmp_path / "repos"
    write_v30_fixture(make_episodes(n_eps=1, T=40, seed=1), root, "acme/a")
    out = tmp_path / "deep.jsonl"
    cfg = CensusConfig(out_dir=tmp_path, files_per_dataset=1)
    n = run_deep_tier(["acme/a"], _NoExistsLocalSource(root), cfg, out)
    assert n == 1
    rows = [json.loads(line) for line in out.read_text().strip().splitlines()]
    assert rows[0]["note"] == ""


def test_deep_tier_does_not_note_partial_when_the_source_can_check_existence(tmp_path):
    # A real LocalSource has exists(); files_per_dataset=None must not
    # be flagged partial just because the layout is packed.
    root = tmp_path / "repos"
    write_v30_fixture(make_episodes(n_eps=1, T=40, seed=1), root, "acme/a")
    out = tmp_path / "deep.jsonl"
    cfg = CensusConfig(out_dir=tmp_path, files_per_dataset=None)
    n = run_deep_tier(["acme/a"], LocalSource(root), cfg, out)
    assert n == 1
    rows = [json.loads(line) for line in out.read_text().strip().splitlines()]
    assert rows[0]["note"] == ""


def test_metadata_tier_writes_one_record_per_dataset(tmp_path):
    # CRITICAL 1: tier 1 is sold (README, design doc section 3) as
    # recording codebase version, fps, episode and frame counts, chunk
    # size, feature schema and layout family. Every one of those fields
    # must actually land on the record, not just codebase/fps/source.
    root = tmp_path / "repos"
    for name in ["acme/a", "acme/b"]:
        write_v30_fixture(make_episodes(n_eps=1, T=40, seed=1), root, name)
        info_path = root / name / "meta/info.json"
        info = json.loads(info_path.read_text())
        info["features"] = {"action": {}, "observation.state": {}, "timestamp": {}}
        info_path.write_text(json.dumps(info))
    out = tmp_path / "meta.jsonl"
    n = run_metadata_tier(["acme/a", "acme/b"], LocalSource(root), out)
    assert n == 2
    rows = [json.loads(line) for line in out.read_text().strip().splitlines()]
    assert [r["repo"] for r in rows] == ["acme/a", "acme/b"]
    assert all(r["codebase"] == "v3.0" for r in rows)
    assert all(not r["error"] for r in rows)
    for r in rows:
        assert r["total_episodes"] == 1
        assert r["total_frames"] == 40
        assert r["chunk_size"] == 1000
        assert r["layout_family"] == "packed"
        assert r["feature_names"] == "action;observation.state;timestamp"
        assert r["n_features"] == 3


def test_metadata_tier_tolerates_a_missing_metadata_key(tmp_path):
    # A repo whose info.json is missing one of the newly added keys must
    # still get a record, with that field left at its dataclass default
    # rather than raising and losing the whole record.
    root = tmp_path / "repos"
    write_v30_fixture(make_episodes(n_eps=1, T=40, seed=1), root, "acme/a")
    info_path = root / "acme/a/meta/info.json"
    info = json.loads(info_path.read_text())
    del info["total_frames"]  # simulate a key the Hub happens not to carry
    info_path.write_text(json.dumps(info))
    out = tmp_path / "meta.jsonl"
    n = run_metadata_tier(["acme/a"], LocalSource(root), out)
    assert n == 1
    rows = [json.loads(line) for line in out.read_text().strip().splitlines()]
    assert not rows[0]["error"]
    assert rows[0]["total_frames"] == 0  # dataclass default, not a raise
    assert rows[0]["total_episodes"] == 1  # the key that *was* present


def test_metadata_tier_records_a_distinguishable_error_when_the_hub_rate_limits(
    monkeypatch, tmp_path
):
    # CRITICAL 2 end to end, through the real HubClient with only _open
    # monkeypatched (no network touched): before the fix, get_info
    # swallowed RateLimited to None and this landed here as the same
    # "no info.json" a genuine 401/403/404 gets, so --resume could never
    # tell the two apart. It must now be distinguishable.
    import urllib.error

    from ledger.hubclient import HubClient
    from ledger.sources import StreamingSource

    def opener(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 429, "rate", {"Retry-After": "0"}, None)

    client = HubClient(min_interval=0.0, max_retries=2)
    monkeypatch.setattr(client, "_open", opener)
    source = StreamingSource(client=client)
    out = tmp_path / "meta.jsonl"

    n = run_metadata_tier(["acme/busy"], source, out)

    assert n == 1
    rows = [json.loads(line) for line in out.read_text().strip().splitlines()]
    assert rows[0]["error"]
    assert rows[0]["error"] != "no info.json"
    assert "RateLimited" in rows[0]["error"]


def test_metadata_tier_records_unknown_layout_family_without_failing_the_dataset(tmp_path):
    # An unrecognised data_path template makes ledger.paths.layout_family
    # raise ValueError. That must degrade to layout_family="unknown", not
    # to a failed dataset: every other field info.json can supply should
    # still land on the record, and error must stay empty.
    root = tmp_path / "repos"
    write_v30_fixture(make_episodes(n_eps=1, T=40, seed=1), root, "acme/a")
    info_path = root / "acme/a/meta/info.json"
    info = json.loads(info_path.read_text())
    info["data_path"] = "weird/{nope}.parquet"
    info_path.write_text(json.dumps(info))
    out = tmp_path / "meta.jsonl"
    n = run_metadata_tier(["acme/a"], LocalSource(root), out)
    assert n == 1
    rows = [json.loads(line) for line in out.read_text().strip().splitlines()]
    assert not rows[0]["error"]
    assert rows[0]["layout_family"] == "unknown"
    assert rows[0]["total_episodes"] == 1  # rest of the record is unaffected


def test_metadata_tier_records_error_for_missing_info(tmp_path):
    root = tmp_path / "repos"
    out = tmp_path / "meta.jsonl"
    n = run_metadata_tier(["acme/missing"], LocalSource(root), out)
    assert n == 1
    rows = [json.loads(line) for line in out.read_text().strip().splitlines()]
    assert rows[0]["error"]


def test_metadata_tier_resume_skips_completed(tmp_path):
    root = tmp_path / "repos"
    for name in ["acme/a", "acme/b"]:
        write_v30_fixture(make_episodes(n_eps=1, T=40, seed=1), root, name)
    out = tmp_path / "meta.jsonl"
    n = run_metadata_tier(["acme/a", "acme/b"], LocalSource(root), out, done={"acme/a@"})
    assert n == 1
    rows = [json.loads(line) for line in out.read_text().strip().splitlines()]
    assert [r["repo"] for r in rows] == ["acme/b"]


class _RevisionedLocalSource:
    """LocalSource plus a fixed revision() answer, standing in for a Hub
    backed source whose revision() proxies HubClient.get_revision after
    a captured X-Repo-Commit header (Ruling 9). Kept offline: no source
    here ever touches the network, only a LocalSource fixture wrapped
    with a canned revision.
    """

    def __init__(self, root, sha):
        self._local = LocalSource(root)
        self._sha = sha

    def info(self, repo):
        return self._local.info(repo)

    def parquet_paths(self, repo, info, limit=None):
        return self._local.parquet_paths(repo, info, limit)

    def frames(self, repo, path):
        return self._local.frames(repo, path)

    def revision(self, repo):
        return self._sha


_SHA = "f641879e22172be7e8161d5e6c1503c2d2feb657"


def test_deep_tier_records_the_real_revision_on_the_report(tmp_path):
    root = tmp_path / "repos"
    write_v30_fixture(make_episodes(n_eps=1, T=40, seed=1), root, "acme/a")
    out = tmp_path / "deep.jsonl"
    cfg = CensusConfig(out_dir=tmp_path, files_per_dataset=1)
    n = run_deep_tier(["acme/a"], _RevisionedLocalSource(root, _SHA), cfg, out)
    assert n == 1
    rows = [json.loads(line) for line in out.read_text().strip().splitlines()]
    assert rows[0]["revision"] == _SHA


def test_deep_tier_resume_key_is_repo_at_sha_not_bare(tmp_path):
    root = tmp_path / "repos"
    write_v30_fixture(make_episodes(n_eps=1, T=40, seed=1), root, "acme/a")
    out = tmp_path / "deep.jsonl"
    cfg = CensusConfig(out_dir=tmp_path, files_per_dataset=1)
    run_deep_tier(["acme/a"], _RevisionedLocalSource(root, _SHA), cfg, out)
    done = load_done(out)
    assert done == {f"acme/a@{_SHA}"}
    assert "acme/a@" not in done


def test_deep_tier_still_resumes_correctly_when_the_revision_is_known_upfront(tmp_path):
    # If the caller already knows the revision before this call (e.g. a
    # metadata tier pass warmed the same HubClient earlier), the
    # pre-fetch skip check must use it, not silently fall back to a
    # bare repo@ that could never match a done set keyed by sha.
    root = tmp_path / "repos"
    write_v30_fixture(make_episodes(n_eps=1, T=40, seed=1), root, "acme/a")
    out = tmp_path / "deep.jsonl"
    cfg = CensusConfig(out_dir=tmp_path, files_per_dataset=1)
    done = {f"acme/a@{_SHA}"}
    n = run_deep_tier(["acme/a"], _RevisionedLocalSource(root, _SHA), cfg, out, done=done)
    assert n == 0
    assert not out.exists()  # nothing was skipped-but-recorded; append_jsonl never ran


def test_metadata_tier_records_the_real_revision_on_the_report(tmp_path):
    root = tmp_path / "repos"
    write_v30_fixture(make_episodes(n_eps=1, T=40, seed=1), root, "acme/a")
    out = tmp_path / "meta.jsonl"
    n = run_metadata_tier(["acme/a"], _RevisionedLocalSource(root, _SHA), out)
    assert n == 1
    done = load_done(out)
    assert done == {f"acme/a@{_SHA}"}


def test_local_source_without_a_revision_method_still_keys_repo_at_bare(tmp_path):
    # LocalSource itself exposes no revision() (Ruling 9 scopes the new
    # capability to the Hub backed sources only). Nothing here should
    # raise, and the resume key stays repo@ exactly as before, so every
    # existing LocalSource based resume test keeps its meaning.
    root = tmp_path / "repos"
    write_v30_fixture(make_episodes(n_eps=1, T=40, seed=1), root, "acme/a")
    out = tmp_path / "deep.jsonl"
    cfg = CensusConfig(out_dir=tmp_path, files_per_dataset=1)
    run_deep_tier(["acme/a"], LocalSource(root), cfg, out)
    rows = [json.loads(line) for line in out.read_text().strip().splitlines()]
    assert rows[0]["revision"] == ""
    assert load_done(out) == {"acme/a@"}


def test_run_census_metadata_tier_only_leaves_deep_jsonl_absent(tmp_path):
    # cfg.tier is the only tier switch run_census honours. "metadata"
    # must run the metadata tier and must not touch the deep tier at
    # all, proven by deep.jsonl never being created.
    root = tmp_path / "repos"
    write_v30_fixture(make_episodes(n_eps=2, T=40, seed=1), root, "acme/a")
    cfg = CensusConfig(out_dir=tmp_path, tier="metadata", files_per_dataset=1)
    n = run_census(["acme/a"], LocalSource(root), cfg, out_dir=tmp_path)
    assert n == 1
    meta_path = tmp_path / "metadata.jsonl"
    deep_path = tmp_path / "deep.jsonl"
    assert meta_path.exists()
    rows = [json.loads(line) for line in meta_path.read_text().strip().splitlines()]
    assert [r["repo"] for r in rows] == ["acme/a"]
    assert not deep_path.exists()


def test_run_census_deep_tier_only_leaves_metadata_jsonl_absent(tmp_path):
    # Symmetric case: cfg.tier="deep" must run only the deep tier, so
    # metadata.jsonl is never created.
    root = tmp_path / "repos"
    write_v30_fixture(make_episodes(n_eps=2, T=40, seed=1), root, "acme/a")
    cfg = CensusConfig(out_dir=tmp_path, tier="deep", files_per_dataset=1)
    n = run_census(["acme/a"], LocalSource(root), cfg, out_dir=tmp_path)
    assert n == 1
    meta_path = tmp_path / "metadata.jsonl"
    deep_path = tmp_path / "deep.jsonl"
    assert deep_path.exists()
    rows = [json.loads(line) for line in deep_path.read_text().strip().splitlines()]
    assert [r["repo"] for r in rows] == ["acme/a"]
    assert not meta_path.exists()


def test_run_census_both_runs_both_tiers(tmp_path):
    root = tmp_path / "repos"
    write_v30_fixture(make_episodes(n_eps=2, T=40, seed=1), root, "acme/a")
    cfg = CensusConfig(out_dir=tmp_path, tier="both", files_per_dataset=1)
    n = run_census(["acme/a"], LocalSource(root), cfg, out_dir=tmp_path)
    assert n == 2
    meta_path = tmp_path / "metadata.jsonl"
    deep_path = tmp_path / "deep.jsonl"
    assert meta_path.exists()
    assert deep_path.exists()
    assert len(meta_path.read_text().strip().splitlines()) == 1
    assert len(deep_path.read_text().strip().splitlines()) == 1


def test_run_census_rejects_an_invalid_tier(tmp_path):
    # A programmatic caller who mistypes cfg.tier must get a clear
    # error naming the bad value, not a silent no-op that runs nothing.
    root = tmp_path / "repos"
    write_v30_fixture(make_episodes(n_eps=1, T=40, seed=1), root, "acme/a")
    cfg = CensusConfig(out_dir=tmp_path, tier="deep-only-ish")
    with pytest.raises(ValueError, match="deep-only-ish"):
        run_census(["acme/a"], LocalSource(root), cfg, out_dir=tmp_path)


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
    reps = [
        DatasetReport(repo=f"o/d{i}", flags="stuck_state" if i < 20 else "") for i in range(100)
    ]
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
