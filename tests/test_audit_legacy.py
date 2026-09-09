"""The quick audit workflow: --top and --repos.

This path predates the census and had no test at all. It carries the same
three problems the census path was fixed for, and one that arrived with
--files all. Every test here is offline: the network boundary is either
HubClient._open or the HfApi tree listing, and both are monkeypatched.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from ledger import audit
from ledger.hubclient import HubClient, RateLimited


class _Resp:
    """Enough of an http.client.HTTPResponse for HubClient._raw."""

    def __init__(self, body, headers=None):
        self._b = json.dumps(body).encode()
        self.headers = headers or {}
        self.status = 200

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _client_returning(resp_or_exc, min_interval=0.0, max_retries=2):
    """A HubClient whose single network call is replaced."""
    c = HubClient(min_interval=min_interval, max_retries=max_retries, token=None)

    def opener(req, timeout=None):
        if isinstance(resp_or_exc, Exception):
            raise resp_or_exc
        return resp_or_exc

    c._open = opener
    return c


def _http_error(code):
    return urllib.error.HTTPError("https://example/x", code, "boom", {"Retry-After": "0"}, None)


INFO = {
    "codebase_version": "v3.0",
    "fps": 30,
    "chunks_size": 1000,
    "total_episodes": 2,
    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
}


# --------------------------------------------------------------------------- #
# load_info
# --------------------------------------------------------------------------- #
def test_load_info_returns_the_parsed_info():
    got = audit.load_info("acme/ok", client=_client_returning(_Resp(INFO)))
    assert got["codebase_version"] == "v3.0"
    assert got["fps"] == 30


def test_load_info_returns_none_for_a_genuine_404():
    assert audit.load_info("acme/missing", client=_client_returning(_http_error(404))) is None


def test_load_info_returns_none_for_a_gated_repo():
    assert audit.load_info("acme/gated", client=_client_returning(_http_error(403))) is None


def test_load_info_does_not_disguise_a_rate_limit_as_a_missing_dataset():
    """A 429 that outlives the backoff is not absence.

    v0 wrapped everything in except Exception and returned None, so a
    throttled dataset was reported with the note "no meta/info.json
    (gated, missing, or not LeRobot)". That is factually wrong and a human
    reading the report cannot tell it apart from a genuinely absent one.
    """
    client = _client_returning(_http_error(429))
    with pytest.raises(RateLimited):
        audit.load_info("acme/throttled", client=client)


# --------------------------------------------------------------------------- #
# sample_parquet_paths
# --------------------------------------------------------------------------- #
class _Entry:
    def __init__(self, path, size=1000):
        self.path = path
        self.size = size


class _FakeApi:
    """Stands in for huggingface_hub.HfApi.list_repo_tree."""

    def __init__(self, tree=None, raises=None):
        self._tree = tree or {}
        self._raises = raises

    def list_repo_tree(self, repo, repo_type=None, path_in_repo=None):
        if self._raises is not None:
            raise self._raises
        return self._tree.get(path_in_repo, [])


def _patch_api(monkeypatch, api):
    monkeypatch.setattr(audit, "_hf_api", lambda: api)


def test_sample_parquet_paths_sorts_and_limits(monkeypatch):
    _patch_api(
        monkeypatch,
        _FakeApi(
            {
                "data": [_Entry("data/chunk-000")],
                "data/chunk-000": [
                    _Entry("data/chunk-000/file-002.parquet"),
                    _Entry("data/chunk-000/file-000.parquet"),
                    _Entry("data/chunk-000/file-001.parquet"),
                ],
            }
        ),
    )
    got = audit.sample_parquet_paths("acme/x", 2, 150.0)
    assert got == ["data/chunk-000/file-000.parquet", "data/chunk-000/file-001.parquet"]


def test_sample_parquet_paths_honours_max_mb(monkeypatch):
    _patch_api(
        monkeypatch,
        _FakeApi(
            {
                "data": [_Entry("data/chunk-000")],
                "data/chunk-000": [_Entry("data/chunk-000/file-000.parquet", size=500_000_000)],
            }
        ),
    )
    assert audit.sample_parquet_paths("acme/big", 1, 150.0) == []


def test_sample_parquet_paths_empty_listing_is_empty(monkeypatch):
    _patch_api(monkeypatch, _FakeApi({"data": []}))
    assert audit.sample_parquet_paths("acme/empty", 1, 150.0) == []


def test_sample_parquet_paths_raises_rather_than_faking_an_empty_dataset(monkeypatch):
    """A failed tree listing is not an empty dataset.

    The tree endpoint is the call measured getting throttled hardest.
    v0 caught every exception and returned [], so audit_repo then produced
    a report that looked like a dataset which had been audited and found to
    contain nothing.
    """
    _patch_api(monkeypatch, _FakeApi(raises=RuntimeError("429 maximum queue size reached")))
    with pytest.raises(audit.TreeListingError):
        audit.sample_parquet_paths("acme/throttled", 1, 150.0)


# --------------------------------------------------------------------------- #
# audit_repo
# --------------------------------------------------------------------------- #
def test_audit_repo_notes_a_genuinely_missing_dataset(monkeypatch):
    monkeypatch.setattr(audit, "load_info", lambda repo, client=None: None)
    rep = audit.audit_repo("acme/missing", 1, 150.0)
    assert "no meta/info.json" in rep.note
    assert rep.error == ""


def test_audit_repo_records_a_rate_limit_distinguishably(monkeypatch):
    """The whole point: throttled must not read as gated or missing."""

    def boom(repo, client=None):
        raise RateLimited("gave up on acme/throttled after 5 attempts")

    monkeypatch.setattr(audit, "load_info", boom)
    rep = audit.audit_repo("acme/throttled", 1, 150.0)
    assert "RateLimited" in rep.error
    assert "no meta/info.json" not in rep.note


def test_audit_repo_records_a_failed_listing_rather_than_an_empty_audit(monkeypatch):
    monkeypatch.setattr(audit, "load_info", lambda repo, client=None: dict(INFO))

    def boom(repo, max_files, max_mb, cap=True):
        raise audit.TreeListingError("cannot list tree for acme/x")

    monkeypatch.setattr(audit, "sample_parquet_paths", boom)
    rep = audit.audit_repo("acme/x", 1, 150.0)
    assert "TreeListingError" in rep.error or "cannot list tree" in rep.error


def test_audit_repo_audits_a_healthy_dataset_from_fakes(monkeypatch, tmp_path):
    from ledger.synth import make_episodes

    df = make_episodes(n_eps=2, T=60, defect="identity", seed=0)
    p = tmp_path / "f.parquet"
    df.to_parquet(p, index=False)

    monkeypatch.setattr(audit, "load_info", lambda repo, client=None: dict(INFO))
    monkeypatch.setattr(
        audit, "sample_parquet_paths", lambda r, mf, mb, cap=True: ["data/f.parquet"]
    )
    monkeypatch.setattr(audit, "download", lambda repo, path: str(p))

    rep = audit.audit_repo("acme/x", 1, 150.0)
    assert rep.error == ""
    assert "action_equals_state" in rep.flags


def test_audit_repo_deletes_each_download_after_reading_it(monkeypatch, tmp_path):
    """v0 left every downloaded parquet in the cache forever."""
    from ledger.synth import make_episodes

    df = make_episodes(n_eps=1, T=40, seed=0)
    p = tmp_path / "throwaway.parquet"
    df.to_parquet(p, index=False)
    assert p.exists()

    monkeypatch.setattr(audit, "load_info", lambda repo, client=None: dict(INFO))
    monkeypatch.setattr(
        audit, "sample_parquet_paths", lambda r, mf, mb, cap=True: ["data/f.parquet"]
    )
    monkeypatch.setattr(audit, "download", lambda repo, path: str(p))

    audit.audit_repo("acme/x", 1, 150.0)
    assert not p.exists(), "the quick audit must not leave downloaded parquet on disk"


# --------------------------------------------------------------------------- #
# the --files all bound
# --------------------------------------------------------------------------- #
def test_quick_audit_caps_an_unbounded_file_request(monkeypatch):
    """--files all parses to None, and small[:None] is every file.

    On a v2.0 per episode dataset that is one parquet per episode, and one
    real dataset on the Hub has 209,880 of them. The quick path downloads
    each one, so an uncapped run fills the disk.
    """
    entries = [_Entry(f"data/chunk-000/episode_{i:06d}.parquet") for i in range(200)]
    _patch_api(
        monkeypatch, _FakeApi({"data": [_Entry("data/chunk-000")], "data/chunk-000": entries})
    )

    got = audit.sample_parquet_paths("acme/huge", None, 150.0)
    assert len(got) == audit.QUICK_AUDIT_FILE_CAP
    assert audit.QUICK_AUDIT_FILE_CAP < 200


def test_the_cap_can_be_lifted_explicitly(monkeypatch):
    entries = [_Entry(f"data/chunk-000/episode_{i:06d}.parquet") for i in range(200)]
    _patch_api(
        monkeypatch, _FakeApi({"data": [_Entry("data/chunk-000")], "data/chunk-000": entries})
    )

    got = audit.sample_parquet_paths("acme/huge", None, 150.0, cap=False)
    assert len(got) == 200


# --------------------------------------------------------------------------- #
# list_top
# --------------------------------------------------------------------------- #
def test_list_top_parses_ids():
    client = _client_returning(_Resp([{"id": "a/b"}, {"id": "c/d"}]))
    assert audit.list_top(2, client=client) == ["a/b", "c/d"]


def test_list_top_survives_a_429_then_succeeds():
    """v0 called urllib directly with no retry, so one 429 ended the run."""
    seq = [_http_error(429), _Resp([{"id": "a/b"}])]
    c = HubClient(min_interval=0.0, max_retries=3, token=None)

    def opener(req, timeout=None):
        item = seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    c._open = opener
    assert audit.list_top(1, client=c) == ["a/b"]
    assert seq == []
