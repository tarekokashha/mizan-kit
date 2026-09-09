import pytest

from ledger.paths import (
    derive_paths,
    estimate_file_count,
    layout_family,
    packed_sample_is_partial,
)
from ledger.sources import LocalSource

V20 = {
    "codebase_version": "v2.0",
    "chunks_size": 1000,
    "total_episodes": 2500,
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
}
V30 = {
    "codebase_version": "v3.0",
    "chunks_size": 1000,
    "total_episodes": 50,
    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
}


def test_family_detection():
    assert layout_family(V20) == "per_episode"
    assert layout_family(V30) == "packed"


def test_v20_paths_are_derived_arithmetically():
    p = derive_paths(V20, limit=3)
    assert p == [
        "data/chunk-000/episode_000000.parquet",
        "data/chunk-000/episode_000001.parquet",
        "data/chunk-000/episode_000002.parquet",
    ]


def test_v20_chunk_rolls_over_at_chunks_size():
    p = derive_paths(V20)
    assert len(p) == 2500
    assert p[1000] == "data/chunk-001/episode_001000.parquet"
    assert p[2499] == "data/chunk-002/episode_002499.parquet"


def test_v30_starts_at_first_file():
    p = derive_paths(V30, limit=1)
    assert p == ["data/chunk-000/file-000.parquet"]


def test_v30_limit_none_without_an_exists_check_falls_back_to_one_file():
    # IMPORTANT 3: the honest fallback when nothing can confirm there is
    # more than one file. This must stay a single file, not zero and not
    # a guess, exactly what derive_paths has always returned here.
    assert derive_paths(V30, limit=None) == ["data/chunk-000/file-000.parquet"]


def test_v30_limit_none_with_exists_discovers_every_file(tmp_path):
    # IMPORTANT 3: a packed local fixture with 3 parquet files must
    # yield all 3 under limit=None, via a real exists() check
    # (ledger.sources.LocalSource), not the silent single-file sample.
    repo = "acme/x"
    for f in range(3):
        p = tmp_path / repo / f"data/chunk-000/file-{f:03d}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    source = LocalSource(tmp_path)
    paths = derive_paths(V30, limit=None, exists=lambda p: source.exists(repo, p))
    assert paths == [
        "data/chunk-000/file-000.parquet",
        "data/chunk-000/file-001.parquet",
        "data/chunk-000/file-002.parquet",
    ]


def test_v30_limit_two_with_exists_still_yields_exactly_two(tmp_path):
    # A bounded limit is unaffected by exists() being available: it
    # still trusts the requested count rather than probing.
    repo = "acme/x"
    for f in range(3):
        p = tmp_path / repo / f"data/chunk-000/file-{f:03d}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    source = LocalSource(tmp_path)
    paths = derive_paths(V30, limit=2, exists=lambda p: source.exists(repo, p))
    assert paths == ["data/chunk-000/file-000.parquet", "data/chunk-000/file-001.parquet"]


def test_v30_limit_none_with_exists_stops_at_the_first_gap(tmp_path):
    # A gap (file 1 missing but file 2 present, which should not happen
    # on a real Hub dataset but must not hang or skip past it either)
    # stops discovery at the gap rather than continuing past it.
    repo = "acme/x"
    for f in (0, 2):
        p = tmp_path / repo / f"data/chunk-000/file-{f:03d}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    source = LocalSource(tmp_path)
    paths = derive_paths(V30, limit=None, exists=lambda p: source.exists(repo, p))
    assert paths == ["data/chunk-000/file-000.parquet"]


def test_packed_sample_is_partial_only_when_packed_unbounded_and_blind():
    assert packed_sample_is_partial(V30, limit=None, has_exists=False) is True
    assert packed_sample_is_partial(V30, limit=None, has_exists=True) is False
    assert packed_sample_is_partial(V30, limit=1, has_exists=False) is False
    assert packed_sample_is_partial(V20, limit=None, has_exists=False) is False


def test_estimate_counts():
    assert estimate_file_count(V20) == 2500
    assert estimate_file_count(V30) is None


def test_unknown_template_raises():
    with pytest.raises(ValueError, match="data_path"):
        derive_paths({"codebase_version": "v9", "data_path": "weird/{nope}.parquet"})
