import pytest
from ledger.paths import derive_paths, layout_family, estimate_file_count

V20 = {"codebase_version": "v2.0", "chunks_size": 1000, "total_episodes": 2500,
       "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"}
V30 = {"codebase_version": "v3.0", "chunks_size": 1000, "total_episodes": 50,
       "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"}


def test_family_detection():
    assert layout_family(V20) == "per_episode"
    assert layout_family(V30) == "packed"


def test_v20_paths_are_derived_arithmetically():
    p = derive_paths(V20, limit=3)
    assert p == ["data/chunk-000/episode_000000.parquet",
                 "data/chunk-000/episode_000001.parquet",
                 "data/chunk-000/episode_000002.parquet"]


def test_v20_chunk_rolls_over_at_chunks_size():
    p = derive_paths(V20)
    assert len(p) == 2500
    assert p[1000] == "data/chunk-001/episode_001000.parquet"
    assert p[2499] == "data/chunk-002/episode_002499.parquet"


def test_v30_starts_at_first_file():
    p = derive_paths(V30, limit=1)
    assert p == ["data/chunk-000/file-000.parquet"]


def test_estimate_counts():
    assert estimate_file_count(V20) == 2500
    assert estimate_file_count(V30) is None


def test_unknown_template_raises():
    with pytest.raises(ValueError, match="data_path"):
        derive_paths({"codebase_version": "v9", "data_path": "weird/{nope}.parquet"})
