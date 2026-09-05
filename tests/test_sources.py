import pytest
from ledger.sources import (
    LocalSource, SyntheticSource, StreamingSource, DownloadSource,
    AUDIT_COLUMNS,
)
from ledger.synth import make_episodes, write_v20_fixture, write_v30_fixture


@pytest.fixture
def v20(tmp_path):
    write_v20_fixture(make_episodes(n_eps=3, T=40, seed=1), tmp_path, "acme/v20")
    return tmp_path


@pytest.fixture
def v30(tmp_path):
    write_v30_fixture(make_episodes(n_eps=3, T=40, seed=1), tmp_path, "acme/v30")
    return tmp_path


def test_local_reads_v20(v20):
    s = LocalSource(v20)
    info = s.info("acme/v20")
    assert info["codebase_version"] == "v2.0"
    paths = s.parquet_paths("acme/v20", info, limit=None)
    assert len(paths) == 3
    df = s.frames("acme/v20", paths[0])
    assert set(df.columns) <= set(AUDIT_COLUMNS)
    assert len(df) == 40


def test_local_reads_v30(v30):
    s = LocalSource(v30)
    info = s.info("acme/v30")
    df = s.frames("acme/v30", s.parquet_paths("acme/v30", info, 1)[0])
    assert len(df) == 120  # 3 episodes packed


def test_local_missing_repo_returns_none(tmp_path):
    assert LocalSource(tmp_path).info("acme/nope") is None


def test_synthetic_source_is_offline():
    s = SyntheticSource(defect="stuck")
    info = s.info("synthetic/stuck")
    df = s.frames("synthetic/stuck", s.parquet_paths("synthetic/stuck", info, 1)[0])
    assert len(df) > 0


def test_projection_drops_non_audit_columns(tmp_path):
    import pandas as pd
    df = make_episodes(n_eps=1, T=20, seed=2)
    df["observation.images.top"] = [b"x" * 1024] * len(df)
    p = tmp_path / "acme/v30/data/chunk-000/file-000.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    (tmp_path / "acme/v30/meta").mkdir(parents=True, exist_ok=True)
    (tmp_path / "acme/v30/meta/info.json").write_text(
        '{"codebase_version":"v3.0","fps":30,"chunks_size":1000,"total_episodes":1,'
        '"data_path":"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"}')
    out = LocalSource(tmp_path).frames("acme/v30", "data/chunk-000/file-000.parquet")
    assert "observation.images.top" not in out.columns


def test_all_four_sources_have_the_protocol_shape(tmp_path):
    # SampleSource is a plain, non runtime checkable Protocol, so this
    # checks the shape by hand instead of with isinstance: every
    # implementation must offer callable info, parquet_paths and frames
    # attributes, regardless of whether it talks to disk, memory or the
    # network.
    instances = [
        LocalSource(tmp_path),
        SyntheticSource(),
        StreamingSource(),
        DownloadSource(),
    ]
    for src in instances:
        assert callable(getattr(src, "info", None))
        assert callable(getattr(src, "parquet_paths", None))
        assert callable(getattr(src, "frames", None))


def test_streaming_and_download_construct_with_no_token_present(monkeypatch):
    # StreamingSource and DownloadSource both hit the live Hub once you
    # call info, parquet_paths or frames on them, and anonymous requests
    # get HTTP 429 quickly (see ledger/hubclient.py). Neither class
    # should need a token just to build, so this forces "no token
    # anywhere" and checks that construction alone stays offline and
    # does not raise.
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
    streaming = StreamingSource()
    download = DownloadSource()
    assert streaming.client.authenticated is False
    assert download.client.authenticated is False


@pytest.mark.network
def test_streaming_source_reads_a_real_small_dataset():
    # The one live test in this dispatch. lerobot/pusht is a small, well
    # known LeRobot dataset. Skipped by default, tests/conftest.py skips
    # anything marked "network" unless run with -m network. Run it with
    # a token exported as HF_TOKEN to avoid the anonymous rate limit.
    s = StreamingSource()
    info = s.info("lerobot/pusht")
    assert info is not None
    paths = s.parquet_paths("lerobot/pusht", info, limit=1)
    assert paths
    df = s.frames("lerobot/pusht", paths[0])
    assert set(df.columns) <= set(AUDIT_COLUMNS)
    assert len(df) > 0
