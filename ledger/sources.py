"""Where audit frames come from.

Four implementations behind one protocol, so the checks never learn
whether they are reading the Hub, a fixture, or a generator. Measured on
2026-09-05: the five audit columns are 2.7 percent of compressed bytes,
so projecting is worth roughly 37x in transfer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import pandas as pd
import pyarrow.parquet as pq

from .hubclient import HubClient
from .paths import derive_paths
from . import synth

AUDIT_COLUMNS = ("timestamp", "frame_index", "episode_index",
                 "action", "observation.state")


def read_projected(handle) -> pd.DataFrame:
    pf = pq.ParquetFile(handle)
    want = [c for c in AUDIT_COLUMNS if c in pf.schema_arrow.names]
    return pf.read(columns=want).to_pandas()


class SampleSource(Protocol):
    def info(self, repo: str) -> dict | None: ...
    def parquet_paths(self, repo: str, info: dict, limit: int | None) -> list[str]: ...
    def frames(self, repo: str, path: str) -> pd.DataFrame: ...


class LocalSource:
    """Fixtures on disk. The backbone of the offline test suite."""

    def __init__(self, root):
        self.root = Path(root)

    def info(self, repo: str) -> dict | None:
        p = self.root / repo / "meta/info.json"
        return json.loads(p.read_text()) if p.exists() else None

    def parquet_paths(self, repo: str, info: dict, limit: int | None = None) -> list[str]:
        found = sorted(str(p.relative_to(self.root / repo).as_posix())
                       for p in (self.root / repo).glob("data/**/*.parquet"))
        return found[:limit] if limit else found

    def frames(self, repo: str, path: str) -> pd.DataFrame:
        return read_projected(self.root / repo / path)


class SyntheticSource:
    """Generated frames. No disk, no network."""

    def __init__(self, defect: str = "", fps: float = 30.0, **kw):
        self.defect, self.fps, self.kw = defect, fps, kw

    def info(self, repo: str) -> dict:
        return {"codebase_version": "demo", "fps": self.fps, "chunks_size": 1000,
                "total_episodes": self.kw.get("n_eps", 6),
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"}

    def parquet_paths(self, repo: str, info: dict, limit: int | None = None) -> list[str]:
        return ["synthetic://0"]

    def frames(self, repo: str, path: str) -> pd.DataFrame:
        return synth.make_episodes(fps=self.fps, defect=self.defect, **self.kw)


class StreamingSource:
    """HTTP range reads with column projection. The census default."""

    def __init__(self, client: HubClient | None = None):
        self.client = client or HubClient()
        from huggingface_hub import HfFileSystem
        self._fs = HfFileSystem()

    def info(self, repo: str) -> dict | None:
        return self.client.get_info(repo)

    def revision(self, repo: str) -> str:
        """The Hub revision sha captured by the most recent info(repo).

        Proxies HubClient.get_revision, so it is "" until info(repo) has
        been called at least once, and stays "" if the response carried
        no X-Repo-Commit header. Never raises.
        """
        return self.client.get_revision(repo)

    def parquet_paths(self, repo: str, info: dict, limit: int | None = None) -> list[str]:
        return derive_paths(info, limit)

    def frames(self, repo: str, path: str) -> pd.DataFrame:
        with self._fs.open(f"datasets/{repo}/{path}", "rb") as fh:
            return read_projected(fh)


class DownloadSource:
    """Whole file download then delete. Fallback when a range read fails."""

    def __init__(self, client: HubClient | None = None, keep: bool = False):
        self.client = client or HubClient()
        self.keep = keep

    def info(self, repo: str) -> dict | None:
        return self.client.get_info(repo)

    def revision(self, repo: str) -> str:
        """See StreamingSource.revision: proxies HubClient.get_revision."""
        return self.client.get_revision(repo)

    def parquet_paths(self, repo: str, info: dict, limit: int | None = None) -> list[str]:
        return derive_paths(info, limit)

    def frames(self, repo: str, path: str) -> pd.DataFrame:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(repo, path, repo_type="dataset")
        try:
            return read_projected(local)
        finally:
            if not self.keep:
                try:
                    Path(local).unlink()
                except OSError:
                    pass
