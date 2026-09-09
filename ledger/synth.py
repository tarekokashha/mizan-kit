"""Seeded synthetic LeRobot style episodes, with one injector per defect.

Used by the demo, the unit tests and the property based calibration.
Every function is pure given its seed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

DEFECTS = ("", "identity", "drops", "stuck", "swapped", "duplicate")


def make_episodes(
    fps: float = 30.0,
    n_eps: int = 6,
    T: int = 300,
    n_joints: int = 6,
    lag: int = 2,
    defect: str = "",
    seed: int = 0,
    noise: float = 1e-3,
) -> pd.DataFrame:
    if defect not in DEFECTS:
        raise ValueError(f"unknown defect {defect!r}, expected one of {DEFECTS}")
    rng = np.random.default_rng(seed)
    rows = []
    first_action = None
    for e in range(n_eps):
        t = np.arange(T) / fps
        base = np.cumsum(rng.normal(0, 0.02, size=(T + lag + 5, n_joints)), axis=0)
        action = base[lag : T + lag]
        state = base[:T] + rng.normal(0, noise, size=(T, n_joints))
        if defect == "identity":
            action = state.copy()
        if defect == "drops" and e % 2 == 0:
            keep = np.ones(T, bool)
            keep[rng.choice(T, size=max(1, T // 8), replace=False)] = False
            t, action, state = t[keep], action[keep], state[keep]
        if defect == "stuck" and e % 2 == 0:
            hi = min(200, len(state))
            state[50:hi] = state[50]
        if defect == "swapped":
            action, state = state, action
        if defect == "duplicate" and first_action is not None:
            action = first_action
        if e == 0:
            first_action = action
        for i in range(len(t)):
            rows.append(
                {
                    "timestamp": np.float32(t[i]),
                    "frame_index": i,
                    "episode_index": e,
                    "action": action[i].astype(np.float32),
                    "observation.state": state[i].astype(np.float32),
                }
            )
    return pd.DataFrame(rows)


# Explicit schema so action and observation.state round trip as float32
# lists rather than degrading to plain Python lists of generic numeric type.
# Without this, pyarrow infers the list value type per column on its own,
# which happens to work in the versions this was built against, but leaves
# shape recovery (np.stack) resting on inference we do not control. Being
# explicit here makes that contract visible and enforced at write time.
_EPISODE_SCHEMA = pa.schema(
    [
        ("timestamp", pa.float32()),
        ("frame_index", pa.int64()),
        ("episode_index", pa.int64()),
        ("action", pa.list_(pa.float32())),
        ("observation.state", pa.list_(pa.float32())),
    ]
)


def _to_parquet(df: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(df, schema=_EPISODE_SCHEMA, preserve_index=False)
    import pyarrow.parquet as pq

    pq.write_table(table, path)


def _write_info(root: Path, info: dict) -> dict:
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta/info.json").write_text(json.dumps(info, indent=2))
    return info


def write_v20_fixture(df: pd.DataFrame, root, repo: str, fps: float = 30.0) -> dict:
    root = Path(root) / repo
    for ep, sub in df.groupby("episode_index", sort=True):
        p = root / f"data/chunk-000/episode_{int(ep):06d}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        _to_parquet(sub, p)
    return _write_info(
        root,
        {
            "codebase_version": "v2.0",
            "fps": fps,
            "chunks_size": 1000,
            "total_episodes": int(df["episode_index"].nunique()),
            "total_frames": int(len(df)),
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        },
    )


def write_v30_fixture(df: pd.DataFrame, root, repo: str, fps: float = 30.0) -> dict:
    root = Path(root) / repo
    p = root / "data/chunk-000/file-000.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    _to_parquet(df, p)
    return _write_info(
        root,
        {
            "codebase_version": "v3.0",
            "fps": fps,
            "chunks_size": 1000,
            "total_episodes": int(df["episode_index"].nunique()),
            "total_frames": int(len(df)),
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        },
    )
