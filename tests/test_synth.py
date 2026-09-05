import json

import numpy as np
import pandas as pd
import pytest
from ledger.synth import make_episodes, DEFECTS, write_v20_fixture, write_v30_fixture


def test_clean_shape_and_dtypes():
    df = make_episodes(fps=30.0, n_eps=4, T=100, n_joints=6, seed=1)
    assert len(df) == 400
    assert set(df.columns) == {"timestamp", "frame_index", "episode_index",
                               "action", "observation.state"}
    assert df["episode_index"].nunique() == 4
    assert len(df["action"].iloc[0]) == 6


def test_is_deterministic_under_seed():
    a = make_episodes(seed=7)
    b = make_episodes(seed=7)
    assert np.array_equal(np.stack(a["action"].to_numpy()),
                          np.stack(b["action"].to_numpy()))


def test_different_seeds_differ():
    a = make_episodes(seed=1)
    b = make_episodes(seed=2)
    assert not np.array_equal(np.stack(a["action"].to_numpy()),
                              np.stack(b["action"].to_numpy()))


@pytest.mark.parametrize("defect", DEFECTS)
def test_every_defect_generates(defect):
    df = make_episodes(defect=defect, seed=0)
    assert len(df) > 0


def test_duplicate_defect_reuses_episode_zero_action():
    df = make_episodes(defect="duplicate", seed=0, n_eps=3, T=50)
    by_episode = {ep: np.stack(sub["action"].to_numpy())
                  for ep, sub in df.groupby("episode_index")}
    for ep in range(1, 3):
        assert np.array_equal(by_episode[ep], by_episode[0])


def test_swapped_defect_exchanges_action_and_state():
    clean = make_episodes(defect="", seed=0, n_eps=2, T=50)
    swapped = make_episodes(defect="swapped", seed=0, n_eps=2, T=50)
    clean_action = np.stack(clean["action"].to_numpy())
    clean_state = np.stack(clean["observation.state"].to_numpy())
    swapped_action = np.stack(swapped["action"].to_numpy())
    swapped_state = np.stack(swapped["observation.state"].to_numpy())
    assert np.array_equal(swapped_action, clean_state)
    assert np.array_equal(swapped_state, clean_action)


def test_identity_defect_makes_action_equal_state():
    df = make_episodes(defect="identity", seed=0, n_eps=2, T=50)
    action = np.stack(df["action"].to_numpy())
    state = np.stack(df["observation.state"].to_numpy())
    assert np.array_equal(action, state)


def test_stuck_defect_has_run_of_identical_state():
    df = make_episodes(defect="stuck", seed=0, n_eps=1, T=100)
    state = np.stack(df["observation.state"].to_numpy())
    run = state[50:100]
    assert np.array_equal(run, np.broadcast_to(state[50], run.shape))


def test_drops_defect_removes_frames_from_even_episodes():
    T = 80
    df = make_episodes(defect="drops", seed=0, n_eps=2, T=T)
    assert int((df["episode_index"] == 0).sum()) < T
    assert int((df["episode_index"] == 1).sum()) == T


def test_n_joints_is_honoured():
    df = make_episodes(n_joints=14, T=50, n_eps=2, seed=3)
    assert len(df["observation.state"].iloc[0]) == 14


def test_v20_fixture_round_trips(tmp_path):
    df = make_episodes(n_eps=3, T=40, seed=5)
    info = write_v20_fixture(df, tmp_path, "acme/v20demo")
    assert info["codebase_version"] == "v2.0"
    assert "episode_{episode_index:06d}" in info["data_path"]
    root = tmp_path / "acme/v20demo"
    assert json.loads((root / "meta/info.json").read_text())["fps"] == 30
    parquet_files = sorted(root.glob("data/**/*.parquet"))
    assert len(parquet_files) == 3  # one per episode

    # Do not assume the round trip preserves array shape: read it back and
    # check. action and observation.state are object columns of numpy
    # arrays going in; confirm np.stack still recovers a clean (T, n_joints)
    # float32 array coming out, not a column of plain Python lists or a
    # ragged/ambiguous shape.
    back = pd.read_parquet(parquet_files[0])
    action = np.stack(back["action"].to_numpy())
    assert action.shape == (40, 6)
    assert action.dtype == np.float32
    state = np.stack(back["observation.state"].to_numpy())
    assert state.shape == (40, 6)
    assert state.dtype == np.float32


def test_v30_fixture_round_trips(tmp_path):
    df = make_episodes(n_eps=3, T=40, seed=5)
    info = write_v30_fixture(df, tmp_path, "acme/v30demo")
    assert info["codebase_version"] == "v3.0"
    assert "file-{file_index:03d}" in info["data_path"]
    root = tmp_path / "acme/v30demo"
    parquet_files = list(root.glob("data/**/*.parquet"))
    assert len(parquet_files) == 1  # episodes packed

    back = pd.read_parquet(parquet_files[0])
    assert len(back) == 120  # 3 episodes times 40 frames, packed into one file
    action = np.stack(back["action"].to_numpy())
    assert action.shape == (120, 6)
    assert action.dtype == np.float32
