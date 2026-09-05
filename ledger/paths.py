"""Derive parquet paths from the info.json data_path template.

Measured on 2026-09-05: every LeRobot dataset carries data_path, so no
tree API call is needed. That matters because the tree endpoint is the
one the anonymous rate limit punishes hardest.

Two families exist on the Hub:
  per_episode  v2.x, one parquet per episode, count is total_episodes
  packed       v3.x, episodes packed into large files, count unknown
               without meta/episodes, so paths are probed in order.
"""
from __future__ import annotations


def layout_family(info: dict) -> str:
    tmpl = info.get("data_path", "")
    if "episode_index" in tmpl and "episode_chunk" in tmpl:
        return "per_episode"
    if "file_index" in tmpl and "chunk_index" in tmpl:
        return "packed"
    raise ValueError(f"unrecognised data_path template: {tmpl!r}")


def estimate_file_count(info: dict) -> int | None:
    if layout_family(info) == "per_episode":
        return int(info.get("total_episodes", 0))
    return None


def derive_paths(info: dict, limit: int | None = None) -> list[str]:
    tmpl = info["data_path"]
    family = layout_family(info)
    chunk = int(info.get("chunks_size", 1000)) or 1000
    if family == "per_episode":
        n = int(info.get("total_episodes", 0))
        if limit is not None:
            n = min(n, limit)
        return [tmpl.format(episode_chunk=e // chunk, episode_index=e)
                for e in range(n)]
    n = limit if limit is not None else 1
    return [tmpl.format(chunk_index=0, file_index=f) for f in range(n)]
