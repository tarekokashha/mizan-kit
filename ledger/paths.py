"""Derive parquet paths from the info.json data_path template.

Measured on 2026-09-05: every LeRobot dataset carries data_path, so no
tree API call is needed. That matters because the tree endpoint is the
one the anonymous rate limit punishes hardest.

Two families exist on the Hub:
  per_episode  v2.x, one parquet per episode, count is total_episodes
  packed       v3.x, episodes packed into large files, count unknown
               without meta/episodes, so paths are probed in order.

IMPORTANT 3: the two families are not symmetric under limit=None. See
derive_paths's own docstring for what that means and how a caller
supplies an `exists` check to close the gap for packed datasets too.
"""

from __future__ import annotations

from collections.abc import Callable


def layout_family(info: dict) -> str:
    tmpl = info.get("data_path", "")
    if "episode_index" in tmpl and "episode_chunk" in tmpl:
        return "per_episode"
    if "file_index" in tmpl and "chunk_index" in tmpl:
        return "packed"
    if "shard_id" in tmpl and "num_shards" in tmpl:
        return "sharded"
    raise ValueError(f"unrecognised data_path template: {tmpl!r}")


def estimate_file_count(info: dict) -> int | None:
    family = layout_family(info)
    if family == "per_episode":
        return int(info.get("total_episodes", 0))
    if family == "sharded":
        # num_shards is declared in info.json, so unlike packed this count
        # is exact rather than unknowable without probing.
        shards = info.get("num_shards")
        return int(shards) if shards is not None else None
    return None


def derive_paths(
    info: dict, limit: int | None = None, exists: Callable[[str], bool] | None = None
) -> list[str]:
    """Format the parquet paths a dataset's data_path template predicts.

    The two families are NOT symmetric under limit=None, and that
    asymmetry is real, not a bug to paper over. A per_episode (v2.x)
    dataset's file count is info["total_episodes"], so limit=None
    already means every file: every path is formatted directly, no I/O
    needed. A packed (v3.x) dataset's file count is not knowable from
    info.json at all (see the module docstring). With limit=None and no
    `exists` callback, this returns exactly one path: an honestly
    documented single-file fallback, not a guess and not a claim to have
    sampled everything.

    Passing `exists`, a callable that reports whether a given relative
    path is actually present, makes limit=None mean every file for a
    packed dataset too: file indices are probed in ascending order
    (chunk_index fixed at 0, same as the bounded branch below, since
    nothing here knows the real chunk rollover convention for packed
    layouts) and the list stops at the first index `exists` reports
    missing. `exists` is consulted only for the packed, limit=None case;
    a bounded limit is still satisfied by formatting paths directly,
    trusting the requested count rather than confirming each one is
    really there, exactly as before.

    See ledger.sources for exists() on the concrete SampleSource
    implementations, and ledger.paths.packed_sample_is_partial for how a
    caller with no exists callback records that the resulting sample was
    partial rather than complete.
    """
    tmpl = info["data_path"]
    family = layout_family(info)
    chunk = int(info.get("chunks_size", 1000)) or 1000
    if family == "per_episode":
        n = int(info.get("total_episodes", 0))
        if limit is not None:
            n = min(n, limit)
        return [tmpl.format(episode_chunk=e // chunk, episode_index=e) for e in range(n)]
    if family == "sharded":
        # Exact, like per_episode and unlike packed: num_shards is declared,
        # so the whole file list is derivable with no probing and nothing
        # is ever silently missing. Found in the wild by the 2026-09-09
        # census, which recorded four datasets whose layout no reader here
        # could follow; this is the one of them that was recoverable.
        shards = info.get("num_shards")
        if shards is None:
            raise ValueError(
                "sharded data_path needs num_shards in info.json to enumerate "
                f"its files, and this one does not declare it: {tmpl!r}"
            )
        n = int(shards)
        total = n
        if limit is not None:
            n = min(n, limit)
        return [tmpl.format(shard_id=i, num_shards=total) for i in range(n)]
    if limit is None and exists is not None:
        paths: list[str] = []
        f = 0
        while True:
            path = tmpl.format(chunk_index=0, file_index=f)
            if not exists(path):
                break
            paths.append(path)
            f += 1
        return paths
    n = limit if limit is not None else 1
    return [tmpl.format(chunk_index=0, file_index=f) for f in range(n)]


def packed_sample_is_partial(info: dict, limit: int | None, has_exists: bool) -> bool:
    """True when a caller asking for every file (limit=None) of a packed
    dataset is about to get derive_paths's single-file fallback because
    no existence check is available to discover the rest of them.

    Per-episode datasets never hit this: their file count already comes
    from total_episodes, so limit=None is never partial there. A caller
    (ledger.census.run_deep_tier) uses this to decide whether to note
    the resulting sample as partial on the report, so a user can tell
    "sampled 1 file because that really is everything" apart from
    "sampled 1 file because nothing could check for more". derive_paths
    itself stays a plain function returning a list of paths, not a
    (paths, partial) pair, so this lives as a small separate function
    rather than changing that contract.
    """
    if limit is not None or has_exists:
        return False
    try:
        return layout_family(info) == "packed"
    except ValueError:
        return False
