# Metadata census, 400 datasets, 2026-09-09

First live run of the M-02 LEDGER census. Tier 1 only: this reads each
dataset's `meta/info.json` and records what it declares. It does not open any
parquet, so nothing here is a temporal integrity finding. The deep tier has
not been run.


## Correction, 2026-09-11

These results stand as a valid random sample of the LeRobot population as
it stood on 2026-09-09. One property claimed for them does not hold.

`draw_sample` drew indices into the sorted frame, so which datasets were
selected depended on the frame's length, and the Hub population grows
daily. The seed and size therefore do not identify this sample: re running
`draw_sample(frame, 400, 20260905)` on a later day returns almost entirely
different datasets. Measured overlap between this sample and a draw two
days later was 3 of 400.

The sampler has been fixed to rank each repo by a hash of its own id and
the seed, which is stable under population growth. But the frame that
produced the sample below was never recorded, so this particular sample
cannot be reconstructed after the fact. Its members are exactly the 400
repositories listed in the accompanying `.jsonl`, and that file is now the
only record of who was in it.

## Method

- Frame: every dataset the Hub returns for the LeRobot filter, paginated.
- Draw: `draw_sample(frame, 400, seed=20260905)`, which sorts the frame
  before drawing so the sample does not depend on Hub ordering.
- Run: `python -m ledger.audit --census metadata --sample-size 400 --seed 20260905`
- Anonymous. No token. The client's backoff and minimum request interval
  carried the run.
- 400 records, 400 unique repositories, 0 duplicated.

Every proportion below carries a Wilson 95 percent interval from
`cairo_protocol.stats.wilson_interval`. None is reported bare.

## Reachability

390 of 400 returned an `info.json`. 10 did not, rate 0.025 [0.014, 0.045],
all recorded as `no info.json`, which covers gated, private, deleted, and
not actually a LeRobot dataset. This tier cannot tell those apart, and does
not claim to.

## Declared codebase version

| value | n | rate | 95 percent interval |
|---|---|---|---|
| v3.0 | 280 | 0.718 | [0.671, 0.760] |
| v2.1 | 100 | 0.256 | [0.216, 0.302] |
| v2.0 | 9 | 0.023 | [0.012, 0.043] |
| v2.1-embeddings-sharded | 1 | 0.003 | [0.000, 0.014] |

Four distinct strings, not the two families the design assumed. The last is
a publisher's own variant, not a LeRobot release.

## Declared layout

| value | n | rate | 95 percent interval |
|---|---|---|---|
| packed | 277 | 0.710 | [0.663, 0.753] |
| per_episode | 109 | 0.279 | [0.237, 0.326] |
| unknown | 4 | 0.010 | [0.004, 0.026] |

## Declared fps

30 fps dominates at 0.756 [0.711, 0.796]. The tail runs 20, 10, 15, 50, 25,
5, 60, 14 and 2 fps. A 2 fps entry exists.

## Size

Median 10 episodes, p90 130, max 2000. Median 6,175 frames, p90 77,268, max
660,229. The population is dominated by small datasets.

## Findings

### 1. About one dataset in twelve declares zero episodes

32 of 390 reachable datasets report `total_episodes: 0`, rate 0.082
[0.059, 0.114].

This is a declaration, not a measurement: the metadata says the dataset is
empty. It is not yet established whether the parquet is also empty, whether
the field is simply unset, or whether these are abandoned uploads. That
requires the deep tier. Nothing here names any dataset as defective.

### 2. Four datasets declare a layout no LeRobot parquet reader can follow

Rate 0.010 [0.004, 0.026]. Two distinct causes:

**Not parquet at all.** Three datasets from one publisher, THULab, declare
`codebase_version: v3.0` with `data_path` values like
`data/jaco_play.tsfile`. A `.tsfile` is not the parquet layout the declared
version implies. The audit records these as `unknown` and continues rather
than crashing.

**A third parquet template.** `saaduddinM/OXE_berkeley_autolab_ur5_embeddings`
declares `codebase_version: v2.1-embeddings-sharded` and
`data_path: data/shard-{shard_id:05d}-of-{num_shards:05d}.parquet`. This one
is ordinary parquet under a sharded naming scheme that `ledger.paths`
does not yet know. It is recoverable and worth supporting.

### 3. The declared version does not determine the layout

`v3.0` appears with both the packed template and the `.tsfile` layout. Any
tool that branches on `codebase_version` alone, rather than reading
`data_path`, will be wrong for some real datasets on the Hub.

## What this does not establish

- No temporal integrity claim. No parquet was opened.
- No dataset here is defective. These are declarations the publishers made,
  recorded as measurements. Per `CLAUDE.md`, a finding requires a human to
  open the data and confirm it.
- 400 of at least 12,000 datasets. The intervals above describe that sample.
- Run anonymously, so throttling shaped the pace but not the sample: the
  draw was fixed before the run began.

## Next

1. Deep tier on this same seeded sample, which is what produces the temporal
   defect prevalence the programme is actually after.
2. Teach `ledger.paths` the sharded template.
3. Open several of the 32 zero-episode datasets by hand and find out which
   of the three explanations is right.
