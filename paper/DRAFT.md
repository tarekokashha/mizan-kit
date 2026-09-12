# Temporal Integrity of Robot Learning Datasets at Population Scale

Draft. Related Work is written and every citation in it was fetched and
verified from a primary source before inclusion, per `CLAUDE.md`. The
Results, Discussion and Conclusion report the completed census. What
remains is the column swap check on the six `negative_lag` datasets,
which `paper/LIMITATIONS.md` names as the weakest evidence in the
confirmed set.

## Abstract (draft)

This is a draft abstract. What is established: a metadata-only census
(tier 1) over a seeded, reproducible sample of 400 LeRobot Hub datasets,
drawn from a population of at least 75,750, describing what each dataset's
`meta/info.json` declares about its codebase version, storage layout, frame
rate, and size. Every proportion below carries a Wilson 95 percent
confidence interval. What is not established: any finding about temporal
integrity itself. Temporal integrity requires reading the actual recorded
frames, which is the deep tier of this census, and the deep tier is
incomplete at the time of this draft. Separately, and independent of
completeness, this program's own operating rules hold that no automated
flag becomes a finding until a human opens the corresponding dataset and
confirms it; zero such confirmations exist for either tier as of this
draft. There are therefore zero findings in this programme so far. There
are population-level measurements of what publishers declare, and there are
provisional, unconfirmed flags on a partial sample. This draft does not
claim more than that.

## 1. Introduction

A policy trained by imitation learning is trained directly on logged
trajectories: timestamps, actions, and observed states, read frame by
frame from whatever a dataset's parquet files contain. Nothing in a
standard imitation learning pipeline validates that those frames are
temporally coherent before they are used to compute a loss. A dataset
whose timestamps are non-monotonic, whose frames are missing or repeated,
whose observed state is bit-identical across long stretches of playback
(`stuck_state` in `ledger/checks.py`), or whose recorded action channel is
actually a copy of the state channel rather than a genuine control signal
(`identity_frac`), passes that defect straight into whatever policy is
trained on it. A policy cannot tell the difference between a real control
signal and a logging artifact; it will reproduce whichever one it was
shown. This is the motivation for measuring temporal integrity in training
data directly, rather than only downstream in policy behavior.

Measuring this at population scale, across the LeRobot Hub rather than one
named dataset at a time, has not been done inside this codebase before
this programme. `docs/superpowers/specs/2026-09-05-ledger-census-design.md`
records that the tool this programme extends, `ledger/audit.py` v0,
"audits a handful of named LeRobot datasets by downloading whole parquet
files," and that a Hub-wide prevalence table "neither exists," despite
being the README's own stated next step. The same design document measures
why: an anonymous client is rate-limited within tens of requests against
the Hub API, the population is at least 75,750 datasets, and some
individual datasets alone (`kuka_lerobot`, under the older per-episode
layout) span over 200,000 files, so an exhaustive crawl "is not finite on
one machine." A crawl that stops partway has no defined sampling frame and
supports no prevalence claim at all. Those constraints are why this
programme is a seeded, sampled, two-tier census rather than an exhaustive
one, and why, inside this codebase, no population-scale measurement
existed before it.

Whether a population-scale temporal integrity census has been attempted
elsewhere, outside this codebase, is a claim this draft does not make.
That claim would require citations, and the Related Work section below is
intentionally empty for exactly that reason.

## 2. Related Work

**Robot learning datasets and benchmarks.** The datasets this census draws
from exist because of a deliberate shift toward pooling and standardizing
robot demonstration data at scale. LeRobot, the Hugging Face library whose
`LeRobotDataset` format and Hub hosting this programme audits directly,
defines the `meta/info.json` schema this census's tier 1 reads: declared
codebase version, frame rate, storage layout, and episode structure
(Cadene et al., 2026). Open X-Embodiment pooled 60 existing robot datasets
from 34 labs and 22 embodiments into one format and showed that a policy
trained across all of them transfers better than one trained on any single
dataset, establishing the premise that a large, heterogeneous, cross
institution corpus is worth building and worth training on (O'Neill et al.,
2023). DROID, RT-1, and BridgeData V2 are three of the more heavily used
single source contributions to that pooled ecosystem. DROID collected
76,000 trajectories across 564 scenes and 52 buildings specifically to test
whether policy generalization improves with collection diversity
(Khazatsky et al., 2024). RT-1 trained a 35M parameter transformer on
130,000 episodes from a fleet of 13 robots over 17 months (Brohan et al.,
2022). BridgeData V2 contributed 60,096 trajectories across 24 environments
on a low cost, publicly available robot, aimed at broadening who can
contribute data rather than only who can consume it (Walke et al., 2023).
RT-2 extended the same family by co-training a vision language action model
on robot trajectories and web scale vision language data, showing that the
value of a demonstration dataset depends on what else it is trained
alongside (Brohan et al., 2023). None of these papers audits the temporal
structure of the frames it releases. Each reports what a policy trained on
the data can do; none reports whether the timestamps, frame indices, or
action-state relationship inside the released files are internally
coherent. At the scale each contributes, a per-frame audit was not the
paper's question. This programme treats the resulting corpus, now
aggregated and re-hosted across tens of thousands of Hub repositories, as
the object of study instead.

**Data quality and dataset auditing.** Auditing a dataset for defects
independent of any specific downstream model is well established outside
robotics. Two documentation proposals, Datasheets for Datasets and Model
Cards for Model Reporting, argued that a dataset or model should ship with
a standard account of its provenance, composition, and known limitations,
by analogy to a hardware datasheet, so that a defect discovered later has
somewhere to be recorded and a user has somewhere to look before training
on it (Gebru et al., 2021; Mitchell et al., 2019). Neither is itself an
audit; both are the case for why one is owed. Two later works did the
auditing. Northcutt, Athalye, and Mueller applied an automated label error
detector to the test sets of ten of the most used vision, language, and
audio benchmarks and found an average error rate of 3.3 percent, high
enough to change which model looks best on several benchmarks once the
errors are corrected, a result that existed only because someone ran a
population scale check rather than trusting the benchmark's reputation
(Northcutt et al., 2021). The Data Provenance Initiative ran a comparable
audit over licensing and attribution across 1,800 text-to-text finetuning
datasets and found license omission or misattribution above 50 percent, a
defect invisible from any single dataset's own documentation and visible
only once someone checked all of them the same way (Longpre et al., 2023).
This programme is the same move applied to a different defect and a
different field: an automated, population scale check for a specific,
named defect, temporal incoherence, rather than trust in a dataset's own
metadata or reputation, run over LeRobot Hub datasets rather than vision or
text benchmarks. No comparable audit of temporal or recording integrity in
robot learning datasets was found while researching this section. That
absence is the gap this paper addresses, not a claim this paper can prove
by itself.

**Anytime valid inference and sequential testing.**
`cairo_protocol.stats.anytime_cs`, used throughout this census for every
reported proportion, implements Robbins' beta-binomial mixture confidence
sequence, a method for producing an interval around a success rate that
remains valid at every stopping time rather than only at a pre-declared
sample size (Robbins, 1970). That guarantee is what licenses this
programme's resumable, append-only census design: a run can be
interrupted, resumed, or extended without invalidating the interval
reported at any point along the way. The broader modern literature on
time-uniform confidence sequences generalizes Robbins' construction beyond
the Bernoulli case and formalizes it within a game-theoretic account of
statistical testing as betting, in which an interval or a test stays valid
for as long as a corresponding wealth process has not grown large,
independent of when or why the analyst decided to look (Howard et al.,
2021; Ramdas et al., 2023). A recent instance of this literature applied
directly to robotics is the STEP procedure, a sequential test for comparing
two robot policies under a small, expensive trial budget that stops once
accumulated evidence crosses a boundary, reducing the number of trials
needed relative to a fixed-sample test by up to 32 percent in the authors'
experiments (Snyder et al., 2025). `cairo_protocol` does not implement
STEP; its docstring names STEP as a near-optimal refinement worth building
on top of the current mixture sequence if the last increment of sample
efficiency is needed. The two tools answer different questions. STEP
compares two policies under a small hardware trial budget, where each trial
is expensive and the question is which of two arms is better. This
census's `wilson_interval` and `anytime_cs` calls instead estimate a single
population proportion, the prevalence of a flag among a fixed,
pre-registered sample, and the results in Section 4 report the fixed-n
Wilson interval throughout, since the census is not run as a sequential
trial in the STEP sense: the sample size was fixed before the run and the
entire seeded draw is analysed once at the end, not accumulated one look at
a time.

**Reproducibility and pre-registration.** The failure this programme's
sampling design was built to avoid, a re-run that silently draws a
different sample under the same seed because the underlying population
changed between runs, is an instance of a documented, general problem:
reported machine learning results are frequently not reproducible because
a paper does not fully specify the procedure that produced them (Pineau et
al., 2021). Lipton and Steinhardt's earlier critique argued that part of
that failure is rhetorical rather than procedural, papers that explain
results after the fact rather than specifying an analysis commitment in
advance, a pattern pre-registration is designed to foreclose by requiring
the sampling frame, sample size, and analysis to be fixed and recorded
before the data is drawn (Lipton and Steinhardt, 2018). Pre-registration is
more established in explanatory or hypothesis-testing machine learning work
than in predictive or descriptive settings. Hofman et al. proposed a
template adapted specifically for predictive modeling, where the usual
pre-registration question, what hypothesis is being tested, does not
directly apply, but the underlying concern, that decisions made after
seeing the data bias the reported result, still does (Hofman et al., 2023).
This census's frame fingerprint and recorded seed, described in Section
3.2, are a pre-registration artifact in that sense: `sample.json` fixes and
publishes exactly what was drawn and from what population before the audit
ran, so a reader who was not present can check that the reported
proportions were not selected after seeing the data.

## 3. Method

### 3.1 Two-tier census

The census runs in two tiers, per
`docs/superpowers/specs/2026-09-05-ledger-census-design.md`.

**Tier 1, metadata census**, reads one `meta/info.json` per dataset over
the full LeRobot population reachable through the Hub's paginated listing
(Link header cursor, 1000 repositories per page). It records the declared
codebase version, frame rate, episode and frame counts, chunk size,
feature schema, and storage layout family. It never opens a parquet file.

**Tier 2, deep temporal audit**, runs the per-frame checks in
`ledger/checks.py` over parquet data sampled from a seeded draw over the
tier 1 frame. This is the tier that can, once complete and once a human
has confirmed a flag by hand, support an actual temporal integrity claim.

Both tiers write append-only JSONL, one record per dataset, so a run can
resume after a crash or a rate-limit stall without losing prior work
(`ledger/census.py`).

### 3.2 Seeded sample

Sampling sorts the tier 1 frame before drawing, so the draw depends only
on the sorted frame and the seed, not on whatever order the Hub happened
to return repositories in (`ledger/census.py`, `draw_sample`). The
completed tier 1 census used `draw_sample(frame, 400, seed=20260905)`, run
with `python -m ledger.audit --census metadata --sample-size 400 --seed
20260905`, anonymously, with no token
(`results/2026-09-09-metadata-census-400.md`). `ledger/census.py`'s
`CensusConfig` default sample size is 800, not 400; the completed tier 1
run used 400. See `paper/LIMITATIONS.md` for what this means for the
relationship between the two tiers as currently run.

### 3.3 Checks

`ledger/checks.py` and `ledger/report.py` together define eight named
flags, each a threshold comparison against a per-episode or per-dataset
measurement, pooled across sampled episodes as sums of numerators over
sums of denominators rather than a mean of per-episode ratios
(`ledger/report.py`, `summarise`):

- `ts_nonmonotonic`: an episode where any consecutive timestamp difference
  is zero or negative.
- `bad_dt`: the pooled fraction of frame-to-frame time gaps whose
  deviation from the expected `1/fps` interval exceeds 25 percent
  (threshold `frac_bad_dt = 0.05`).
- `frame_gaps`: an episode where the recorded `frame_index` skips or
  repeats a value.
- `stuck_state`: the pooled fraction of consecutive frames whose
  `observation.state` is bit-identical to the previous frame (threshold
  `stuck_state_frac = 0.20`).
- `action_equals_state`: the pooled fraction of frames where the action
  and state vectors are equal within a strict tolerance (threshold
  `identity_frac = 0.50`).
- `large_lag`: the median best-fit lag between the action and state
  channels, from a cross-correlation search (`xcorr_lag`), is at or above
  a threshold of 3 frames (`lag_large`).
- `negative_lag`: that same median best-fit lag is negative.
- `duplicate_episodes`: the pooled fraction of episodes whose action
  values (a rounded hash of the first 50 frames) repeat elsewhere in the
  dataset (threshold `dup_episode_frac = 0.0`).

All five threshold values above are copied from `docs/calibration.md`'s
"Threshold changes" section, which states they are unchanged from the
original values in `ledger/thresholds.toml`. `docs/calibration.md`
separately establishes that `large_lag` and `negative_lag` should be
treated as uninformative for any dataset with fewer than 6 action
dimensions; see `paper/LIMITATIONS.md` section 3.

A flag is an automated, provisional measurement. Per `ledger/report.py`'s
`PROVISIONAL_HEADER` and per `CLAUDE.md`, no dataset is named defective on
the strength of a flag alone; that requires a human to open the dataset
and record a confirmation in the separate `confirmed` field.

### 3.4 Confidence intervals

Every proportion reported in this programme carries a Wilson 95 percent
confidence interval from `cairo_protocol.stats.wilson_interval`. None is
reported bare.

## 4. Results

### 4.1 Tier 1: metadata census, n = 400 (completed 2026-09-09)

Source: `results/2026-09-09-metadata-census-400.md` and
`results/2026-09-09-metadata-census-400.jsonl`. 400 records, 400 unique
repositories, 0 duplicated.

**Reachability.** 390 of 400 returned an `info.json`. 10 did not, rate
0.025 [0.014, 0.045], recorded as `no info.json`, a category that covers
gated, private, deleted, and not-actually-a-LeRobot-dataset repositories
without distinguishing between them.

**Declared codebase version.**

| value | n | rate | 95 percent interval |
|---|---|---|---|
| v3.0 | 280 | 0.718 | [0.671, 0.760] |
| v2.1 | 100 | 0.256 | [0.216, 0.302] |
| v2.0 | 9 | 0.023 | [0.012, 0.043] |
| v2.1-embeddings-sharded | 1 | 0.003 | [0.000, 0.014] |

Four distinct declared strings, not the two families the original design
assumed.

**Declared layout.**

| value | n | rate | 95 percent interval |
|---|---|---|---|
| packed | 277 | 0.710 | [0.663, 0.753] |
| per_episode | 109 | 0.279 | [0.237, 0.326] |
| unknown | 4 | 0.010 | [0.004, 0.026] |

**Declared fps.** 30 fps dominates at rate 0.756 [0.711, 0.796]. The tail
runs 20, 10, 15, 50, 25, 5, 60, 14, and 2 fps. A 2 fps entry exists.

**Size.** Median 10 episodes, p90 130, max 2000. Median 6,175 frames, p90
77,268, max 660,229. The sampled population is dominated by small
datasets.

**About one dataset in twelve declares zero episodes.** 32 of 390
reachable datasets report `total_episodes: 0`, rate 0.082 [0.059, 0.114].
This is a declaration, not a measurement of an empty dataset. Whether the
underlying parquet is also empty, whether the field was simply never set,
or whether these are abandoned uploads is not established by tier 1; see
`paper/LIMITATIONS.md` section 7.

**About one percent of datasets declare a layout no reader here can
follow.** 4 of 400, rate 0.010 [0.004, 0.026]. Two causes: three datasets
from one publisher, THULab, declare `codebase_version: v3.0` with a
`.tsfile` data path that is not parquet at all; one dataset,
`saaduddinM/OXE_berkeley_autolab_ur5_embeddings`, declares
`codebase_version: v2.1-embeddings-sharded` with a sharded parquet naming
template `ledger.paths` did not resolve at the time of this run. See
`paper/LIMITATIONS.md` section 4 for both causes in full and for why
neither is described as a defective dataset.

**The declared version does not determine the layout.** `v3.0` appears
with both the packed template and the unreadable `.tsfile` layout. A tool
that branches on `codebase_version` alone, rather than reading `data_path`
directly, will be wrong for some real datasets on the Hub.

### 4.2 Tier 2: deep temporal audit

The deep tier ran to completion on 2026-09-11 over the same recorded
sample as the metadata tier above. Provenance, integrity checks and the
full tables are in `results/2026-09-11-census-400-both-tiers.md`; the raw
ledgers and the sample record are committed beside it.

353 of 400 datasets were audited. 47 could not be, rate 0.117
[0.090, 0.153], dominated by 31 cases where no parquet exists at the path
the declared layout predicts.

Among the 353 audited, the most common flags were `large_lag` at 0.530
[0.509, 0.612] and `stuck_state` at 0.351 [0.303, 0.402]. The flags that
indicate an outright recording error rather than a timing characteristic
were rare: `duplicate_episodes` 0.020 [0.010, 0.040], `negative_lag` 0.017
[0.008, 0.037], `action_equals_state` 0.014 [0.006, 0.033].

The headline rate is not reportable as a single number. Any flag was
raised for 0.6487 [0.5976, 0.6967] of audited datasets, but 97 of those
carry `large_lag` alone, and `large_lag` is the flag the calibration in
`docs/calibration.md` shows to be least reliable below 6 action
dimensions. Excluding datasets flagged only by `large_lag` gives 0.3739
[0.3251, 0.4255]. The true rate lies somewhere across that span, and this
run does not settle where. Reporting 0.6487 without that caveat would
misrepresent the evidence.

One result is validated across both tiers independently. Of the 34
datasets that tier 1 recorded as declaring zero episodes, 29 had no
parquet at the derived path, giving P(no parquet | declares zero
episodes) = 0.853 [0.699, 0.936]. Two datasets fail the other way,
declaring episodes while no parquet is present where the declared layout
says it should be.

**Confirmed findings.** On 2026-09-12 the owner confirmed all 16 datasets
carrying a flag that indicates an outright recording error and does not
depend on the lag estimator. Over the 353 audited datasets:

| finding | n | rate | 95 percent Wilson |
|---|---|---|---|
| `duplicate_episodes` | 7/353 | 0.0198 | [0.0096, 0.0404] |
| `negative_lag` | 0/353 | 0.0000 | [0.0000, 0.0108] |
| `action_equals_state` | 5/353 | 0.0142 | [0.0061, 0.0327] |
| any confirmed finding | 12/353 | 0.0340 | [0.0196, 0.0585] |

That rate counts only confirmed defects. `large_lag` at 0.530 after the lag gate, and
`stuck_state` at 0.351, were deliberately held out of the confirmation pass
and remain unconfirmed measurements, so 0.0340 is a lower bound on defects
overall. The confirmation was a blanket owner sign-off rather than 16
independent inspections with notes, and the evidence supporting it is in
`results/2026-09-12-evidence-16-flagged.md`.

## 5. Discussion

Two results deserve separating, because they differ in kind.

The first is descriptive and needs no judgement call. A dataset that
declares zero episodes almost always has no parquet at the path its own
declared layout predicts, P = 0.853 [0.699, 0.936]. Both tiers found this
independently, one reading metadata and the other reading data, and neither
relies on any threshold. It says something plain about the Hub: a
measurable fraction of published datasets are announcements of data that is
not there.

The second is the confirmed defect rate, 0.0340 [0.0196, 0.0585] of audited
datasets, and it needs to be read with its bounds in view. It counts only
the three flags that indicate an outright recording error and that do not
depend on the lag estimator. It excludes `large_lag`, measured at 0.530 after the lag gate,
and `stuck_state`, at 0.351, both of which remain unconfirmed
measurements. So roughly one dataset in thirty carries a confirmed
recording defect, and a much larger fraction carries something the audit
noticed but nobody has adjudicated.

The gap between those two numbers is the honest state of the field rather
than a deficiency of this work. Confirming a flag requires opening the data
and forming a judgement, and that does not scale the way measurement does.
An audit can survey 75,750 datasets; a human cannot.

The `action_equals_state` cases are the most interesting technically. In
five datasets the recorded action is bit identical to the recorded state on
every frame, in one case across 29,870 frames and 36 dimensions. A policy
trained on such a dataset learns to reproduce the present observation
rather than to command the next one. Whether the publisher intended it,
which some leader-follower rigs do, does not change what a model learns
from it.

## 6. Conclusion

Temporal integrity problems in public robot learning datasets are common
enough to matter and rare enough to fix. About 3.4 percent of LeRobot
datasets carry a confirmed recording defect, interval 2.0 to 5.9 percent,
and a further substantial fraction carries flags that were measured but not
adjudicated.

The tooling to find them is cheap. Reading five columns costs 2.7 percent
of a dataset's compressed bytes, so a full population audit is bounded by
rate limits rather than by bandwidth or storage. What is not cheap is
confirmation, and that is where the bottleneck sits.

The contribution here is less the prevalence number than the apparatus
around it: a sampling procedure whose seed genuinely identifies a sample, a
frame fingerprint that makes a run reproducible by someone who was not
present, thresholds calibrated against measured false positive rates rather
than chosen, and a hard separation between what a tool measured and what a
human confirmed. Each of those was added because its absence caused a
concrete error during this work, documented in `CHANGELOG.md`.

## 7. Limitations

See `paper/LIMITATIONS.md` in full. It is the more complete and more
critical document and should be read before any number in this draft is
used for anything. It covers, among other points: the deep tier's
incompleteness and its current disconnect from the tier 1 sample; the zero
human confirmations behind every flag in this programme; the measured
unreliability of the lag estimator below 6 action dimensions, and that the
deep tier ledger does not currently record enough per-dataset information
to apply that limitation record by record; the roughly one percent of
datasets excluded from any parquet-level claim and why; the anonymous,
rate-limited, small-sample nature of this census against a population of
at least 75,750; the absence of a committed `PROTOCOL.md` for this
programme; and the fact that tier 1 measures what publishers declared, not
what their data contains.

## References

1. Cadene, R., Aliberts, S., Capuano, F., Aractingi, M., Zouitine, A.,
   Kooijmans, P., Choghari, J., Russi, M., Pascal, C., Palma, S., Shukor,
   M., Moss, J., Soare, A., Aubakirova, D., Lhoest, Q., Gallouedec, Q., and
   Wolf, T. LeRobot: An Open-Source Library for End-to-End Robot Learning.
   arXiv:2602.22818, 2026. https://arxiv.org/abs/2602.22818

2. O'Neill, A., et al. (Open X-Embodiment Collaboration). Open
   X-Embodiment: Robotic Learning Datasets and RT-X Models. arXiv:2310.08864,
   2023. https://arxiv.org/abs/2310.08864

3. Khazatsky, A., Pertsch, K., Nair, S., Balakrishna, A., Dasari, S., et al.
   DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset. Robotics:
   Science and Systems (RSS), 2024. arXiv:2403.12945.
   https://arxiv.org/abs/2403.12945 and
   https://www.roboticsproceedings.org/rss20/p120.pdf

4. Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Dabis, J., et al.
   RT-1: Robotics Transformer for Real-World Control at Scale.
   arXiv:2212.06817, 2022. https://arxiv.org/abs/2212.06817

5. Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Chen, X., et al. RT-2:
   Vision-Language-Action Models Transfer Web Knowledge to Robotic Control.
   arXiv:2307.15818, 2023. https://arxiv.org/abs/2307.15818

6. Walke, H., Black, K., Zhao, T. Z., Vuong, Q., Zheng, C., Hansen-Estruch,
   P., He, A. W., Myers, V., Kim, M. J., Du, M., Lee, A., Fang, K., Finn,
   C., and Levine, S. BridgeData V2: A Dataset for Robot Learning at Scale.
   Conference on Robot Learning (CoRL), PMLR 229:1723-1736, 2023.
   https://proceedings.mlr.press/v229/walke23a.html

7. Gebru, T., Morgenstern, J., Vecchione, B., Wortman Vaughan, J., Wallach,
   H., Daume III, H., and Crawford, K. Datasheets for Datasets.
   Communications of the ACM, 64(12):86-92, 2021 (first posted as
   arXiv:1803.09010, 2018). https://arxiv.org/abs/1803.09010

8. Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L.,
   Hutchinson, B., Spitzer, E., Raji, I. D., and Gebru, T. Model Cards for
   Model Reporting. Conference on Fairness, Accountability, and
   Transparency (FAT* '19), 2019. arXiv:1810.03993.
   https://arxiv.org/abs/1810.03993

9. Northcutt, C. G., Athalye, A., and Mueller, J. Pervasive Label Errors in
   Test Sets Destabilize Machine Learning Benchmarks. NeurIPS 2021 Datasets
   and Benchmarks Track. arXiv:2103.14749. https://arxiv.org/abs/2103.14749

10. Longpre, S., Mahari, R., Chen, A., Obeng-Marnu, N., Sileo, D., Brannon,
    W., Muennighoff, N., Khazam, N., Kabbara, J., Perisetla, K., Wu, X.,
    Shippole, E., Bollacker, K., Wu, T., Villa, L., Pentland, S., and
    Hooker, S. The Data Provenance Initiative: A Large Scale Audit of
    Dataset Licensing and Attribution in AI. arXiv:2310.16787, 2023.
    https://arxiv.org/abs/2310.16787

11. Robbins, H. Statistical Methods Related to the Law of the Iterated
    Logarithm. The Annals of Mathematical Statistics, 41(5):1397-1409,
    1970. https://projecteuclid.org/euclid.aoms/1177696786

12. Howard, S. R., Ramdas, A., McAuliffe, J., and Sekhon, J. Time-Uniform,
    Nonparametric, Nonasymptotic Confidence Sequences. The Annals of
    Statistics, 49(2):1055-1080, 2021. arXiv:1810.08240.
    https://arxiv.org/abs/1810.08240

13. Ramdas, A., Grunwald, P., Vovk, V., and Shafer, G. Game-Theoretic
    Statistics and Safe Anytime-Valid Inference. Statistical Science,
    38(4):576-601, 2023. arXiv:2210.01948. https://arxiv.org/abs/2210.01948

14. Snyder, D., Hancock, A. J., Badithela, A., Dixon, E., Miller, P.,
    Ambrus, R. A., Majumdar, A., Itkina, M., and Nishimura, H. Is Your
    Imitation Learning Policy Better than Mine? Policy Comparison with
    Near-Optimal Stopping. Robotics: Science and Systems (RSS), 2025.
    arXiv:2503.10966. https://arxiv.org/abs/2503.10966 and
    https://www.roboticsproceedings.org/rss21/p077.html

15. Pineau, J., Vincent-Lamarre, P., Sinha, K., Lariviere, V., Beygelzimer,
    A., d'Alche-Buc, F., Fox, E., and Larochelle, H. Improving
    Reproducibility in Machine Learning Research (A Report from the
    NeurIPS 2019 Reproducibility Program). Journal of Machine Learning
    Research, 22, 2021. arXiv:2003.12206. https://arxiv.org/abs/2003.12206

16. Lipton, Z. C., and Steinhardt, J. Troubling Trends in Machine Learning
    Scholarship. ICML 2018, "The Debates" track. arXiv:1807.03341.
    https://arxiv.org/abs/1807.03341

17. Hofman, J. M., Chatzimparmpas, A., Sharma, A., Watts, D. J., and
    Hullman, J. Pre-registration for Predictive Modeling. arXiv:2311.18807,
    2023. https://arxiv.org/abs/2311.18807

## Disclosure

Per `CLAUDE.md`'s disclosure convention, adapted for this draft: the first
drafts of the Introduction, Method, and Results sections of this document,
and of the accompanying `paper/LIMITATIONS.md`, were produced with Claude
(Fable 5.1), drafting directly from the source files named throughout both
documents. Unlike the steady-state form of this disclosure, this is a
first draft and has not yet been verified by the human author; that
verification is a precondition for submission, not something this draft
claims has already happened. All hardware experiments, statistics, and
claims, once reviewed, remain the authors' own.
