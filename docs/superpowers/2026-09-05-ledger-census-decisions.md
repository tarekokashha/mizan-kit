# M-02 LEDGER census: decision record

Preserved from the execution ledger of
docs/superpowers/plans/2026-09-05-ledger-census.md, merged to main on
2026-09-09 as commit a7d159f.

This file records every ruling taken during implementation, each with what
it would cost if it turns out wrong, plus the deferred minor findings that
were triaged as safe to ship. It is kept because those decisions are not
recoverable from the diff alone.


Spec: docs/superpowers/specs/2026-09-05-ledger-census-design.md (read, reachable)
Branch: feat/ledger-census
Base at start: b4f8ac5
Test command: ./.venv/Scripts/python.exe -m pytest -q

## Pre-flight conflict scan

### Cross-task rows (tasks sharing a file or interface)

| pair | produced / consumed | finding |
|------|--------------------|---------|
| T1 to T2 | `make_episodes(defect, seed, **kw)` | consistent |
| T1 to T4 | `make_episodes(fps, n_eps, T, n_joints, lag, defect, seed)` | consistent |
| T1 to T7 | `make_episodes`, `write_v20_fixture`, `write_v30_fixture` | consistent |
| T1 to T8 | `make_episodes(defect, seed)` | consistent |
| T1 to T9 | `make_episodes(n_eps, T, seed)`, `write_v30_fixture` | consistent |
| T2 to T4 | `episode_stats(ep, fps)`, `run_checks(stats)`, `xcorr_lag` | consistent |
| T2 to T8 | registry drives report schema | **CONFLICT 1** |
| T3 to T4 | `load_thresholds()`; T4 may edit `thresholds.toml` created by T3 | consistent, ordering correct |
| T3 to T8 | `load_thresholds()` | consistent |
| T5 to T7 | `derive_paths(info, limit)` | consistent |
| T6 to T7 | `HubClient()` | consistent |
| T8 to T9 | `DatasetReport`, `summarise`, per-episode audit fn | **CONFLICT 2** |
| T8 to T10 | `ledger/audit.py` re-exports for `tests/test_audit.py` | **CONFLICT 2** (same root) |
| T9 to T10 | `CensusConfig`, `run_deep_tier`, `run_metadata_tier` | consistent |

### Per-task internal consistency rows

| task | tests vs code it specifies | files created vs later touched | finding |
|------|---------------------------|-------------------------------|---------|
| T1 | generator tests match generator; fixture tests match writers | `conftest.py` created here, used by all later | **CONFLICT 3** |
| T2 | 6 tests match registry and `xcorr_lag` | `checks.py` consumed by T4, T8 | see CONFLICT 1 |
| T3 | 3 tests match loader and validation | `thresholds.toml` edited by T4 | consistent |
| T4 | properties match generator and thresholds | edits T3 output only on failure | consistent |
| T5 | 6 tests match both layout families | none | consistent |
| T6 | 5 tests match client, all monkeypatched, no network | none | consistent |
| T7 | 5 tests cover Local and Synthetic only | Streaming and Download untested offline | **CONFLICT 4** |
| T8 | 6 tests match report API | v0 flag names pinned | see CONFLICT 2 |
| T9 | 6 tests match census API | consumes every module | consistent |
| T10 | 5 tests match CLI | rewrites `audit.py` | see CONFLICT 2 |

### Rulings

**Ruling 1 (CONFLICT 1): the registry holds per-episode scalar checks only; lag and duplicates are declared aggregates.**
The plan claims every flag derives from `REGISTRY`, but `ts_nonmonotonic`,
`frame_gaps`, `large_lag`, `negative_lag` and `duplicate_episodes` are not
per-episode scalars, and Task 2 parks `dup_episode_frac` as a function
returning NaN, which is a placeholder a reviewer would rightly flag.
Decision: `REGISTRY` carries per-episode scalar checks, gaining
`ts_nonmonotonic` and `frame_gap` alongside the existing three. Lag and
duplicate detection are cross-episode aggregates computed in `report.py`
from `xcorr_lag` and `action_head_hash`, declared explicitly as
`AGGREGATE_FLAGS` so the schema still cannot drift silently. Remove the
NaN-returning `dup_episode_frac` registration.
Cost if wrong: `report.py` carries two flag sources instead of one, and a
future check author must know which list to add to. Rework is local to
`checks.py` and `report.py`.

**Ruling 2 (CONFLICT 2): keep v0's `audit_frame(df, fps) -> dict` exactly; do not introduce `audit_dataframe`.**
The plan's Task 8 renames the per-dataframe function to
`audit_dataframe(df, fps) -> list[dict]`, but the untouchable
`tests/test_audit.py` calls `summarise("t", {"fps": 30}, [audit_frame(df, 30.0)])`,
wrapping a single dict in a list. A list-returning function would make that
a list of lists and break a test the Global Constraints forbid changing.
Decision: `report.py` exposes `audit_frame(df, fps) -> dict` with v0's exact
aggregate-per-dataframe semantics. Task 8 and Task 9 tests use that name.
`audit_dataframe` is not created.
Cost if wrong: none identified. This is strictly the conservative reading,
and it is what keeps the v0 contract byte identical.

**Ruling 3 (CONFLICT 3): `conftest.py` must use `config.getoption("markexpr")`.**
The plan writes `config.getoption("-m")`, which raises `ValueError` in
pytest because options are addressed by dest name, not by flag. Decision:
use `markexpr`. Also correct the skip reason string to `-m network`, since
`--m network` in the plan is not valid pytest syntax.
Cost if wrong: the network marker either never skips or always skips.
Caught immediately by the first suite run.

**Ruling 4 (CONFLICT 4): `StreamingSource` and `DownloadSource` get constructor-level tests only, plus one network-marked live test.**
Both hit the Hub, and the anonymous 429 ceiling measured in the spec makes
them untestable in the default offline suite. Decision: Task 7 additionally
asserts that both classes satisfy the `SampleSource` protocol and that
`StreamingSource` builds without a token, and adds one
`@pytest.mark.network` live read that is deselected by default. Correctness
of the projection path is already covered by `read_projected` through
`LocalSource`.
Cost if wrong: a Hub-specific regression in the two network sources escapes
the offline suite. Mitigated by the shared `read_projected` code path.

**Ruling 5 (advisory to implementers, not a conflict): pandas 3.0.5 parquet round trip of list columns.**
`make_episodes` builds `action` and `observation.state` as object columns of
numpy arrays. Writing those to parquet under pandas 3.0.5 with pyarrow 25
may materialise them as lists rather than arrays on read back. `_stack`
already tolerates both. Implementers must verify the round trip in Task 1
rather than assume it.
Cost if wrong: fixture tests fail loudly in Task 1, before anything depends
on them.

## Progress

Task 1: implemented (commit c6d3733, 20 passed). Implementer verified both
  ruling premises empirically and found neither reproduced on this exact
  stack: the parquet round trip did not lose shape under pandas 3.0.5 with
  pyarrow 25.0.1, and config.getoption("-m") did not raise in pytest 9.1.1.
  Both rulings were applied regardless, since markexpr and an explicit
  pyarrow schema are the canonical forms. Correction to Ruling 3: its stated
  justification was wrong, its instruction was still right.
Task 1: review -> Spec PASS. Quality CHANGES REQUESTED, 1 Important, 2 Minor.
Task 1: Ruling: the Important finding (test_every_defect_generates asserts
  only len(df) > 0) is plan-mandated, because that test came verbatim from
  the plan. The reviewer is upheld. A test that asserts nothing is a defect
  regardless of what mandated it, and the duplicate and swapped injectors
  are exactly the two whose behaviour nothing else pins. Entering the fix
  loop rather than parking.
  Cost if wrong: a few minutes of rework on a test file, no production code.
Task 1: minor (deferred): pyarrow.parquet import sits inside _to_parquet
  while pyarrow is imported at module top.
Task 1: minor (deferred): near-identical info dict construction duplicated
  between write_v20_fixture and write_v30_fixture.
Task 1: fix round 1/5 dispatched (resumed original implementer).

Tasks 3, 5, 6: implemented as one batched dispatch, three commits
  5a34418, af01fe0, 30baf4b. Full suite 34 passed.
Tasks 3, 5, 6: Ruling: the Task 6 brief's own sample code stored the bearer
  token in self._headers, which contradicts that task's own
  test_token_is_never_in_repr assertion. This is a defect in my plan, and a
  security relevant one. The implementer changed the implementation rather
  than the test, holding the token in a private attribute and merging it
  into the headers only at request build time. Upheld: fixing the code and
  keeping the assertion is the correct direction, since the assertion
  encodes the actual requirement.
  Cost if wrong: if the merge at request build time is faulty the
  Authorization header never ships, which is a silent auth failure. The
  reviewer has been told explicitly to verify the header still reaches the
  request.
Tasks 3, 5, 6: review dispatched.
Task 1: fix round 1/5 (1 addressed, 0 open; commits 30baf4b..7b386d4).
  Re-review verdict: all findings addressed, no new breakage.
Task 1: complete (commits b4f8ac5..7b386d4, review clean).

Tasks 3, 5, 6: review -> Spec PASS, Quality APPROVED, 1 Important, 3 Minor.
  Reviewer independently verified the token fix: confirmed urllib HTTPError
  str and repr carry only code and message so RateLimited cannot leak the
  token, and confirmed the Authorization header does reach the built
  request. The deviation was correct and necessary.
Tasks 3, 5, 6: Ruling: the Important finding stands. The suite asserts only
  the token's ABSENCE from _headers and __repr__, which is one sided. A
  refactor could stop sending the header entirely with every test still
  green, and since the entire reason this client exists is the measured
  anonymous 429 ceiling, a silently unauthenticated client is this module's
  worst failure mode. Entering the fix loop.
  Cost if wrong: one extra offline test. No production behaviour changes.
Tasks 3, 5, 6: minor (deferred): self._token still holds the raw secret in
  the instance __dict__, so vars(client) would show it.
Tasks 3, 5, 6: minor (deferred): config.py raises a bare ValueError from
  float() for a non-numeric threshold without naming the key.
Tasks 3, 5, 6: minor (deferred): Retry-After honoured only for digit-only
  values; an HTTP-date value silently falls back to exponential backoff.
Tasks 3, 5, 6: fix round 1/5 dispatched (resumed original implementer).

Tasks 2, 7: implemented, commits 4c7083a and 0429cca. 57 passed, 1 skipped.
  Rulings 1 and 4 reported applied. Implementer added 4 tests beyond the
  required two, covering ts_nonmonotonic and frame_gap directly because no
  synth defect exercises them. Reasonable, kept.
Tasks 2, 7: note: the implementer ran the suite once with -m network as a
  post hoc sanity check, making one real read only HTTP request to
  huggingface.co. Benign, a public metadata read, but recorded because the
  default suite is meant to be offline.
Tasks 2, 7: review dispatched, with the v0 arithmetic equivalence check
  called out as the crux.
Tasks 3, 5, 6: fix round 1/5 (1 addressed, 0 open; commits 0429cca..f19673a).
  Implementer went beyond the instruction and also stubbed
  huggingface_hub.get_token, because reading its source showed a fallback to
  the cached CLI login file and OIDC, so clearing only the two env vars
  would have left the test flaky on a machine with a real login. Correct.
  They also verified the positive test has teeth by temporarily breaking
  header sending and watching it fail. Accepted.
Tasks 3, 5, 6: complete (commits c6d3733..f19673a, review clean).

Tasks 2, 7: review -> Spec PASS, Quality APPROVED, 0 Critical, 0 Important,
  1 Minor. Crux confirmed: the ported arithmetic was checked line by line
  against v0 and does not diverge, including the np.isclose(action, state,
  atol=0.0) argument order that preserves the default rtol asymmetry, and
  the T<30 and len(aa)<20 guards in xcorr_lag.
Tasks 2, 7: minor (deferred): read_projected tolerates an audit column
  missing from a file, but no test pins that direction. Only the extra
  column dropped direction is covered.
Tasks 2, 7: complete (commits d94d2ec..0429cca, review clean).

**Ruling 6 (raised by the Tasks 2, 7 reviewer, binds Task 8): report.py must
aggregate POOLED, exactly as v0 does, not as a mean of per-episode
fractions.** Verified directly against ledger/audit.py summarise: v0
computes frac_bad_dt as sum(bad_dt) / sum(n_dt) across parts, and the same
pooled form for stuck_state_frac and identity_frac. A mean of per-episode
fractions gives a different answer whenever episodes have unequal length,
which is exactly what the drops defect and real Hub data produce. Getting
this wrong would silently change every number in the published README table
while leaving the whole suite green.
  Cost if wrong: every prevalence figure in the paper shifts by an unknown
  amount, and the v0 numbers stop reproducing. This is the single highest
  risk item in the branch.

Tasks 4, 8: implemented, commits 733e419 and 450b6e9. 71 passed, 1 skipped.
  Controller independently verified v0 equivalence across all six defects
  for a single part: every numeric field within 1e-12, flags byte identical.
  Ruling 6 held.
Tasks 4, 8: review -> Spec PASS. Quality CHANGES REQUESTED, 1 Critical,
  2 Important, 1 Minor. Reviewer independently reproduced the calibration
  numbers rather than trusting the doc, and confirmed multi part pooling
  correct on a 4 part run where a naive mean of per part medians would have
  given lag_frames 1.75 instead of 1.0.

**Ruling 7 (Critical, upheld): the relaxed lag property must not derive its
own ground truth from the function under test.**
tests/test_calibration.py computes r_true by calling xcorr_lag again with
lags=(lag,). The reviewer demonstrated concretely that a stubbed xcorr_lag
which ignores its inputs and returns a fixed triple would pass: lag==0 draws
take the early return, and every other draw gets r_true from the same fixed
score, so gap is 0. A label only sign flip also passes. That is a property
test that cannot fail for the failure mode it exists to catch. The relaxation
itself was justified and stays; the self referential oracle does not.
  Cost if wrong: a genuinely broken lag estimator ships undetected, and
  large_lag and negative_lag are two of the eight published flags.

**Ruling 8 (Important, upheld): the low DoF lag limitation must be user
facing, not a test comment.**
The measured limitation is that xcorr_lag does not reliably resolve the
argmax below 6 action dimensions. large_lag and negative_lag depend on it,
the README presents lag_frames as a headline column, and many real LeRobot
datasets are low DoF. A caveat that lives only in tests/test_calibration.py
and a private report is not disclosure. It goes in docs/calibration.md as a
named limitations section and in the report.py module docstring.
  Cost if wrong: a user audits a low DoF dataset and reads a lag flag that
  was never validated in that regime. This is exactly the hardware opacity
  failure the kit's own CLAUDE.md kill list names.

Tasks 4, 8: minor (deferred): test_error_report_is_recorded_not_raised pins
  only dataclass defaults and exercises no path that could raise.
Tasks 4, 8: note: hypothesis is not in requirements.txt though the suite now
  needs it. Definition of done requires a clean checkout to run. Folded into
  the Task 10 dispatch.
Task 9: first dispatch terminated early by a session rate limit before any
  commit. Working tree was clean, nothing half written. Re dispatching.
  SendMessage is no longer available this session, so fix rounds use fresh
  implementers carrying the brief and report paths.
Tasks 4, 8: fix round 1/5 (3 addressed, 0 open; commit dcefb12). Finding A
  teeth proof: both stubs, constant triple and sign flipped label, failed the
  fixed property as required. Finding B passed on first run, report.py needed
  no change, pooling was already correct. Finding C surfaced in
  docs/calibration.md and the report.py docstring.
Task 9: implemented (commit e370c83). 90 passed, 1 skipped.

**Ruling 9 (raised by the Task 9 implementer): record the real Hub revision,
do not ship an empty resume key.**
The implementer flagged that no source exposes a revision, so every record's
revision is the empty string and repo@revision degenerates to repo@. Resume
still works but cannot tell two audits of the same repo at different Hub
revisions apart, which the spec explicitly requires so a changed dataset is
re audited rather than skipped. I verified against the live Hub that the
resolve response get_info ALREADY makes carries the exact sha in the
X-Repo-Commit header, measured as f641879e for lerobot/svla_so101_pickplace.
So the fix costs zero extra requests. HubClient.get_info must capture that
header and census must record it. Folded into Task 10.
  Cost if wrong: a census re run silently skips datasets that changed on the
  Hub since the previous run, so the prevalence table quietly goes stale.
Task 9: review -> Findings A, B, C all ADDRESSED. Spec PASS, Quality
  APPROVED, 0 Critical, 1 Important, 1 Minor. Reviewer independently derived
  that both teeth-proof stubs fail on the new abs(best-lag)<=1 assert alone,
  and confirmed per dataset flush by reading the file back mid run.
Task 9: minor (deferred): the independent lag helper reimplements the same
  normalisation formula as xcorr_lag rather than an independently derived one
  such as per joint corrcoef, so a shared formula bug could hide from both.
  Not blocking: the required teeth come from the label assert, not the helper.
Task 9: complete (commits 450b6e9..e370c83, review clean).

Task 10: Ruling 10 (controller, before audit.py was touched): freeze the v0
  reference first. tests/test_report.py imported ledger.audit.audit_frame and
  summarise as its v0 oracle. Rewriting audit.py into a thin re-export of
  report.py would have made those tests compare report.py to itself, still
  pass, and silently destroy the strongest guarantee on the branch.
  Decision: snapshot v0 into tests/v0_reference.py, repoint the equivalence
  tests there, then allow the re-export.
  Cost if wrong: the refactor equivalence guarantee becomes decorative and a
  future behaviour change ships unnoticed.
  Verified by the controller: a deliberate one character break in report.py's
  pooled denominator made BOTH equivalence tests fail, and they passed again
  on restore with a clean tree. The guarantee is real.
Task 10: implemented across commits 1484f67, 0a66930, c79015f, 164dbff and
  3969653. 114 passed, 1 skipped.
Task 10: controller verified all five acceptance criteria independently, and
  confirmed the demo output is byte identical to the v0 baseline captured at
  session start, including stuck_state_frac 0.249 and dup_episode_frac 0.833.
Task 10: note: on a cold cache a resumed run still fetches info.json before
  it can decide to skip, because the revision sha arrives with that same
  response. One cheap request per dataset. The written key and the resumed
  decision are both correct.
Task 10: complete.

FINAL whole-branch review -> CHANGES REQUIRED: 2 Critical, 2 Important,
  1 Minor. All eight deferred minors triaged as ship.
  C1: the metadata tier recorded only repo, codebase, fps and source, where
  the spec and README promise six fields including episode and frame counts,
  chunk size, feature schema and layout family. Controller verified: a tier 1
  record populated four fields, and info.json already carried every missing
  value. A genuine spec fidelity failure in the headline tier 1 deliverable.
  C2: get_info caught RateLimited and returned None, indistinguishable from a
  404, so an exhausted-backoff 429 was recorded as done and --resume never
  retried it. Precisely the failure the client exists to survive.
  I3: derive_paths treated limit=None as one file for packed v3.0 layouts but
  as no cap for per-episode v2.0. Controller verified 50 files versus 1. So
  --files all silently audited a single file on every v3.0 dataset, directly
  contradicting the README written earlier in this session.
  I4: _stack's bare except turned malformed columns into indistinguishable
  "no data". M5: Link header rel="next" parsed by position.
FINAL fix wave: one dispatch, all five fixed, commits 6d7397d, 5f80936,
  7087b4b, 529453d, a7d159f. 118 -> 148 passed, 1 skipped.
Controller verification after the fix wave:
  tier 1 now records codebase, fps, total_episodes, total_frames, chunk_size,
  layout_family, feature_names and n_features.
  packed --files all now discovers all 3 files of a 3 file fixture, and
  limit=2 yields 2.
  golden regenerated twice, but every pre-existing value is unchanged and only
  columns were appended. Confirmed field by field, including
  stuck_state_frac 0.2491638795986622 and dup_episode_frac 0.8333333333333334.
  demo flags byte identical to the v0 baseline captured at session start.
  148 passed, 17 doctests, protected paths untouched, clean tree, 24 commits.
