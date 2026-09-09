# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-09-09

The M-02 LEDGER release. The dataset audit becomes a resumable two tier
census over the whole LeRobot population, and the repository becomes an
installable package with its own offline CI.

### Added

- `ledger.census`: two tier Hub survey. Tier 1 records metadata for every
  LeRobot dataset. Tier 2 runs the temporal checks over a seeded random
  sample and reports prevalence through `cairo_protocol.wilson_interval`, so
  no rate is ever published without a confidence interval.
- `ledger.sources`: a `SampleSource` protocol with four implementations,
  streaming, download, local and synthetic. The census streams parquet over
  HTTP range reads with column projection, and falls back to whole file
  download when a range read fails.
- `ledger.hubclient`: token aware Hub client with `Retry-After` compliance,
  exponential backoff with jitter, a global minimum request interval, and
  Link header cursor pagination. The token is never written to the repository
  and never appears in a repr, a header dump or an error message.
- `ledger.paths`: derives every parquet path from the `data_path` template in
  `meta/info.json`, so the census makes no calls to the tree API.
- `ledger.checks`: temporal checks in an open registry. Adding a check is
  adding a function.
- `ledger.synth`: seeded synthetic episode generator with one injector per
  defect, plus parquet fixture writers for the v2.0 and v3.0 layouts.
- `ledger.config`: thresholds moved out of code into validated
  `ledger/thresholds.toml`.
- Resume: an append only JSONL ledger keyed by `repo@revision`, flushed per
  dataset. The revision is the real Hub sha, read from the `X-Repo-Commit`
  header of a response the run already makes, so a dataset that changed on
  the Hub is audited again rather than skipped.
- `--files all`, which the README promised in v0 and the CLI never had.
- `docs/calibration.md`: measured threshold calibration, including the
  false positive sweep and the low degree of freedom lag limitation.
- `pyproject.toml`, `LICENSE` (Apache-2.0), `CITATION.cff`, this changelog,
  `CONTRIBUTING.md` and `.editorconfig`.
- `.github/workflows/tests.yml`: the offline suite on Linux and Windows
  across Python 3.11 and 3.13, plus ruff and three repository convention
  gates.

### Changed

- `ledger.audit` is now a thin CLI over the modules above. Every v0 command
  keeps working, and `_synthetic`, `audit_frame` and `summarise` stay
  importable so `tests/test_audit.py` passes unmodified.
- Test suite grew from 8 tests to 148 tests and 1 deliberately deselected
  live Hub test, plus 17 doctests.

### Fixed

- `.github/workflows/ursim-ci.yml` installed a dependency list that predated
  `hypothesis` and `huggingface_hub`, so its test step could not have
  collected. It now installs the package with its test extra.
- Line endings normalised to LF via `.gitattributes`. The kit is developed on
  Windows and its CI runs on Ubuntu.

### Notes on scope

An exhaustive every file crawl of the Hub is deliberately not attempted.
Measured on 2026-09-05: at least 12,000 LeRobot datasets, anonymous requests
returning HTTP 429 within tens of requests on both the api and resolve hosts,
and v2.0 datasets storing one parquet per episode, with one dataset alone at
209,880 files. A partial crawl also has no sampling frame and so supports no
prevalence claim. A seeded random sample is both cheaper and more defensible.

### Behaviour guarantee

`tests/v0_reference.py` is a frozen snapshot of v0. Equivalence tests assert
that the refactored pipeline reproduces it exactly across every defect, for
single and multi part inputs. The guarantee was verified as non tautological:
a deliberate change to the pooled denominator makes both equivalence tests
fail.

## [0.1.0] - 2026-09-02

Initial kit.

### Added

- `cairo_protocol.stats`: Wilson intervals, Barnard exact test, bootstrap
  difference intervals, anytime valid confidence sequences by Robbins beta
  binomial mixture, sequential comparison of two policies, and fixed n power.
- `ledger.audit` v0: temporal integrity audit sampling named LeRobot datasets.
- `lerobot_ur`: Universal Robots e-Series follower over RTDE with velocity
  and force clamps, and a URSim smoke test used as a CI gate.
- `protocols/PROTOCOL_TEMPLATE.md`: the pre-registration file a programme
  commits before its first trial.

[0.2.0]: https://github.com/tarekokashha/mizan-kit/releases/tag/v0.2.0
[0.1.0]: https://github.com/tarekokashha/mizan-kit/releases/tag/v0.1.0
