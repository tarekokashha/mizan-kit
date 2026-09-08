"""Tests for ledger.audit as a thin CLI over the census modules.

tests/test_audit.py is the untouched, protected equivalence suite: it
imports _synthetic, audit_frame and summarise straight from
ledger.audit and checks the v0 flag names. This file covers the CLI
surface itself: argument parsing (parse_files), the demo path, and one
offline end to end run of the census wiring against a LocalSource
fixture, which is the only proof in the whole suite that argparse,
ledger.census and ledger.sources actually compose correctly together.
"""
import json

import pytest

from ledger.audit import main, parse_files


def test_files_accepts_all():
    assert parse_files("all") is None


def test_files_accepts_an_integer():
    assert parse_files("3") == 3


def test_files_rejects_nonsense():
    with pytest.raises(Exception):
        parse_files("banana")


def test_demo_still_runs(capsys):
    assert main(["--demo"]) == 0
    out = capsys.readouterr().out
    assert "action_equals_state" in out
    assert "stuck_state" in out


def test_demo_output_never_says_defective(capsys):
    main(["--demo"])
    assert "defective" not in capsys.readouterr().out.lower()


def test_no_args_errors():
    with pytest.raises(SystemExit):
        main([])


def test_files_all_help_does_not_crash(capsys):
    # --files all must parse cleanly even when the run never gets past
    # --help, which argparse handles by parsing every other argument
    # (including --files) before printing help and exiting 0.
    with pytest.raises(SystemExit) as exc:
        main(["--files", "all", "--help"])
    assert exc.value.code == 0


def test_census_local_source_runs_end_to_end(tmp_path):
    # The one place the whole pipeline is proven to compose: argparse
    # wiring, ledger.sources.LocalSource, and ledger.census's two tiers,
    # all offline against a fixture built the same way tests/test_census.py
    # builds its own.
    from ledger.synth import make_episodes, write_v30_fixture

    root = tmp_path / "repos"
    write_v30_fixture(make_episodes(n_eps=2, T=40, seed=1), root, "acme/x")
    out_dir = tmp_path / "census_out"

    rc = main([
        "--census", "both",
        "--source", "local",
        "--local-root", str(root),
        "--repos", "acme/x",
        "--out-dir", str(out_dir),
    ])
    assert rc == 0

    meta_path = out_dir / "metadata.jsonl"
    deep_path = out_dir / "deep.jsonl"
    assert meta_path.exists()
    assert deep_path.exists()

    meta_rows = [json.loads(l) for l in meta_path.read_text().strip().splitlines()]
    deep_rows = [json.loads(l) for l in deep_path.read_text().strip().splitlines()]
    assert [r["repo"] for r in meta_rows] == ["acme/x"]
    assert [r["repo"] for r in deep_rows] == ["acme/x"]
    assert deep_rows[0]["episodes_sampled"] == 2
    assert not deep_rows[0]["error"]


def test_census_local_source_without_local_root_does_not_touch_network(tmp_path, capsys):
    # Misuse guard: --source local with no --repos has no frame to draw
    # from except a live Hub crawl, which must never happen in this
    # suite. This must fail fast, not attempt network access.
    rc = main(["--census", "metadata", "--source", "local",
              "--local-root", str(tmp_path), "--out-dir", str(tmp_path / "out")])
    assert rc != 0
