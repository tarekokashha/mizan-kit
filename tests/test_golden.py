"""Golden regression test for the demo's CSV output.

Pins today's numbers in tests/golden/demo_report.csv so that a future
threshold change (ledger/thresholds.toml) or a change to the checks
themselves surfaces as a reviewable diff in that file, rather than as a
silent shift nothing catches. If this test ever fails because a change
was intentional, regenerate the golden file with:

    ./.venv/Scripts/python.exe -m ledger.audit --demo --out tests/golden/demo_report.csv

and review the diff before committing it.

The comparison normalises line endings before comparing text, rather
than comparing bytes. ledger.report.write_csv opens its output with
newline="" (required by the csv module) and writes the PROVISIONAL
header line itself with a bare "\n", while csv.DictWriter's own rows
use its default "\r\n" line terminator; the two therefore do not even
agree on line ending style within a single freshly generated file, let
alone against .gitattributes' `* text=auto eol=lf`, which normalises
whatever this file's line endings were at commit time to LF in the
repository. None of that is an observable difference in the report's
actual content, so normalising both sides to "\n" before comparing is
the correct, robust check, not a weakened one.
"""

from pathlib import Path

from ledger.audit import demo

GOLDEN = Path(__file__).parent / "golden" / "demo_report.csv"


def _normalised(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def test_golden_file_exists():
    assert GOLDEN.exists(), f"missing {GOLDEN}; see this module's docstring to regenerate it"


def test_demo_reproduces_the_golden_report(tmp_path):
    out = tmp_path / "demo_report.csv"
    demo(str(out))
    actual = _normalised(out.read_text(encoding="utf-8"))
    expected = _normalised(GOLDEN.read_text(encoding="utf-8"))
    assert actual == expected


def test_golden_report_never_says_defective():
    assert "defective" not in GOLDEN.read_text(encoding="utf-8").lower()
