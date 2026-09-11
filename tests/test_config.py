import pytest

from ledger.config import load_thresholds


def test_defaults_match_v0_values():
    t = load_thresholds()
    assert t["frac_bad_dt"] == 0.05
    assert t["stuck_state_frac"] == 0.20
    assert t["identity_frac"] == 0.50
    assert t["lag_large"] == 3
    assert t["dup_episode_frac"] == 0.0


def test_rejects_out_of_range_fraction(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text(
        "[thresholds]\nfrac_bad_dt = 1.5\nstuck_state_frac = 0.2\n"
        "identity_frac = 0.5\nlag_large = 3\ndup_episode_frac = 0.0\n"
        "min_lag_correlation = 0.5\n"
    )
    with pytest.raises(ValueError, match="frac_bad_dt"):
        load_thresholds(p)


def test_rejects_missing_key(tmp_path):
    p = tmp_path / "short.toml"
    p.write_text("[thresholds]\nfrac_bad_dt = 0.05\n")
    with pytest.raises(ValueError, match="stuck_state_frac"):
        load_thresholds(p)
