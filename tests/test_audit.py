from ledger.audit import _synthetic, audit_frame, summarise


def _flags(defect):
    df = _synthetic(defect=defect)
    return summarise("t", {"fps": 30}, [audit_frame(df, 30.0)]).flags


def test_clean_has_no_flags():
    assert _flags("") == ""


def test_each_defect_is_named():
    assert "action_equals_state" in _flags("identity")
    assert "bad_dt" in _flags("drops")
    assert "stuck_state" in _flags("stuck")
    assert "negative_lag" in _flags("swapped")
    assert "duplicate_episodes" in _flags("duplicate")
