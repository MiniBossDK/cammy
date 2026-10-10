import pytest

from cammy.presence import Presence


def run(p, seq):
    """seq: list of (time, detected) -> list of values."""
    return [p.update(d, t) for t, d in seq]


def test_starts_at_zero_and_ramps_up_when_detected():
    p = Presence(fade_in=0.1)
    vals = run(p, [(0.0, True), (0.05, True), (0.1, True), (0.2, True)])
    assert vals[0] == 0.0 and vals[1] == pytest.approx(0.5)
    assert vals[2] == pytest.approx(1.0) and vals[3] == 1.0


def test_holds_through_short_dropouts():
    p = Presence(hold=0.3, fade_in=0.01, fade_out=0.4)
    run(p, [(0.0, True), (0.1, True)])
    vals = run(p, [(0.2, False), (0.3, False), (0.39, False)])
    assert vals == [1.0, 1.0, 1.0]            # within 0.3 s of the last sighting: still held


def test_fades_out_after_hold_and_never_goes_negative():
    p = Presence(hold=0.3, fade_in=0.01, fade_out=0.4)
    run(p, [(0.0, True), (0.1, True)])        # last seen at t=0.1
    v1 = p.update(False, 0.5)                 # hold ended at 0.4, so only 0.1 s of fade counted
    v2 = p.update(False, 0.7)
    assert v1 == pytest.approx(0.75) and v2 == pytest.approx(0.25)
    assert p.update(False, 5.0) == 0.0


def test_comes_back_quickly_but_not_instantly():
    p = Presence(hold=0.0, fade_in=0.1, fade_out=0.2)
    run(p, [(0.0, True), (0.2, True)])
    p.update(False, 0.3); p.update(False, 0.6)
    assert p.value == 0.0
    assert p.update(True, 0.65) == pytest.approx(0.05 / 0.1)           # ramps, does not jump to 1
    assert p.update(True, 0.80) == pytest.approx(1.0)


def test_never_detected_stays_zero():
    p = Presence()
    assert run(p, [(0.0, False), (1.0, False), (2.0, False)]) == [0.0, 0.0, 0.0]


def test_zero_fades_are_immediate():
    p = Presence(hold=0.0, fade_in=0, fade_out=0)
    assert p.update(True, 0.0) == 1.0
    assert p.update(False, 0.1) == 0.0
