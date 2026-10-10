import numpy as np

from cammy.segment import smooth_probability, tighten

EDGE = (0.3, 0.7)


def step(prev, raw, smoothing=0.3):
    prob = smooth_probability(prev, raw, smoothing)
    return prob, tighten(prob, EDGE)


def test_vacated_pixels_drop_to_background_immediately():
    person = np.ones((4, 4), np.float32)
    gone = np.zeros((4, 4), np.float32)
    prob, _ = step(None, person)
    prob, mask = step(prob, gone)             # the person left this frame
    assert (mask == 0).all()                  # no half-strength trail


def test_newly_covered_pixels_appear_immediately():
    prob, _ = step(None, np.zeros((4, 4), np.float32))
    prob, mask = step(prob, np.ones((4, 4), np.float32))
    assert (mask == 1).all()


def test_old_order_would_leave_a_trail():
    """Documents why tightening must come after smoothing."""
    old_mask = tighten(np.ones((2, 2), np.float32), EDGE)                      # tightened first: 1.0
    after_leaving = 0.5 * old_mask + 0.5 * tighten(np.zeros((2, 2), np.float32), EDGE)
    assert after_leaving.max() == 0.5                                           # lingering ghost


def test_borderline_flicker_is_damped():
    rng = np.random.default_rng(0)
    prev, masks = None, []
    for _ in range(60):
        raw = (0.5 + rng.normal(0, 0.15, (8, 8))).clip(0, 1).astype(np.float32)   # noisy edge pixel
        prev, m = step(prev, raw, smoothing=0.3)
        masks.append(m)
    raw_masks = [tighten((0.5 + rng.normal(0, 0.15, (8, 8))).clip(0, 1).astype(np.float32), EDGE) for _ in range(60)]
    assert np.abs(np.diff(masks, axis=0)).mean() < np.abs(np.diff(raw_masks, axis=0)).mean()


def test_resizes_reset_the_history():
    prob, _ = step(None, np.ones((4, 4), np.float32))
    prob, _ = step(prob, np.zeros((6, 6), np.float32))
    assert prob.shape == (6, 6) and prob.max() == 0
