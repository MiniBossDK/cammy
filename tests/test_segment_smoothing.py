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


# ---- anti-aliasing the outline ---------------------------------------------------------

from cammy.segment import finish_mask  # noqa: E402


def staircase(h=90, w=160, step=8):
    """A diagonal edge with big, blocky steps, like an enlarged low-resolution mask."""
    mask = np.zeros((h, w), np.float32)
    for y in range(0, h, step):
        mask[y:y + step, : 20 + y] = 1.0
    return mask


def roughness(mask):
    """Total variation along the edge band: staircases have more of it than smooth diagonals."""
    gy, gx = np.gradient(mask)
    return float(np.hypot(gx, gy).sum())


def test_the_outline_is_smoothed_and_stays_a_valid_matte():
    steps = staircase()
    out = finish_mask(steps, (160, 90), edge_blur=1.5)
    assert out.shape == steps.shape and out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert ((out > 0.05) & (out < 0.95)).sum() > ((steps > 0.05) & (steps < 0.95)).sum()      # a soft transition now exists
    assert roughness(out) < roughness(steps)                                                    # sharp corners rounded off


def test_zero_blur_only_scales():
    small = staircase(45, 80, 4)
    out = finish_mask(small, (160, 90), edge_blur=0)
    assert out.shape == (90, 160) and set(np.unique(out.round(2))) <= set(np.unique(cv2_resize(small).round(2)))


def cv2_resize(mask):
    import cv2
    return cv2.resize(mask, (160, 90), interpolation=cv2.INTER_LINEAR)


def test_solid_regions_stay_solid():
    full = np.ones((60, 80), np.float32)
    assert np.allclose(finish_mask(full, (80, 60), 1.5), 1.0, atol=1e-4)
    assert np.allclose(finish_mask(np.zeros((60, 80), np.float32), (80, 60), 1.5), 0.0)


def test_features_at_the_models_resolution_survive_the_blur():
    """The model's mask has about 5-6 px steps at 720p, so that is the thinnest thing it can show."""
    bar = np.zeros((60, 80), np.float32)
    bar[:, 37:43] = 1.0                                     # 6 px wide
    assert finish_mask(bar, (80, 60), 1.5)[30, 40] > 0.9


def test_a_very_thin_bar_loses_some_opacity_at_its_core():
    """The cost of the smoothing, written down: a 4 px bar keeps about 82% at its centre."""
    bar = np.zeros((60, 80), np.float32)
    bar[:, 38:42] = 1.0
    core = finish_mask(bar, (80, 60), 1.5)[30, 40]
    assert 0.75 < core < 0.9
