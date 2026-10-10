import numpy as np

from cammy.effects.temporal_denoise import TemporalDenoise


def noisy_sequence(n=40, shape=(72, 96, 3), base=100, sigma=6, seed=0):
    rng = np.random.default_rng(seed)
    return [np.clip(base + rng.normal(0, sigma, shape), 0, 255).astype(np.uint8) for _ in range(n)]


def test_first_frame_and_zero_strength_pass_through():
    seq = noisy_sequence(2)
    fx = TemporalDenoise(strength=1.0)
    assert fx(seq[0]) is seq[0]
    off = TemporalDenoise(strength=0)
    assert off(seq[0]) is seq[0] and off(seq[1]) is seq[1]


def test_reduces_noise_on_a_static_scene():
    seq = noisy_sequence(40)
    fx = TemporalDenoise(strength=1.0)
    outs = [fx(f) for f in seq]
    raw_flicker = np.stack(seq[20:]).astype(float).std(axis=0).mean()
    out_flicker = np.stack(outs[20:]).astype(float).std(axis=0).mean()
    assert out_flicker < raw_flicker * 0.5


def test_does_not_drift_the_mean_level():
    seq = noisy_sequence(60, base=100)
    fx = TemporalDenoise(strength=1.0)
    outs = [fx(f) for f in seq]
    assert abs(outs[-1].astype(float).mean() - 100) < 1.0


def test_sudden_change_is_not_ghosted():
    still = [np.full((72, 96, 3), 50, np.uint8) for _ in range(10)]
    fx = TemporalDenoise(strength=1.0)
    for f in still:
        fx(f)
    moved = still[0].copy()
    moved[20:50, 30:70] = 200                        # an object appears
    out = fx(moved)
    assert abs(int(out[35, 50, 0]) - 200) <= 2        # inside it: current frame wins
    assert abs(int(out[5, 5, 0]) - 50) <= 2           # untouched background


def test_moving_object_leaves_no_trail():
    fx = TemporalDenoise(strength=1.0)
    bg = 50
    for step in range(12):
        f = np.full((72, 128, 3), bg, np.uint8)
        x = 10 + step * 8
        f[25:45, x:x + 16] = 220
        out = fx(f)
    # Where the object was 3 frames ago (24 px behind) the output is background again.
    behind = 10 + 11 * 8 - 24
    assert int(out[35, behind + 4, 0]) < bg + 15


def test_noise_reduction_does_not_blur_static_edges():
    rng = np.random.default_rng(1)
    base = np.zeros((72, 96, 3), np.float32)
    base[:, 48:] = 180
    fx = TemporalDenoise(strength=1.0)
    for _ in range(30):
        out = fx(np.clip(base + rng.normal(0, 5, base.shape), 0, 255).astype(np.uint8))
    assert out[:, 40].mean() < 15 and out[:, 56].mean() > 165      # edge still sharp


def test_resets_when_frame_size_changes():
    fx = TemporalDenoise(strength=1.0)
    fx(np.full((40, 40, 3), 10, np.uint8))
    other = np.full((60, 80, 3), 90, np.uint8)
    assert fx(other) is other


def test_reset_forgets_history():
    fx = TemporalDenoise(strength=1.0)
    fx(np.full((40, 40, 3), 10, np.uint8))
    fx.reset()
    f = np.full((40, 40, 3), 90, np.uint8)
    assert fx(f) is f
