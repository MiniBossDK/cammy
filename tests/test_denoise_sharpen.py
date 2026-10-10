import numpy as np

from cammy.effects.denoise import Denoise
from cammy.effects.sharpen import Sharpen


def noisy(seed=0):
    rng = np.random.default_rng(seed)
    base = np.full((64, 64, 3), 128, np.int16)
    return np.clip(base + rng.normal(0, 15, base.shape), 0, 255).astype(np.uint8)


def test_denoise_reduces_variance():
    frame = noisy()
    assert Denoise()(frame).std() < frame.std()


def test_sharpen_zero_is_identity():
    frame = noisy()
    assert (Sharpen(amount=0)(frame) == frame).all()


def test_sharpen_increases_edge_contrast():
    frame = np.zeros((32, 32, 3), np.uint8)
    frame[:, 16:] = 200
    soft = Sharpen(amount=0)(frame)
    sharp = Sharpen(amount=1.0)(frame)
    assert sharp.astype(int).std() >= soft.astype(int).std()
