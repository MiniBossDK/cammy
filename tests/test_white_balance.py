import numpy as np

from cammy.effects.white_balance import WhiteBalance


def test_removes_color_cast():
    frame = np.zeros((4, 4, 3), np.uint8)
    frame[..., 0] = 60   # B
    frame[..., 1] = 100  # G
    frame[..., 2] = 140  # R
    out = WhiteBalance()(frame)
    means = out.reshape(-1, 3).mean(axis=0)
    assert np.ptp(means) <= 1


def test_zero_strength_is_identity():
    frame = np.random.randint(0, 256, (8, 8, 3), np.uint8)
    assert (WhiteBalance(strength=0.0)(frame) == frame).all()
