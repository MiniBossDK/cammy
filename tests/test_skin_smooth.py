import numpy as np

from cammy.effects.skin_smooth import SkinSmooth
from conftest import FakeTracker


def noisy_frame(shape, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(128 + rng.normal(0, 12, shape), 0, 255).astype(np.uint8)


def test_smooths_inside_face(face, tracker):
    frame = noisy_frame(face.shape)
    out = SkinSmooth(tracker, strength=1.0)(frame)
    inside = face.face_mask() > 0.9
    assert out[inside].std() < frame[inside].std()


def test_leaves_outside_untouched(face, tracker):
    frame = noisy_frame(face.shape)
    out = SkinSmooth(tracker, strength=1.0)(frame)
    x0, y0, x1, y1 = face.bbox(pad=0.1)
    assert (out[:y0] == frame[:y0]).all()
    assert (out[:, :x0] == frame[:, :x0]).all()


def test_zero_strength_and_no_face_are_identity(face, tracker):
    frame = noisy_frame(face.shape)
    assert SkinSmooth(tracker, strength=0)(frame) is frame
    assert SkinSmooth(FakeTracker(None))(frame) is frame
