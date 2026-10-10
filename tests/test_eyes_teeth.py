import numpy as np

from cammy.effects.eyes_teeth import EyeBrighten, TeethWhiten
from cammy.face import INNER_LIPS, LEFT_EYE, RIGHT_EYE
from conftest import FakeTracker


def with_region(face, indices, center, half=8):
    """Put the given landmark loop on a small square around `center`."""
    cx, cy = center
    corners = [(cx - half, cy - half), (cx + half, cy - half),
               (cx + half, cy + half), (cx - half, cy + half)]
    for i, idx in enumerate(indices):
        face.points[idx] = corners[i * 4 // len(indices)]
    return face


def test_eye_brighten_lifts_light_pixels_only(face, tracker):
    with_region(face, RIGHT_EYE, (130, 100))
    with_region(face, LEFT_EYE, (190, 100))
    frame = np.full(face.shape, 20, np.uint8)
    frame[92:108, 122:138] = 170                      # sclera-ish patch
    out = EyeBrighten(tracker, amount=0.6)(frame)
    assert out[100, 130].mean() > frame[100, 130].mean() + 20
    assert (out[10, 10] == frame[10, 10]).all()       # dark / outside untouched
    assert (out[100, 190] == frame[100, 190]).all()   # dark pixel in eye stays dark


def test_teeth_whiten_reduces_yellow(face, tracker):
    with_region(face, INNER_LIPS, (160, 160))
    frame = np.zeros(face.shape, np.uint8)
    frame[:] = (150, 200, 220)                         # BGR, yellowish-light teeth
    out = TeethWhiten(tracker, amount=1.0)(frame)
    # yellow = R/G high relative to B; whitening raises blue relative to red
    before = int(frame[160, 160, 2]) - int(frame[160, 160, 0])
    after = int(out[160, 160, 2]) - int(out[160, 160, 0])
    assert after < before
    assert (out[10, 10] == frame[10, 10]).all()


def test_identity_cases(face, tracker):
    frame = np.full(face.shape, 128, np.uint8)
    assert EyeBrighten(tracker, amount=0)(frame) is frame
    assert TeethWhiten(tracker, amount=0)(frame) is frame
    assert EyeBrighten(FakeTracker(None))(frame) is frame
    assert TeethWhiten(FakeTracker(None))(frame) is frame
