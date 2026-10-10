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


def eyes_frame(face, right_centre=(130, 100), left_centre=(190, 100)):
    with_region(face, RIGHT_EYE, right_centre)
    with_region(face, LEFT_EYE, left_centre)
    return np.full(face.shape, 20, np.uint8)               # dark eyes: iris and lashes


def test_eye_brighten_lifts_light_pixels_only(face, tracker):
    frame = eyes_frame(face)
    frame[92:108, 122:126] = 170                      # a sliver of sclera (about a quarter)
    out = EyeBrighten(tracker, amount=0.6)(frame)
    assert out[100, 124].mean() > frame[100, 124].mean() + 20
    assert (out[10, 10] == frame[10, 10]).all()       # outside untouched
    assert (out[100, 190] == frame[100, 190]).all()   # dark pixel in the other eye stays dark


def test_eye_brighten_ignores_saturated_skin_tones(face, tracker):
    frame = eyes_frame(face)
    frame[92:108, 122:126] = (110, 150, 200)          # bright but strongly coloured: skin, not sclera
    out = EyeBrighten(tracker, amount=1.0)(frame)
    assert (out == frame).all()


def test_eye_brighten_backs_off_when_the_whole_eye_is_bright(face, tracker):
    frame = eyes_frame(face)
    frame[92:108, 122:138] = 170                      # right eye entirely bright: hand, not an eye
    out = EyeBrighten(tracker, amount=1.0)(frame)
    assert (out == frame).all()


def test_eye_brighten_judges_each_eye_on_its_own(face, tracker):
    frame = eyes_frame(face)
    frame[92:108, 122:138] = 170                      # right eye covered
    frame[92:108, 182:186] = 170                      # left eye: normal sliver of sclera
    out = EyeBrighten(tracker, amount=1.0)(frame)
    assert (out[92:108, 122:138] == frame[92:108, 122:138]).all()
    assert out[100, 184].mean() > frame[100, 184].mean() + 20


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
