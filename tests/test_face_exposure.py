import numpy as np

from cammy.effects.face_exposure import FaceExposure
from conftest import FakeTracker


def dark_scene(face):
    frame = np.full(face.shape, 230, np.uint8)       # bright backdrop
    frame[face.face_mask() > 0.5] = 40               # dark face
    return frame


def luma(frame):
    return frame.mean(axis=2)


def test_lifts_dark_face(face, tracker):
    frame = dark_scene(face)
    fx = FaceExposure(tracker, smoothing=0.0)
    out = fx(frame)
    inside = face.face_mask() > 0.5
    assert luma(out)[inside].mean() > luma(frame)[inside].mean() + 30


def test_does_not_clip_backdrop(face, tracker):
    frame = dark_scene(face)
    out = FaceExposure(tracker, smoothing=0.0)(frame)
    outside = face.face_mask() < 0.01
    assert luma(out)[outside].max() <= 255
    assert luma(out)[outside].mean() >= luma(frame)[outside].mean() - 1


def test_no_face_is_identity():
    frame = np.random.randint(0, 256, (20, 20, 3), np.uint8)
    assert FaceExposure(FakeTracker(None))(frame) is frame


def test_bright_face_unchanged(face, tracker):
    frame = np.full(face.shape, 200, np.uint8)
    out = FaceExposure(tracker, smoothing=0.0)(frame)
    assert (out == frame).all()


def test_keeps_the_lift_when_the_face_is_lost(face, tracker):
    frame = dark_scene(face)
    fx = FaceExposure(tracker, smoothing=0.0)
    lifted = fx(frame)
    tracker.face = None                       # face covered or turned away
    held = fx(frame)
    assert (held == lifted).all()             # no jump back to the dark original


def test_no_lift_before_any_face_was_seen(face, tracker):
    frame = dark_scene(face)
    tracker.face = None
    assert FaceExposure(tracker)(frame) is frame
