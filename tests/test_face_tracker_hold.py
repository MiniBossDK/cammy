from types import SimpleNamespace

import numpy as np
import pytest

import cammy.face as face_module
from cammy.face import FaceTracker
from cammy.presence import Presence


class Landmarker:
    """Stand-in for the MediaPipe landmarker: returns a face or nothing on demand."""

    def __init__(self):
        self.has_face = True

    def detect_for_video(self, image, ts):
        if not self.has_face:
            return SimpleNamespace(face_landmarks=[])
        lm = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
        return SimpleNamespace(face_landmarks=[lm])


@pytest.fixture
def tracker(monkeypatch):
    clock = SimpleNamespace(t=100.0)
    monkeypatch.setattr(face_module.time, "perf_counter", lambda: clock.t)
    t = FaceTracker.__new__(FaceTracker)
    t._landmarker, t.detect_width, t._t0 = Landmarker(), 640, 100.0
    t.face, t.presence, t._presence, t._last_face = None, 0.0, Presence(), None
    t._detections = []
    t.clock = clock
    return t


def step(tracker, dt, has_face):
    tracker.clock.t += dt
    tracker._landmarker.has_face = has_face
    return tracker.update(np.zeros((48, 64, 3), np.uint8))


def test_keeps_last_face_through_a_brief_loss(tracker):
    for _ in range(5):
        step(tracker, 0.05, True)
    assert tracker.presence == 1.0
    lost = step(tracker, 0.1, False)
    assert lost is not None and tracker.presence == 1.0       # held, not dropped


def test_fades_then_drops_the_face_after_a_long_loss(tracker):
    for _ in range(5):
        step(tracker, 0.05, True)
    seen = []
    for _ in range(20):
        step(tracker, 0.1, False)
        seen.append((tracker.presence, tracker.face is not None))
    presences = [p for p, _ in seen]
    assert presences == sorted(presences, reverse=True)       # only ever fades down
    assert any(0 < p < 1 for p in presences)                  # passes through partial values
    assert seen[-1] == (0.0, False)                           # finally gone


def test_face_returns_after_loss(tracker):
    for _ in range(5):
        step(tracker, 0.05, True)
    for _ in range(20):
        step(tracker, 0.1, False)
    for _ in range(4):
        step(tracker, 0.05, True)
    assert tracker.face is not None and tracker.presence == 1.0


class MovingLandmarker:
    """Face centre moves 0.05 of the frame width per detection."""

    def __init__(self):
        self.n = 0

    def detect_for_video(self, image, ts):
        x = 0.2 + 0.05 * self.n
        self.n += 1
        return SimpleNamespace(face_landmarks=[[SimpleNamespace(x=x, y=0.5) for _ in range(478)]])


def moving_tracker(tracker):
    tracker._landmarker = MovingLandmarker()
    return tracker


def test_predict_follows_the_faces_velocity(tracker):
    moving_tracker(tracker)
    step(tracker, 0.1, True)                   # x = 12.8 px
    step(tracker, 0.1, True)                   # x = 16.0 px -> 32 px/s
    tracker.clock.t += 0.05                    # a skipped frame, 50 ms after the last detection
    face = tracker.predict()
    assert face.points[0, 0] == pytest.approx(16.0 + 32 * 0.05, abs=0.01)


def test_predict_needs_two_recent_detections(tracker):
    moving_tracker(tracker)
    step(tracker, 0.1, True)
    before = tracker.face
    assert tracker.predict() is before         # only one detection: no velocity yet


def test_predict_does_not_extrapolate_stale_detections(tracker):
    moving_tracker(tracker)
    step(tracker, 0.1, True)
    step(tracker, 0.1, True)
    detected = tracker.face
    tracker.clock.t += 0.5                     # far too long since the last detection
    assert tracker.predict() is detected


def test_predict_after_a_lost_face_does_nothing(tracker):
    moving_tracker(tracker)
    step(tracker, 0.1, True)
    step(tracker, 0.1, True)
    step(tracker, 0.05, False)                 # face lost: history is dropped
    held = tracker.face
    assert tracker.predict() is held


def test_predicted_shift_is_capped(tracker):
    class Jumpy:
        n = 0

        def detect_for_video(self, image, ts):
            x = 0.0 if self.n == 0 else 1.0     # leaps across the whole frame
            self.n += 1
            return SimpleNamespace(face_landmarks=[[SimpleNamespace(x=x, y=0.5) for _ in range(478)]])

    tracker._landmarker = Jumpy()
    step(tracker, 0.1, True)
    step(tracker, 0.1, True)
    last_x = tracker.face.points[0, 0]
    tracker.clock.t += 0.1
    assert tracker.predict().points[0, 0] - last_x <= tracker.MAX_SHIFT + 1e-3


def test_reset_forgets_the_face_and_its_history(tracker):
    moving_tracker(tracker)
    step(tracker, 0.1, True)
    step(tracker, 0.1, True)
    assert tracker.face is not None and tracker.presence > 0
    tracker.reset()
    assert tracker.face is None and tracker.presence == 0.0 and tracker._detections == []
    assert tracker.predict() is None
