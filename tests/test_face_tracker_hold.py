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
