import numpy as np
import pytest

from cammy.face import FACE_OVAL, Face


class FakeTracker:
    def __init__(self, face):
        self.face = face


def synthetic_face(shape=(240, 320, 3), center=(160, 120), radius=60):
    """A Face whose 478 landmarks all sit on/inside a circle; oval indices on the rim."""
    cx, cy = center
    pts = np.tile(np.array([cx, cy], np.float32), (478, 1))
    for i, idx in enumerate(FACE_OVAL):
        a = 2 * np.pi * i / len(FACE_OVAL)
        pts[idx] = (cx + radius * np.cos(a), cy + radius * np.sin(a))
    return Face(pts, shape)


@pytest.fixture
def face():
    return synthetic_face()


@pytest.fixture
def tracker(face):
    return FakeTracker(face)
