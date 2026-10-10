"""A face very close to the camera puts landmarks beyond the frame edges."""
import itertools

import numpy as np
import pytest

from cammy.face import Face, clip_box
from cammy.processor import Processor
from cammy.settings import Settings
from conftest import synthetic_face

SHAPE = (240, 320, 3)


def placed(scale, dx, dy):
    base = synthetic_face(SHAPE)
    centre = base.points.mean(axis=0)
    return Face(((base.points - centre) * scale + centre + [dx, dy]).astype(np.float32), SHAPE)


PLACEMENTS = list(itertools.product((0.5, 1.0, 2.0, 4.0, 10.0), (-800, -200, 0, 200, 800), (-600, -150, 0, 150, 600)))


def test_clip_box_is_never_inverted_or_outside():
    for raw in [(-50, -50, 10, 10), (400, 300, 500, 400), (-90, 20, -10, 60), (30, 500, 90, 600), (10, 10, 5, 5)]:
        x0, y0, x1, y1 = clip_box(*raw, 320, 240)
        assert 0 <= x0 <= x1 <= 320 and 0 <= y0 <= y1 <= 240


@pytest.mark.parametrize("scale,dx,dy", PLACEMENTS)
def test_masks_and_boxes_stay_valid_for_any_placement(scale, dx, dy):
    face = placed(scale, dx, dy)
    h, w = SHAPE[:2]
    for mask, (x0, y0, x1, y1) in (face.skin_mask(), face.eye_mask(), face.mouth_mask()):
        assert 0 <= x0 <= x1 <= w and 0 <= y0 <= y1 <= h
        assert mask.shape == (y1 - y0, x1 - x0)
    x0, y0, x1, y1 = face.bbox(0.1)
    assert 0 <= x0 <= x1 <= w and 0 <= y0 <= y1 <= h


class Stub:
    presence = 1.0

    hands = []

    def __init__(self, face=None):
        self.face, self.mask = face, np.full(SHAPE[:2], 0.5, np.float32)

    def update(self, frame, stamp=None):
        pass

    def predict(self):
        pass

    def close(self):
        pass


EVERYTHING = Settings(temporal_denoise=1, white_balance=1, face_exposure=1, denoise=True, skin_smooth=1,
                      eyes=1, teeth=1, contrast=1.2, brightness=-20, sharpen=1, background="blur", bg_blur=0.05)


@pytest.mark.parametrize("scale,dx,dy", PLACEMENTS[::4])
def test_every_effect_survives_any_face_placement(scale, dx, dy):
    stub = Stub(placed(scale, dx, dy))
    p = Processor(EVERYTHING, tracker_factory=lambda: stub, segmenter_factory=lambda: stub, hand_factory=lambda: stub, detect="sequential")
    frame = np.random.default_rng(0).integers(0, 256, SHAPE, dtype=np.uint8)
    out = p.process(frame)
    assert out.shape == frame.shape and out.dtype == np.uint8


def test_feature_wholly_outside_leaves_the_frame_alone():
    from cammy.effects.eyes_teeth import EyeBrighten, TeethWhiten
    face = placed(1.0, 0, -5000)                  # everything far above the frame
    frame = np.full(SHAPE, 120, np.uint8)
    stub = Stub(face)
    assert EyeBrighten(stub, amount=1)(frame) is frame
    assert TeethWhiten(stub, amount=1)(frame) is frame


def test_processing_never_modifies_the_input_frame():
    """Detection threads keep reading the raw frame while effects run, so effects must not write to it."""
    stub = Stub(placed(1.0, 0, 0))
    stub.hands = [np.array([(150 + 10 * i, 100 + 5 * i) for i in range(21)], np.float32)]
    p = Processor(EVERYTHING.replace(background="color", bg_color="#00ff00"), tracker_factory=lambda: stub,
                  segmenter_factory=lambda: stub, hand_factory=lambda: stub, detect="sequential")
    frame = np.random.default_rng(3).integers(0, 256, SHAPE, dtype=np.uint8)
    original = frame.copy()
    for _ in range(3):
        p.process(frame)
    assert (frame == original).all()
