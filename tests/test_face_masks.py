import numpy as np

from cammy.face import INNER_LIPS, RIGHT_EYE
from test_eyes_teeth import with_region


def test_boxes_stay_inside_frame(face):
    h, w = face.shape[:2]
    for mask, (x0, y0, x1, y1) in (face.skin_mask(), face.eye_mask(), face.mouth_mask()):
        assert 0 <= x0 <= x1 <= w and 0 <= y0 <= y1 <= h
        assert mask.shape == (y1 - y0, x1 - x0)
        assert mask.min() >= 0.0 and mask.max() <= 1.0


def test_skin_mask_excludes_eye_but_keeps_cheek(face):
    with_region(face, RIGHT_EYE, (150, 100), half=6)
    mask, (x0, y0, _, _) = face.skin_mask(feather=1)
    assert mask[100 - y0, 150 - x0] < 0.2      # eye centre carved out
    assert mask[140 - y0, 160 - x0] > 0.9      # cheek kept


def test_mouth_mask_covers_mouth_only(face):
    with_region(face, INNER_LIPS, (160, 160), half=10)
    mask, (x0, y0, _, _) = face.mouth_mask(feather=1)
    assert mask[160 - y0, 160 - x0] > 0.9
    assert mask.shape[0] < 60 and mask.shape[1] < 60
