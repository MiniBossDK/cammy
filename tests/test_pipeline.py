import numpy as np

from cammy.pipeline import Pipeline


def test_empty_pipeline_is_identity():
    frame = np.zeros((4, 4, 3), np.uint8)
    assert Pipeline()(frame) is frame


def test_effects_apply_in_order():
    frame = np.zeros((2, 2, 3), np.uint8)
    p = Pipeline([lambda f: f + 1, lambda f: f * 2])
    assert (p(frame) == 2).all()
