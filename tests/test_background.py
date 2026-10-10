import numpy as np

from cammy.effects.background import BackgroundBlur, BackgroundReplace


class FakeSegmenter:
    def __init__(self, mask):
        self.mask = mask


def scene(h=120, w=160):
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), np.float32)
    mask[30:90, 50:110] = 1.0
    return frame, mask


def test_blur_keeps_person_and_blurs_background():
    frame, mask = scene()
    out = BackgroundBlur(FakeSegmenter(mask), strength=0.03)(frame)
    person, bg = mask > 0.5, mask < 0.5
    assert (out[person] == frame[person]).all()
    assert out[bg].astype(float).std() < frame[bg].astype(float).std() * 0.5


def test_blur_without_mask_or_strength_is_identity():
    frame, mask = scene()
    assert BackgroundBlur(FakeSegmenter(None))(frame) is frame
    assert BackgroundBlur(FakeSegmenter(mask), strength=0)(frame) is frame


def test_replace_with_colour():
    frame, mask = scene()
    out = BackgroundReplace(FakeSegmenter(mask), (0, 255, 0))(frame)
    person, bg = mask > 0.5, mask < 0.5
    assert (out[person] == frame[person]).all()
    assert (out[bg] == (0, 255, 0)).all()


def test_replace_with_image_covers_frame_and_keeps_aspect():
    frame, mask = scene()
    image = np.zeros((50, 200, 3), np.uint8)       # wide, short image
    image[:, :100] = (255, 0, 0)
    image[:, 100:] = (0, 0, 255)
    out = BackgroundReplace(FakeSegmenter(mask), image)(frame)
    bg = mask < 0.5
    assert out.shape == frame.shape
    left, right = out[5, 5], out[5, -5]
    assert left[0] > 200 and right[2] > 200          # left blue, right red survive the crop
    assert (out[bg][:, 1] == 0).all()


def test_soft_mask_blends():
    frame = np.full((4, 4, 3), 200, np.uint8)
    mask = np.full((4, 4), 0.5, np.float32)
    out = BackgroundReplace(FakeSegmenter(mask), (100, 100, 100))(frame)
    assert (out == 150).all()


def test_replace_follows_frame_size_change():
    mask_a = np.ones((10, 10), np.float32)
    seg = FakeSegmenter(mask_a)
    fx = BackgroundReplace(seg, (1, 2, 3))
    fx(np.zeros((10, 10, 3), np.uint8))
    seg.mask = np.zeros((20, 30), np.float32)
    out = fx(np.zeros((20, 30, 3), np.uint8))
    assert out.shape == (20, 30, 3) and (out == (1, 2, 3)).all()
