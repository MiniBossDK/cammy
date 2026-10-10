import numpy as np

from cammy.effects.brightness_contrast import BrightnessContrast


def test_identity():
    frame = np.random.randint(0, 256, (8, 8, 3), np.uint8)
    assert (BrightnessContrast()(frame) == frame).all()


def test_brightness_adds_and_clips():
    frame = np.full((2, 2, 3), 250, np.uint8)
    assert (BrightnessContrast(brightness=20)(frame) == 255).all()


def test_contrast_scales():
    frame = np.full((2, 2, 3), 100, np.uint8)
    assert (BrightnessContrast(contrast=1.5)(frame) == 150).all()


def test_negative_brightness_clips_shadows_to_black():
    frame = np.array([[[0, 5, 10], [20, 27, 40]]], np.uint8)
    out = BrightnessContrast(brightness=-27)(frame)
    assert out.tolist() == [[[0, 0, 0], [0, 0, 13]]]        # not mirrored up to 27, 22, 17...


def test_dark_pixels_stay_ordered_and_never_brighten_with_negative_brightness():
    frame = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, axis=2)
    out = BrightnessContrast(contrast=1.29, brightness=-27.3)(frame)
    values = out[..., 0].ravel().astype(int)
    assert (np.diff(values) >= 0).all()                      # monotonic: no folding
    assert values[0] == 0 and values[-1] == 255


def test_lut_updates_when_parameters_change():
    fx = BrightnessContrast()
    frame = np.full((2, 2, 3), 100, np.uint8)
    assert (fx(frame) == 100).all()
    fx.brightness = 30
    assert (fx(frame) == 130).all()
    fx.contrast, fx.brightness = 0.5, 0
    assert (fx(frame) == 50).all()
