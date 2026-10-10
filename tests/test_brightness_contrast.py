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
