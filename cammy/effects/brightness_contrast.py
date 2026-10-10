"""Brightness and contrast adjustment."""
import cv2
import numpy as np

_LEVELS = np.arange(256, dtype=np.float32)


class BrightnessContrast:
    """out = clip(frame * contrast + brightness, 0, 255). contrast 1.0 / brightness 0 is identity.

    Applied through a 256-entry lookup table. (cv2.convertScaleAbs is not
    used: it takes the absolute value, so negative results mirror back up
    instead of clipping to black, which scrambles shadows.)
    """

    def __init__(self, contrast=1.0, brightness=0):
        self.contrast = contrast
        self.brightness = brightness
        self._lut_key = None
        self._lut = None

    def __call__(self, frame):
        key = (self.contrast, self.brightness)
        if key != self._lut_key:
            self._lut = np.clip(np.rint(_LEVELS * self.contrast + self.brightness), 0, 255).astype(np.uint8)
            self._lut_key = key
        return cv2.LUT(frame, self._lut)
