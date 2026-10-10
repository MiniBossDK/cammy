"""Brightness and contrast adjustment."""
import cv2


class BrightnessContrast:
    """out = frame * contrast + brightness. contrast 1.0 / brightness 0 is identity."""

    def __init__(self, contrast=1.0, brightness=0):
        self.contrast = contrast
        self.brightness = brightness

    def __call__(self, frame):
        return cv2.convertScaleAbs(frame, alpha=self.contrast, beta=self.brightness)
