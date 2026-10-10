"""Skin smoothing limited to the face."""
import cv2
import numpy as np


class SkinSmooth:
    """Bilateral-filter the face and blend it back through a soft skin mask.

    The mask excludes eyes, brows and lips, so those stay sharp. strength 0 is
    identity, 1 is the full filter result. Only the padded face box is filtered,
    which keeps the cost low.
    """

    def __init__(self, tracker, strength=0.6, diameter=9, sigma_color=40, sigma_space=9):
        self.tracker = tracker
        self.strength = strength
        self.diameter = diameter
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def __call__(self, frame):
        face = self.tracker.face
        if face is None or self.strength <= 0:
            return frame
        x0, y0, x1, y1 = face.bbox(pad=0.1)
        if x1 <= x0 or y1 <= y0:
            return frame
        roi = frame[y0:y1, x0:x1]
        smooth = cv2.bilateralFilter(roi, self.diameter, self.sigma_color, self.sigma_space)
        alpha = (face.skin_mask()[y0:y1, x0:x1] * self.strength)[..., None]
        out = frame.copy()
        out[y0:y1, x0:x1] = (roi * (1 - alpha) + smooth * alpha).astype(np.uint8)
        return out
