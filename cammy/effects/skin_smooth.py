"""Skin smoothing limited to the face."""
import cv2
import numpy as np


class SkinSmooth:
    """Bilateral-filter the face and blend it back through a soft skin mask.

    The mask excludes eyes, brows and lips, so those stay sharp. strength 0 is
    identity, 1 is the full filter result. Only the face box is filtered, at half
    resolution, which keeps the cost low.
    """

    def __init__(self, tracker, strength=0.6, diameter=5, sigma_color=40, sigma_space=5):
        self.tracker = tracker
        self.strength = strength
        self.diameter = diameter
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def __call__(self, frame):
        face = self.tracker.face
        if face is None or self.strength <= 0:
            return frame
        mask, (x0, y0, x1, y1) = face.skin_mask()
        if x1 <= x0 or y1 <= y0:
            return frame
        roi = frame[y0:y1, x0:x1]
        # Filter at half resolution: smoothing is low-frequency work, ~4x cheaper.
        half = cv2.resize(roi, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        half = cv2.bilateralFilter(half, self.diameter, self.sigma_color, self.sigma_space)
        smooth = cv2.resize(half, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)
        alpha = (mask * self.strength)[..., None]
        out = frame.copy()
        out[y0:y1, x0:x1] = (roi * (1 - alpha) + smooth * alpha).astype(np.uint8)
        return out
