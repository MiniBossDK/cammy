"""Gray-world automatic white balance."""
import cv2
import numpy as np

_LEVELS = np.arange(256, dtype=np.float32)


class WhiteBalance:
    """Scale each channel so the channel means match the overall mean.

    strength blends between the original (0.0) and fully corrected (1.0) frame.
    Means come from a small thumbnail and gains are applied with a per-channel
    lookup table, which keeps this fast on full-size frames.
    """

    def __init__(self, strength=1.0):
        self.strength = strength

    def __call__(self, frame):
        thumb = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
        means = thumb.reshape(-1, 3).mean(axis=0)
        gains = means.mean() / np.maximum(means, 1e-6)
        gains = 1.0 + self.strength * (gains - 1.0)
        lut = np.clip(_LEVELS[:, None] * gains[None, :], 0, 255).astype(np.uint8)
        return cv2.LUT(frame, lut.reshape(256, 1, 3))
