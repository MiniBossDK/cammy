"""Gray-world automatic white balance."""
import numpy as np


class WhiteBalance:
    """Scale each channel so the channel means match the overall mean.

    strength blends between the original (0.0) and fully corrected (1.0) frame.
    """

    def __init__(self, strength=1.0):
        self.strength = strength

    def __call__(self, frame):
        means = frame.reshape(-1, 3).mean(axis=0)
        gains = means.mean() / np.maximum(means, 1e-6)
        gains = 1.0 + self.strength * (gains - 1.0)
        return np.clip(frame * gains, 0, 255).astype(np.uint8)
