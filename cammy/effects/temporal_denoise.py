"""Motion-adaptive temporal noise reduction."""
import cv2
import numpy as np


def _smoothstep(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class TemporalDenoise:
    """Blend each frame with the previous output, except where things move.

    A recursive average over time: static pixels converge toward their true
    value and sensor noise cancels out, while the blend weight drops to zero
    where the picture changed, so moving areas are not smeared into ghosts.

    The motion test runs on quarter-size copies, so single noisy pixels do not
    count as motion. strength 0 is identity; 1 allows up to ~10 frames of
    averaging in still areas.

    Run it first, before any brightening, so the noise it removes is not
    multiplied by later effects.
    """

    MAX_WEIGHT = 0.9

    def __init__(self, strength=0.7, still=2.5, moving=9.0, scale=0.25):
        self.strength = strength
        self.still = still       # diff (levels, at reduced size) below which a pixel counts as still
        self.moving = moving     # diff above which the previous frame is ignored completely
        self.scale = scale
        self._acc = None         # previous output, float32
        self._small = None       # previous output at reduced size, uint8

    def reset(self):
        self._acc = self._small = None

    def _reduce(self, image):
        return cv2.resize(image, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

    def __call__(self, frame):
        if self.strength <= 0:
            self.reset()
            return frame
        cur = frame.astype(np.float32)
        small = self._reduce(frame)
        if self._acc is None or self._acc.shape != cur.shape:
            self._acc, self._small = cur, small
            return frame

        diff = cv2.absdiff(small, self._small).max(axis=2).astype(np.float32)
        diff = cv2.blur(diff, (3, 3))
        weight = self.MAX_WEIGHT * self.strength * (1.0 - _smoothstep(diff, self.still, self.moving))
        # Grow motion areas by a pixel so a moving edge does not leave a ghost.
        weight = cv2.erode(weight, np.ones((3, 3), np.uint8))
        weight = cv2.resize(weight, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

        self._acc = cv2.blendLinear(cur, self._acc, 1.0 - weight, weight)
        out = cv2.convertScaleAbs(self._acc)
        self._small = self._reduce(out)
        return out
