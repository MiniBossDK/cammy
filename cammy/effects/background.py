"""Background blur and replacement, driven by the person segmentation mask."""
import cv2
import numpy as np


def _composite(frame, background, mask):
    """frame where mask is 1, background where it is 0 (uint8 images, float mask)."""
    mask = np.ascontiguousarray(mask, np.float32)
    return cv2.blendLinear(frame, background, mask, 1.0 - mask)


class BackgroundBlur:
    """Blur everything except the person.

    strength is the blur sigma as a fraction of frame width (0.02 is a soft
    blur, 0.05 is heavy). The blur runs on a reduced copy and is scaled back
    up. The copy is as small as the blur allows (about 4 px of sigma at the
    reduced size), so a heavy blur costs no more than a light one; pass
    `scale` to force a fixed reduction instead.
    """

    TARGET_SIGMA = 4.0   # sigma, in pixels, at the reduced size
    MIN_SIDE = 8         # never reduce below this many pixels on the short side

    def __init__(self, segmenter, strength=0.03, scale=None):
        self.segmenter = segmenter
        self.strength = strength
        self.scale = scale

    def _scale_for(self, shape):
        h, w = shape[:2]
        if self.scale is not None:
            return self.scale
        sigma_full = self.strength * w
        scale = min(0.25, self.TARGET_SIGMA / max(sigma_full, 1e-6))
        return max(scale, self.MIN_SIDE / min(h, w))

    def __call__(self, frame):
        mask = self.segmenter.mask
        if mask is None or self.strength <= 0:
            return frame
        h, w = frame.shape[:2]
        scale = self._scale_for(frame.shape)
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), max(self.strength * w * scale, 0.1))
        blurred = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return _composite(frame, blurred, mask)


class BackgroundReplace:
    """Replace everything except the person with an image or a solid colour.

    `background` is a BGR tuple like (0, 255, 0), or an image (ndarray). Images
    are scaled to cover the frame and centre-cropped, so aspect ratio is kept.
    """

    def __init__(self, segmenter, background):
        self.segmenter = segmenter
        self.background = background
        self._cache = None  # (frame shape, uint8 background)

    def _fit(self, shape):
        h, w = shape[:2]
        if self._cache is not None and self._cache[0] == (h, w):
            return self._cache[1]
        bg = self.background
        if isinstance(bg, np.ndarray):
            scale = max(w / bg.shape[1], h / bg.shape[0])
            bg = cv2.resize(bg, (round(bg.shape[1] * scale), round(bg.shape[0] * scale)),
                            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            y0, x0 = (bg.shape[0] - h) // 2, (bg.shape[1] - w) // 2
            fitted = np.ascontiguousarray(bg[y0:y0 + h, x0:x0 + w])
        else:
            fitted = np.empty((h, w, 3), np.uint8)
            fitted[:] = bg
        self._cache = ((h, w), fitted)
        return fitted

    def __call__(self, frame):
        mask = self.segmenter.mask
        if mask is None:
            return frame
        return _composite(frame, self._fit(frame.shape), mask)
