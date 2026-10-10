"""Face-aware exposure: lift a dark (e.g. backlit) face without blowing out the rest."""
import cv2
import numpy as np

_LEVELS = np.arange(256, dtype=np.float32) / 255.0


class FaceExposure:
    """Apply a gamma curve to luma so the face reaches `target` brightness.

    Gamma keeps black at 0 and white at 255, so bright areas such as a window
    behind the user are compressed rather than clipped further. The gamma is
    smoothed over time to avoid flicker. Does nothing when no face is found.
    """

    def __init__(self, tracker, target=120, strength=1.0, min_gamma=0.45, smoothing=0.9):
        self.tracker = tracker
        self.target = target
        self.strength = strength
        self.min_gamma = min_gamma
        self.smoothing = smoothing
        self._gamma = 1.0

    def _wanted_gamma(self, luma, face):
        x0, y0, x1, y1 = face.bbox()
        patch = luma[y0:y1, x0:x1]
        mask = face.face_mask()[y0:y1, x0:x1] > 0.5
        if not mask.any():
            return 1.0
        mean = float(patch[mask].mean())
        if mean < 1:
            return self.min_gamma
        # gamma such that (mean/255) ** gamma == target/255
        gamma = np.log(self.target / 255.0) / np.log(mean / 255.0)
        gamma = 1.0 + self.strength * (gamma - 1.0)
        return float(np.clip(gamma, self.min_gamma, 1.0))

    def __call__(self, frame):
        face = self.tracker.face
        if face is None:
            return frame
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        wanted = self._wanted_gamma(ycrcb[..., 0], face)
        self._gamma = self.smoothing * self._gamma + (1 - self.smoothing) * wanted
        if self._gamma > 0.995:
            return frame
        lut = np.clip(255.0 * _LEVELS ** self._gamma, 0, 255).astype(np.uint8)
        ycrcb[..., 0] = cv2.LUT(ycrcb[..., 0], lut)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
