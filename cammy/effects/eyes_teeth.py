"""Eye brightening and teeth whitening."""
import cv2
import numpy as np


def _smoothstep(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class EyeBrighten:
    """Lighten the bright parts (sclera) of the eyes; pupils and lashes stay dark.

    amount 0 is identity, 1 pulls sclera pixels fully to white.
    """

    def __init__(self, tracker, amount=0.3, threshold=(80, 160)):
        self.tracker = tracker
        self.amount = amount
        self.threshold = threshold

    def __call__(self, frame):
        face = self.tracker.face
        if face is None or self.amount <= 0:
            return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        weight = face.eye_mask() * _smoothstep(gray, *self.threshold) * self.amount
        out = frame.astype(np.float32)
        out += (255.0 - out) * weight[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)


class TeethWhiten:
    """Reduce yellow and lift lightness on the teeth inside the mouth opening.

    Only light pixels inside the inner-lip polygon are touched, so tongue and
    shadow are left alone. amount 0 is identity.
    """

    def __init__(self, tracker, amount=0.5, threshold=(70, 130), lift=12.0):
        self.tracker = tracker
        self.amount = amount
        self.threshold = threshold
        self.lift = lift

    def __call__(self, frame):
        face = self.tracker.face
        if face is None or self.amount <= 0:
            return frame
        mask = face.mouth_mask()
        ys, xs = np.nonzero(mask > 0)
        if len(ys) == 0:
            return frame
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        roi = frame[y0:y1, x0:x1]
        # OpenCV 8-bit LAB: L in [0, 255], a/b centered on 128.
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        weight = mask[y0:y1, x0:x1] * _smoothstep(lab[..., 0], *self.threshold) * self.amount
        lab[..., 2] -= (lab[..., 2] - 128.0) * weight          # pull yellow toward neutral
        lab[..., 0] += self.lift * weight                      # slight lift
        white = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        out = frame.copy()
        touched = (weight > 0)[..., None]
        out[y0:y1, x0:x1] = np.where(touched, white, roi)      # untouched pixels stay exact
        return out
