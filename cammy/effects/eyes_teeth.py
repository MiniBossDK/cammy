"""Eye brightening and teeth whitening."""
import cv2
import numpy as np

from ..face import LEFT_EYE, RIGHT_EYE


def _smoothstep(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class EyeBrighten:
    """Lighten the bright parts (sclera) of the eyes; pupils and lashes stay dark.

    amount 0 is identity, 1 pulls sclera pixels fully to white.

    Two checks keep it from painting white blobs when the eyes are not really
    visible (hand over the face, glare on glasses):
      * sclera is nearly colourless, so saturated pixels such as skin are
        left alone (`saturation` = (start, full) on a 0-255 scale);
      * sclera is only part of an eye, so if more than `coverage` of an eye's
        area would be lifted, that eye is treated as covered and fades out.
        Each eye is judged on its own.
    """

    def __init__(self, tracker, amount=0.3, threshold=(80, 160),
                 saturation=(35, 65), coverage=(0.35, 0.6)):
        self.tracker = tracker
        self.amount = amount
        self.threshold = threshold
        self.saturation = saturation
        self.coverage = coverage

    def __call__(self, frame):
        face = self.tracker.face
        if face is None or self.amount <= 0:
            return frame
        mask, (x0, y0, x1, y1) = face.eye_mask()
        if x1 <= x0 or y1 <= y0:
            return frame
        roi = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sat = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
        candidate = mask * _smoothstep(gray, *self.threshold) * (1.0 - _smoothstep(sat, *self.saturation))

        # Judge each eye separately: split the box midway between the two eyes.
        centres = [face.pts(loop)[:, 0].mean() for loop in (RIGHT_EYE, LEFT_EYE)]
        split = int(np.clip((centres[0] + centres[1]) / 2 - x0, 0, x1 - x0))
        weight = candidate * self.amount
        for sl in (slice(0, split), slice(split, x1 - x0)):
            area = mask[:, sl].sum()
            if area <= 0:
                continue
            share = candidate[:, sl].sum() / area
            weight[:, sl] *= 1.0 - _smoothstep(share, *self.coverage)

        lifted = roi.astype(np.float32)
        lifted += (255.0 - lifted) * weight[..., None]
        out = frame.copy()
        out[y0:y1, x0:x1] = np.clip(lifted, 0, 255).astype(np.uint8)
        return out


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
        mask, (x0, y0, x1, y1) = face.mouth_mask()
        if x1 <= x0 or y1 <= y0:
            return frame
        roi = frame[y0:y1, x0:x1]
        # OpenCV 8-bit LAB: L in [0, 255], a/b centered on 128.
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        weight = mask * _smoothstep(lab[..., 0], *self.threshold) * self.amount
        lab[..., 2] -= (lab[..., 2] - 128.0) * weight          # pull yellow toward neutral
        lab[..., 0] += self.lift * weight                      # slight lift
        white = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        out = frame.copy()
        touched = (weight > 0)[..., None]
        out[y0:y1, x0:x1] = np.where(touched, white, roi)      # untouched pixels stay exact
        return out
