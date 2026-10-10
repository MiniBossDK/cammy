"""Face landmark tracking (MediaPipe FaceLandmarker) and region masks."""
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"

# Ordered landmark loops from the MediaPipe 478-point face mesh.
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
             379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
             234, 127, 162, 21, 54, 103, 67, 109]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_BROW = [46, 53, 52, 65, 55, 107, 66, 105, 63, 70]
LEFT_BROW = [276, 283, 282, 295, 285, 336, 296, 334, 293, 300]
OUTER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269,
              267, 0, 37, 39, 40, 185]
INNER_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311,
              312, 13, 82, 81, 80, 191]


def ensure_model(path=MODEL_PATH):
    """Download the landmarker model on first use."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.rename(path)
    return path


def polygon_mask(shape, points, feather=0):
    """Float mask in [0, 1] for a filled polygon, optionally Gaussian-feathered."""
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 255)
    mask = mask.astype(np.float32) / 255.0
    if feather > 0:
        k = int(feather) * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


class Face:
    """One detected face: landmarks in pixel coordinates plus mask helpers."""

    def __init__(self, points, shape):
        self.points = points  # (478, 2) float32, x/y in pixels
        self.shape = shape

    def pts(self, indices):
        return self.points[indices]

    def bbox(self, pad=0.0):
        """(x0, y0, x1, y1) of the face oval, padded by a fraction, clipped to frame."""
        p = self.pts(FACE_OVAL)
        x0, y0 = p.min(axis=0)
        x1, y1 = p.max(axis=0)
        dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
        h, w = self.shape[:2]
        return (int(max(x0 - dx, 0)), int(max(y0 - dy, 0)),
                int(min(x1 + dx, w)), int(min(y1 + dy, h)))

    def face_mask(self, feather=0):
        return polygon_mask(self.shape, self.pts(FACE_OVAL), feather)

    def _local_mask(self, loops, feather, subtract=(), margin=None, pad_loops=None):
        """Draw polygons into a small canvas around `loops` and feather it there.

        Returns (mask, (x0, y0, x1, y1)); the mask covers only that box, which
        keeps masking cheap compared with full-frame canvases.
        """
        h, w = self.shape[:2]
        margin = int(feather) * 3 + 2 if margin is None else margin
        allpts = np.concatenate([self.pts(l) for l in loops])
        x0 = int(max(allpts[:, 0].min() - margin, 0))
        y0 = int(max(allpts[:, 1].min() - margin, 0))
        x1 = int(min(allpts[:, 0].max() + margin + 1, w))
        y1 = int(min(allpts[:, 1].max() + margin + 1, h))
        local = (y1 - y0, x1 - x0)
        origin = np.array([x0, y0], np.float32)
        mask = np.zeros(local, np.float32)
        for loop in loops:
            mask = np.maximum(mask, polygon_mask(local, self.pts(loop) - origin, 0))
        for loop in subtract:
            hull = cv2.convexHull((self.pts(loop) - origin).astype(np.float32))[:, 0, :]
            mask = np.minimum(mask, 1.0 - polygon_mask(local, hull, 0))
        if feather > 0:
            k = int(feather) * 2 + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)
        return mask, (x0, y0, x1, y1)

    def skin_mask(self, feather=6):
        """Face oval minus eyes, brows and lips, so detail there stays sharp.

        Returns (mask, box) covering the face oval.
        """
        return self._local_mask([FACE_OVAL], feather,
                                subtract=(RIGHT_EYE, LEFT_EYE, RIGHT_BROW, LEFT_BROW, OUTER_LIPS))

    def eye_mask(self, feather=2):
        """Returns (mask, box) covering both eyes."""
        return self._local_mask([RIGHT_EYE, LEFT_EYE], feather)

    def mouth_mask(self, feather=2):
        """Returns (mask, box) covering the inner-lip opening."""
        return self._local_mask([INNER_LIPS], feather)


class FaceTracker:
    """Runs landmark detection once per captured frame.

    Call update(frame) with the raw captured frame, before the effect pipeline.
    Effects then read `tracker.face` (a Face, or None when no face is found).
    """

    def __init__(self, model_path=None, max_faces=1, detect_width=640):
        self.detect_width = detect_width
        from mediapipe.tasks.python import BaseOptions, vision

        path = ensure_model() if model_path is None else Path(model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_faces,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._t0 = time.perf_counter()
        self.face = None

    def update(self, frame):
        import mediapipe as mp

        h, w = frame.shape[:2]
        small = frame
        if w > self.detect_width:
            small = cv2.resize(frame, (self.detect_width, round(h * self.detect_width / w)),
                               interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.perf_counter() - self._t0) * 1000)
        result = self._landmarker.detect_for_video(image, ts)
        if result.face_landmarks:
            pts = np.array([(lm.x * w, lm.y * h) for lm in result.face_landmarks[0]],
                           np.float32)
            self.face = Face(pts, frame.shape)
        else:
            self.face = None
        return self.face

    def close(self):
        self._landmarker.close()
