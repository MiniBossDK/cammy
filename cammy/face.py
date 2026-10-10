"""Face landmark tracking (MediaPipe FaceLandmarker) and region masks."""
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .models import ensure_model
from .presence import Presence

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_NAME = "face_landmarker.task"

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


def polygon_mask(shape, points, feather=0):
    """Float mask in [0, 1] for a filled polygon, optionally Gaussian-feathered."""
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 255)
    mask = mask.astype(np.float32) / 255.0
    if feather > 0:
        k = int(feather) * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


def clip_box(x0, y0, x1, y1, width, height):
    """Integer box clamped to the frame, never inverted (empty boxes have x1 == x0 or y1 == y0).

    A face very close to the camera puts landmarks beyond the frame edges, so a
    feature's raw box can lie wholly outside it; this keeps sizes non-negative.
    """
    x0, y0 = min(max(int(x0), 0), width), min(max(int(y0), 0), height)
    x1, y1 = min(max(int(x1), x0), width), min(max(int(y1), y0), height)
    return x0, y0, x1, y1


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
        return clip_box(x0 - dx, y0 - dy, x1 + dx, y1 + dy, w, h)

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
        x0, y0, x1, y1 = clip_box(allpts[:, 0].min() - margin, allpts[:, 1].min() - margin,
                                  allpts[:, 0].max() + margin + 1, allpts[:, 1].max() + margin + 1, w, h)
        if x1 <= x0 or y1 <= y0:                      # wholly outside the frame
            return np.zeros((y1 - y0, x1 - x0), np.float32), (x0, y0, x1, y1)
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
    # One lock for the class: the background detection thread commits results
    # while the main thread predicts, and both touch the same few fields.
    _lock = threading.RLock()

    """Runs landmark detection once per captured frame.

    Call update(frame) with the raw captured frame, before the effect pipeline.
    Effects then read `tracker.face` (a Face, or None when no face is found).
    """

    def __init__(self, model_path=None, max_faces=1, detect_width=640):
        self.detect_width = detect_width
        from mediapipe.tasks.python import BaseOptions, vision

        path = ensure_model(MODEL_NAME, MODEL_URL) if model_path is None else Path(model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_faces,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._t0 = time.perf_counter()
        self.face = None
        self.presence = 0.0          # 0..1; fades instead of snapping when the face is lost
        self._presence = Presence()
        self._last_face = None
        self._detections = []        # (time, points) of the last two real detections

    def update(self, frame, stamp=None):
        """Detect on `frame`. `stamp` is when the frame was captured (default: now); it dates
        the detection so velocity estimates stay right when the result arrives late."""
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
        with self._lock:
            detected = None
            now = time.perf_counter() if stamp is None else stamp
            if result.face_landmarks:
                pts = np.array([(lm.x * w, lm.y * h) for lm in result.face_landmarks[0]],
                               np.float32)
                if np.isfinite(pts).all():
                    detected = Face(pts, frame.shape)
                if detected is not None:
                    self._last_face = detected
                    self._detections = (self._detections + [(now, pts)])[-2:]
            if detected is None:
                self._detections = []
            self.presence = self._presence.update(detected is not None, now)
            # While presence lasts, keep using the last known face (covered, turned
            # away) so effects fade out rather than switching off in one frame.
            if detected is not None:
                self.face = detected
            elif self.presence > 0 and self._last_face is not None:
                self.face = self._last_face
            else:
                self.face = None
                self._last_face = None
            return self.face

    # A face moves several pixels per frame, so landmarks from the previous
    # detection are visibly behind on frames where detection is skipped.
    MAX_PREDICT_AGE = 0.12      # s: never extrapolate further than this past the last detection
    MAX_VELOCITY_GAP = 0.25     # s: the two detections must be this close to trust their velocity
    MAX_SHIFT = 40.0            # px: cap on the predicted displacement

    def predict(self, shape=None):
        """Move the current face along its recent velocity for a skipped frame.

        Call on frames where update() is not run. Does nothing unless the last
        two updates both found the face and are recent; the extrapolation is
        capped, so a wrong guess stays small.
        """
        with self._lock:
            return self._predict_locked()

    def _predict_locked(self):
        if len(self._detections) < 2 or self.face is None:
            return self.face
        (t0, p0), (t1, p1) = self._detections
        dt = time.perf_counter() - t1
        if not (0.0 < dt <= self.MAX_PREDICT_AGE) or not (0.0 < t1 - t0 <= self.MAX_VELOCITY_GAP):
            return self.face
        shift = (p1 - p0) / (t1 - t0) * dt
        norm = np.linalg.norm(shift, axis=1, keepdims=True)
        shift = shift * np.minimum(1.0, self.MAX_SHIFT / np.maximum(norm, 1e-6))
        self.face = Face(p1 + shift, self.face.shape)
        return self.face

    def close(self):
        self._landmarker.close()
