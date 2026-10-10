"""'AI look': Real-ESRGAN enhancement of the face, blended in through the skin mask."""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

# Where the model, the effect and a possible failure meet: the model runs on its own thread and the
# frame never waits for it. Each frame blends the newest finished patch, shifted to where the face is now.


class AILook:
    """Enhance the face with an AI upscaler and blend the result into the skin.

    A square patch around the face is shrunk to `patch` pixels, enlarged 2x by the model and brought
    back to its original size: smooth, clean skin and crisp edges. The model works on a background
    thread (one patch at a time; when it is busy that frame just reuses the last patch), so a frame
    costs about the blend. The patch is moved by how far the face has travelled since it was made and
    dropped after `max_age` seconds, so it never trails behind a moving face.

    It is blended through the skin mask only: eyes, brows and lips stay live, so blinks and speech
    are never delayed or redrawn. strength 0 is identity. Because it redraws detail it can change
    fine structure (it also does this to text), so it is off by default.
    """

    def __init__(self, tracker, upscaler_factory=None, strength=0.7, patch=192, margin=0.25, max_age=0.35):
        self.tracker = tracker
        self.strength = strength
        self.patch = int(patch)
        self.margin = margin
        self.max_age = max_age
        self._factory = upscaler_factory
        self._upscaler = None
        self._pool = None
        self._future = None
        self._result = None          # (enhanced patch, (x0, y0), (cx, cy) at the time, stamp)
        self._error = None
        self._lock = threading.Lock()

    # ---- the background job ---------------------------------------------------------
    def _make_upscaler(self):
        if self._factory is not None:
            return self._factory()
        from ..ai import Upscaler                     # imported late: ncnn is optional
        return Upscaler()

    def _work(self, crop, origin, centre, stamp):
        try:
            if self._upscaler is None:
                self._upscaler = self._make_upscaler()
            small = cv2.resize(crop, (self.patch, self.patch), interpolation=cv2.INTER_AREA)
            big = self._upscaler.upscale(small)
            interpolation = cv2.INTER_AREA if big.shape[0] >= crop.shape[0] else cv2.INTER_CUBIC
            enhanced = cv2.resize(big, (crop.shape[1], crop.shape[0]), interpolation=interpolation)
            with self._lock:
                self._result = (enhanced, origin, centre, stamp)
        except Exception as exc:                      # surfaced on the next frame, once
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"

    def _square_around_face(self, face, shape):
        """(x0, y0, x1, y1) of a square, padded around the face and kept inside the frame, or None."""
        x0, y0, x1, y1 = face.bbox(self.margin)
        height, width = shape[:2]
        side = min(max(x1 - x0, y1 - y0), width, height)
        if side < 32:
            return None
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        left = int(min(max(cx - side // 2, 0), width - side))
        top = int(min(max(cy - side // 2, 0), height - side))
        return left, top, left + side, top + side

    def _maybe_submit(self, frame, face):
        if self._future is not None and not self._future.done():
            return
        box = self._square_around_face(face, frame.shape)
        if box is None:
            return
        x0, y0, x1, y1 = box
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cammy-ai")
        # the crop is copied: the camera's frame must stay untouched and may be reused while this runs
        self._future = self._pool.submit(self._work, frame[y0:y1, x0:x1].copy(), (x0, y0),
                                         ((x0 + x1) / 2, (y0 + y1) / 2), time.perf_counter())

    # ---- per frame --------------------------------------------------------------------
    def __call__(self, frame):
        with self._lock:
            error, result = self._error, self._result
        if error:
            raise RuntimeError(error)
        face = self.tracker.face
        if face is None or self.strength <= 0:
            return frame
        self._maybe_submit(frame, face)
        if result is None or time.perf_counter() - result[3] > self.max_age:
            return frame
        patch, (px, py), (old_cx, old_cy), _ = result

        bx0, by0, bx1, by1 = face.bbox()
        dx, dy = int(round((bx0 + bx1) / 2 - old_cx)), int(round((by0 + by1) / 2 - old_cy))
        x0, y0 = px + dx, py + dy                      # where the patch sits now
        h, w = patch.shape[:2]
        mask, (mx0, my0, mx1, my1) = face.skin_mask()
        # overlap of: the frame, the shifted patch and the skin-mask box
        ox0, oy0 = max(x0, mx0, 0), max(y0, my0, 0)
        ox1, oy1 = min(x0 + w, mx1, frame.shape[1]), min(y0 + h, my1, frame.shape[0])
        if ox1 <= ox0 or oy1 <= oy0:
            return frame
        alpha = (mask[oy0 - my0:oy1 - my0, ox0 - mx0:ox1 - mx0] * self.strength)[..., None]
        live = frame[oy0:oy1, ox0:ox1].astype(np.float32)
        new = patch[oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0].astype(np.float32)
        out = frame.copy()
        out[oy0:oy1, ox0:ox1] = (live * (1.0 - alpha) + new * alpha).astype(np.uint8)
        return out

    def close(self):
        """Wait for a running job to finish, then stop the thread."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
