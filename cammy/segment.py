"""Person segmentation (MediaPipe selfie segmenter)."""
import time
from pathlib import Path

import cv2
import numpy as np

from .models import ensure_model

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/image_segmenter/"
             "selfie_segmenter/float16/latest/selfie_segmenter.tflite")
MODEL_NAME = "selfie_segmenter.tflite"


def smooth_probability(previous, raw, smoothing):
    """Exponential average of the model's raw person probability over time."""
    if previous is None or previous.shape != raw.shape:
        return raw
    return smoothing * previous + (1.0 - smoothing) * raw


def tighten(probability, edge):
    """Smoothstep the probability between `edge` bounds into a crisp 0..1 mask.

    This must run AFTER temporal smoothing. Tightening first and averaging
    afterwards leaves a vacated pixel at a half-strength value for several
    frames, which shows up as a trail of earlier head positions.
    """
    lo, hi = edge
    t = np.clip((probability - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class PersonSegmenter:
    """Runs person segmentation once per captured frame.

    Call update(frame) with the raw frame before the effect pipeline. Effects
    then read `segmenter.mask`: float32 (H, W) in [0, 1], 1 on the person.
    The probability is smoothed over time to stop the edge from shimmering,
    and then its soft edge is tightened so the cut-out does not look hazy.
    """

    def __init__(self, model_path=None, detect_width=640, smoothing=0.3, edge=(0.3, 0.7)):
        from mediapipe.tasks.python import BaseOptions, vision

        path = ensure_model(MODEL_NAME, MODEL_URL) if model_path is None else Path(model_path)
        options = vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        self._segmenter = vision.ImageSegmenter.create_from_options(options)
        self._t0 = time.perf_counter()
        self.detect_width = detect_width
        self.smoothing = smoothing
        self.edge = edge
        self.mask = None
        self._small = None

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
        result = self._segmenter.segment_for_video(image, ts)
        raw = np.array(result.confidence_masks[0].numpy_view(), np.float32).reshape(small.shape[:2])
        # Smooth the probability over time, THEN tighten the edge, at detection
        # size; upscale once at the end (much cheaper than at full size).
        self._small = smooth_probability(self._small, raw, self.smoothing)
        tight = tighten(self._small, self.edge)
        if tight.shape == (h, w):
            self.mask = tight
        else:
            self.mask = cv2.resize(tight, (w, h), interpolation=cv2.INTER_LINEAR)
        return self.mask

    def close(self):
        self._segmenter.close()
