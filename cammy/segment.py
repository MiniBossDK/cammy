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


def finish_mask(tight, size, edge_blur):
    """Scale the tightened mask up to `size` (width, height) and anti-alias its outline.

    The model's own mask is coarse (about 5 px per step at 720p), so even a tightened, bilinearly
    enlarged mask has a stair-stepped outline. A light Gaussian on the full-size mask rounds the
    steps off (measured on a still head: 8% less stair-step detail at sigma 1.5, edge 2.2 -> 3.6 px
    wide) for about 2 ms, and it runs in the segmentation thread. Larger values soften the edge into
    a visible halo; the more expensive guided filter did not do better (and cost 17 ms).
    """
    w, h = size
    mask = tight if tight.shape == (h, w) else cv2.resize(tight, (w, h), interpolation=cv2.INTER_LINEAR)
    if edge_blur > 0:
        mask = np.clip(cv2.GaussianBlur(mask, (0, 0), edge_blur), 0.0, 1.0)
    return mask


class PersonSegmenter:
    """Runs person segmentation once per captured frame.

    Call update(frame) with the raw frame before the effect pipeline. Effects
    then read `segmenter.mask`: float32 (H, W) in [0, 1], 1 on the person.
    The probability is smoothed over time to stop the edge from shimmering,
    and then its soft edge is tightened so the cut-out does not look hazy.
    """

    def __init__(self, model_path=None, detect_width=640, smoothing=0.3, edge=(0.3, 0.7), edge_blur=1.5):
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
        self.edge_blur = edge_blur      # sigma (px) of the anti-aliasing blur on the full-size mask; 0 = off
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
        self.mask = finish_mask(tighten(self._small, self.edge), (w, h), self.edge_blur)
        return self.mask

    def reset(self):
        """Drop the current mask and its history (e.g. after the image is flipped)."""
        self.mask = None
        self._small = None

    def close(self):
        self._segmenter.close()
