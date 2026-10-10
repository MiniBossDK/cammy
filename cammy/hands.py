"""Hand landmark tracking and the hand cut-out mask.

The person segmenter works on a small internal image, so thin fingers are
lost and hands in front of the body dissolve. Knowing where the fingers are
(from hand landmarks) lets us draw them back into the mask.
"""
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .models import ensure_model
from .presence import Presence

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_NAME = "hand_landmarker.task"

# MediaPipe hand model: 21 points. 0 wrist; thumb 1-4, index 5-8, middle 9-12,
# ring 13-16, pinky 17-20.
FINGERS = [(0, 1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
FINGER_WIDTH = [0.34, 0.26, 0.27, 0.25, 0.21]   # thickness as a fraction of palm width
PALM = [0, 1, 2, 5, 9, 13, 17]


def palm_width(hand):
    return float(np.linalg.norm(hand[5] - hand[17])) + 1e-6


def hand_box(hand, shape, margin=0.6):
    """Box (x0, y0, x1, y1) around a hand, padded by `margin` palm widths, clamped to the frame."""
    h, w = shape[:2]
    pad = palm_width(hand) * margin
    x0, y0 = hand.min(axis=0) - pad
    x1, y1 = hand.max(axis=0) + pad
    x0, y0 = min(max(int(x0), 0), w), min(max(int(y0), 0), h)
    x1, y1 = min(max(int(x1) + 1, x0), w), min(max(int(y1) + 1, y0), h)
    return x0, y0, x1, y1


def draw_hand(canvas, hand, origin, scale=1.0):
    """Draw the palm and thick finger strokes for one hand onto a uint8 canvas."""
    pw = palm_width(hand)
    pts = np.round(hand - origin).astype(np.int32)
    cv2.fillPoly(canvas, [pts[PALM]], 255)
    for chain, width in zip(FINGERS, FINGER_WIDTH):
        for a, b, taper in zip(chain[:-1], chain[1:], np.linspace(1.0, 0.8, len(chain) - 1)):
            cv2.line(canvas, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]),
                     255, max(int(pw * width * taper * scale), 2))
        cv2.circle(canvas, tuple(int(v) for v in pts[chain[-1]]), max(int(pw * width * 0.4 * scale), 2), 255, -1)


def skin_in_region(roi, core, region):
    """Pixels inside `region` whose colour matches this hand's skin (sampled under `core`)."""
    ycc = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    samples = ycc[cv2.erode(core, np.ones((3, 3), np.uint8)) > 0]
    if len(samples) < 30:
        return np.zeros(core.shape, np.uint8)
    med = np.median(samples, axis=0)
    spread = np.median(np.abs(samples - med), axis=0) * 1.4826
    chroma = np.sqrt((((ycc[..., 1:] - med[1:]) / (spread[1:] * 2.5 + 3)) ** 2).sum(axis=2))
    luma_tol = spread[0] * 3.5 + 25
    skin = ((chroma < 1.0) & (np.abs(ycc[..., 0] - med[0]) < luma_tol) & (region > 0)).astype(np.uint8) * 255
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def hand_mask(frame, hand, feather=1.5, work=0.35):
    """Mask for one hand as (float mask in [0, 1], (x0, y0, x1, y1)) covering only a box around it.

    Thick strokes along the finger landmarks guarantee every finger is kept
    even when motion blur hides its edges; skin colour then fills out the real
    outline of the palm and fingers. A hand wholly outside the frame gives an
    empty box. Fingers are tens of pixels wide, so all of the work happens at
    `work` times the frame size (default about a third) and the mask is scaled back up.
    """
    if not np.isfinite(hand).all():
        return np.zeros((0, 0), np.float32), (0, 0, 0, 0)
    x0, y0, x1, y1 = hand_box(hand, frame.shape)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0), np.float32), (0, 0, 0, 0)
    full = (y1 - y0, x1 - x0)
    sw, sh = max(int(round(full[1] * work)), 1), max(int(round(full[0] * work)), 1)
    roi = cv2.resize(frame[y0:y1, x0:x1], (sw, sh), interpolation=cv2.INTER_AREA)
    # Landmarks into the reduced box (same scale on both axes up to rounding).
    small_hand = (hand - np.array([x0, y0], np.float32)) * np.array([sw / full[1], sh / full[0]], np.float32)
    origin = np.zeros(2, np.float32)
    pw = palm_width(small_hand)

    strokes = np.zeros((sh, sw), np.uint8)
    draw_hand(strokes, small_hand, origin, 1.0)
    core = np.zeros((sh, sw), np.uint8)
    draw_hand(core, small_hand, origin, 0.5)
    region = np.zeros((sh, sw), np.uint8)
    draw_hand(region, small_hand, origin, 2.7)            # generous: thicker strokes instead of a slow dilate

    skin = skin_in_region(roi, core, region)
    mask = np.maximum(strokes, skin).astype(np.float32) / 255.0
    mask = cv2.resize(mask, (full[1], full[0]), interpolation=cv2.INTER_LINEAR)
    if feather > 0:
        size = int(feather) * 2 + 1
        mask = cv2.GaussianBlur(mask, (size, size), feather)
    return mask, (x0, y0, x1, y1)


def add_hands(person_mask, frame, hands, strength=1.0):
    """Union of the person mask and every hand's mask (returns a new array)."""
    if not hands or strength <= 0:
        return person_mask
    out = person_mask.copy()
    for hand in hands:
        mask, (x0, y0, x1, y1) = hand_mask(frame, hand)
        if x1 <= x0 or y1 <= y0:
            continue
        roi = out[y0:y1, x0:x1]
        np.maximum(roi, mask * strength, out=roi)
    return out


class HandTracker:
    # One lock for the class: the background detection thread commits results
    # while the main thread predicts, and both touch the same few fields.
    _lock = threading.RLock()

    """Runs hand landmark detection and keeps `hands` current between detections.

    Mirrors FaceTracker: call update(frame) on frames where detection runs and
    predict() on the frames in between, which moves each hand along its recent
    velocity. When hands vanish (motion blur, leaving the frame) they are held
    briefly and `presence` fades 1 -> 0, so the cut-out does not pop.
    """

    MAX_PREDICT_AGE = 0.12      # s past the last detection to extrapolate
    MAX_VELOCITY_GAP = 0.25     # s between the two detections used for velocity
    MAX_SHIFT = 90.0            # px cap on the extrapolation (hands move fast)

    def __init__(self, model_path=None, max_hands=2, detect_width=640):
        from mediapipe.tasks.python import BaseOptions, vision

        path = ensure_model(MODEL_NAME, MODEL_URL) if model_path is None else Path(model_path)
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._t0 = time.perf_counter()
        self.detect_width = detect_width
        self.hands = []                 # list of (21, 2) float32 arrays, in pixels
        self.presence = 0.0
        self._presence = Presence(hold=0.15, fade_in=0.05, fade_out=0.25)
        self._detections = []           # [(time, [hand arrays])] for the last two detections
        self._last_hands = []

    def update(self, frame, stamp=None):
        """Detect on `frame`; `stamp` is its capture time (default: now), see FaceTracker.update."""
        import mediapipe as mp

        h, w = frame.shape[:2]
        small = frame
        if w > self.detect_width:
            small = cv2.resize(frame, (self.detect_width, round(h * self.detect_width / w)),
                               interpolation=cv2.INTER_AREA)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        ts = int((time.perf_counter() - self._t0) * 1000)
        result = self._landmarker.detect_for_video(image, ts)
        now = time.perf_counter() if stamp is None else stamp
        found = []
        for landmarks in result.hand_landmarks or []:
            pts = np.array([(p.x * w, p.y * h) for p in landmarks], np.float32)
            if np.isfinite(pts).all():
                found.append(pts)
        with self._lock:
            if found:
                self._last_hands = found
                self._detections = (self._detections + [(now, found)])[-2:]
            else:
                self._detections = []
            self.presence = self._presence.update(bool(found), now)
            if found:
                self.hands = found
            elif self.presence > 0:
                self.hands = self._last_hands
            else:
                self.hands = []
                self._last_hands = []
            return self.hands

    def predict(self):
        """Advance the current hands along their velocity for a skipped frame."""
        with self._lock:
            return self._predict_locked()

    def _predict_locked(self):
        if len(self._detections) < 2:
            return self.hands
        (t0, previous), (t1, latest) = self._detections
        dt = time.perf_counter() - t1
        if not (0.0 < dt <= self.MAX_PREDICT_AGE) or not (0.0 < t1 - t0 <= self.MAX_VELOCITY_GAP):
            return self.hands
        moved = []
        for hand in latest:
            near = min(previous, key=lambda p: np.linalg.norm(p[0] - hand[0]), default=None)
            if near is None or np.linalg.norm(near[0] - hand[0]) > palm_width(hand) * 3:
                moved.append(hand)                     # no matching earlier hand: no velocity
                continue
            shift = (hand - near) / (t1 - t0) * dt
            norm = np.linalg.norm(shift, axis=1, keepdims=True)
            moved.append(hand + shift * np.minimum(1.0, self.MAX_SHIFT / np.maximum(norm, 1e-6)))
        self.hands = moved
        return self.hands

    def reset(self):
        """Forget the current hands and their history (e.g. after the image is flipped)."""
        with self._lock:
            self.hands, self.presence, self._last_hands, self._detections = [], 0.0, [], []
            self._presence = Presence(hold=0.15, fade_in=0.05, fade_out=0.25)

    def close(self):
        self._landmarker.close()
