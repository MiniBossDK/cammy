"""Turns raw frames into enhanced frames, following live-adjustable Settings."""
import cv2

from .effects.background import BackgroundBlur, BackgroundReplace
from .effects.brightness_contrast import BrightnessContrast
from .effects.denoise import Denoise
from .effects.eyes_teeth import EyeBrighten, TeethWhiten
from .effects.face_exposure import FaceExposure
from .effects.sharpen import Sharpen
from .effects.skin_smooth import SkinSmooth
from .effects.temporal_denoise import TemporalDenoise
from .effects.white_balance import WhiteBalance
from .face import FaceTracker
from .schedule import due
from .segment import PersonSegmenter
from .settings import Settings


def _hex_to_bgr(text):
    r, g, b = (int(text[i:i + 2], 16) for i in (1, 3, 5))
    return (b, g, r)


class Processor:
    """Owns the detectors and effect objects and applies Settings to them.

    Effects are created once and kept, so changing a slider only updates a
    parameter: temporal state (exposure smoothing, mask smoothing) survives,
    and detectors start only when an effect that needs them is first enabled.

    Call `settings = ...` from any thread; it is read once per frame.
    `tracker_factory` / `segmenter_factory` exist so tests can run without models.
    """

    def __init__(self, settings=None, detect_every=2, segment_every=1, tracker_factory=FaceTracker,
                 segmenter_factory=PersonSegmenter):
        self.settings = settings or Settings()
        self.detect_every = detect_every      # face landmarks: every Nth frame
        self.segment_every = segment_every    # person mask: every frame, a stale mask leaves a halo
        self._tracker_factory = tracker_factory
        self._segmenter_factory = segmenter_factory
        self.tracker = None
        self.segmenter = None
        self._fx = {}
        self._replace_key = None
        self._index = 0
        self.error = None  # last non-fatal problem, e.g. unreadable background image

    def _effect(self, name, factory):
        if name not in self._fx:
            self._fx[name] = factory()
        return self._fx[name]

    def _background_effect(self, s):
        if s.background == "blur":
            fx = self._effect("bg_blur", lambda: BackgroundBlur(self.segmenter))
            fx.strength = s.bg_blur
            return fx
        key = (s.background, s.bg_image if s.background == "image" else s.bg_color)
        if key != self._replace_key:
            self._replace_key = key
            self._fx.pop("bg_replace", None)
        if s.background == "image":
            image = cv2.imread(s.bg_image) if s.bg_image else None
            if image is None:
                self.error = f"cannot read background image: {s.bg_image or '(none chosen)'}"
                return None
            self.error = None
            return self._effect("bg_replace", lambda: BackgroundReplace(self.segmenter, image))
        self.error = None
        return self._effect("bg_replace",
                            lambda: BackgroundReplace(self.segmenter, _hex_to_bgr(s.bg_color)))

    def _face_effect(self, fx, frame):
        """Apply a face effect scaled by how surely a face is there (fades, no snap)."""
        face = self.tracker.face
        if face is None:
            return frame
        presence = getattr(self.tracker, "presence", 1.0)
        if presence <= 0:
            return frame
        out = fx(frame)
        if presence >= 0.999:
            return out
        return cv2.addWeighted(out, presence, frame, 1.0 - presence, 0)

    def process(self, frame):
        """Run detectors (on their schedule) and every enabled effect, in order.

        Order matters: temporal denoise runs first so later brightening does not
        amplify noise it could have removed, exposure runs early so later
        effects see a well-lit face, and denoise runs before sharpen.
        """
        s = self.settings
        if s.wants_face and self.tracker is None:
            self.tracker = self._tracker_factory()
        if s.wants_background and self.segmenter is None:
            self.segmenter = self._segmenter_factory()

        # Detectors see the raw frame and alternate frames to share the budget.
        if s.wants_face:
            if due(self._index, self.detect_every, 0):
                self.tracker.update(frame)
            elif hasattr(self.tracker, "predict"):
                self.tracker.predict()      # skipped frame: follow the face's motion
        if s.wants_background and due(self._index, self.segment_every, 1):
            self.segmenter.update(frame)
        self._index += 1

        if s.temporal_denoise:
            fx = self._effect("tdn", TemporalDenoise)
            fx.strength = s.temporal_denoise
            frame = fx(frame)
        elif "tdn" in self._fx:
            self._fx["tdn"].reset()        # forget history so re-enabling starts clean
        if s.white_balance:
            fx = self._effect("wb", WhiteBalance)
            fx.strength = s.white_balance
            frame = fx(frame)
        if s.face_exposure:
            fx = self._effect("exposure", lambda: FaceExposure(self.tracker))
            fx.strength = s.face_exposure
            frame = fx(frame)
        if s.denoise:
            frame = self._effect("denoise", Denoise)(frame)
        if s.skin_smooth:
            fx = self._effect("skin", lambda: SkinSmooth(self.tracker))
            fx.strength = s.skin_smooth
            frame = self._face_effect(fx, frame)
        if s.contrast != 1.0 or s.brightness:
            fx = self._effect("bc", BrightnessContrast)
            fx.contrast, fx.brightness = s.contrast, s.brightness
            frame = fx(frame)
        if s.eyes:
            fx = self._effect("eyes", lambda: EyeBrighten(self.tracker))
            fx.amount = s.eyes
            frame = self._face_effect(fx, frame)
        if s.teeth:
            fx = self._effect("teeth", lambda: TeethWhiten(self.tracker))
            fx.amount = s.teeth
            frame = self._face_effect(fx, frame)
        if s.wants_background:
            fx = self._background_effect(s)
            if fx is not None:
                frame = fx(frame)
        if s.sharpen:
            fx = self._effect("sharpen", Sharpen)
            fx.amount = s.sharpen
            frame = fx(frame)
        return frame

    def close(self):
        for detector in (self.tracker, self.segmenter):
            if detector is not None:
                detector.close()
        self.tracker = self.segmenter = None
        self._fx.clear()
