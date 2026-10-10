"""Turns raw frames into enhanced frames, following live-adjustable Settings."""
import logging
import time
from concurrent.futures import ThreadPoolExecutor

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
from .cutout import Cutout
from .face import FaceTracker
from .hands import HandTracker
from .schedule import due
from .segment import PersonSegmenter
from .settings import Settings


log = logging.getLogger("cammy")

# async: detectors run in background threads and effects use their latest results
#   (the frame never waits for inference; results are at most about a frame old).
# parallel: this frame's detectors run on threads, but the frame waits for all of them.
# sequential: one after another on the calling thread (deterministic: used for replays).
DETECT_MODES = ("async", "parallel", "sequential")


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

    def __init__(self, settings=None, detect_every=1, segment_every=1, tracker_factory=FaceTracker,
                 segmenter_factory=PersonSegmenter, hand_factory=HandTracker, detect="async"):
        self.settings = settings or Settings()
        self.detect_every = detect_every      # face landmarks: every Nth frame
        self.segment_every = segment_every    # person mask: every frame, a stale mask leaves a halo
        self._tracker_factory = tracker_factory
        self._segmenter_factory = segmenter_factory
        self._hand_factory = hand_factory
        if detect not in DETECT_MODES:
            raise ValueError(f"detect must be one of {DETECT_MODES}, not {detect!r}")
        self.detect = detect
        self._pool = None
        self._inflight = {}           # detector name -> Future of its running job
        self.tracker = None
        self.segmenter = None
        self.hand_tracker = None
        self.cutout = None
        self._fx = {}
        self._replace_key = None
        self._index = 0
        self.error = None  # fatal-for-the-user problem, e.g. unreadable background image
        self._failures = {}   # step name -> message, for steps that raised on their last run
        self._reported = set()

    @property
    def warning(self):
        """Text describing steps that failed on their last run, or None."""
        failures = list(self._failures.items())     # copy: background threads add entries
        if not failures:
            return None
        return "; ".join(f"{name} failed ({message})" for name, message in failures)

    def _guard(self, name, fn, *args):
        """Call fn(*args); if it raises, record it and return (False, None).

        One bad step (an odd landmark layout, a bug in an effect) must not stop
        the video. The failure is kept in `warning` and logged once per
        distinct error; the step is tried again on the next frame.
        """
        try:
            out = fn(*args)
        except Exception as exc:
            self._failures[name] = f"{type(exc).__name__}: {exc}"
            key = (name, type(exc).__name__, str(exc))
            if key not in self._reported:
                self._reported.add(key)
                log.warning("%s failed, skipping it for this frame", name, exc_info=True)
            return False, None
        self._failures.pop(name, None)
        return True, out

    def _run(self, name, fn, frame):
        """Apply a frame -> frame step, passing the frame through unchanged if it fails."""
        ok, out = self._guard(name, fn, frame)
        return out if ok else frame

    def _effect(self, name, factory):
        if name not in self._fx:
            self._fx[name] = factory()
        return self._fx[name]

    def _background_effect(self, s):
        if s.background == "blur":
            fx = self._effect("bg_blur", lambda: BackgroundBlur(self.cutout))
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
            return self._effect("bg_replace", lambda: BackgroundReplace(self.cutout, image))
        self.error = None
        return self._effect("bg_replace",
                            lambda: BackgroundReplace(self.cutout, _hex_to_bgr(s.bg_color)))

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

    def _pool_for_jobs(self):
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="cammy-detect")
        return self._pool

    def _detect(self, s, frame):
        """Bring the detectors up to date for this frame, then refresh the cut-out.

        Detectors see the raw frame. Which of them run this frame follows
        `detect_every` / `segment_every`; face and hand landmarks use offsets
        0 and 1 so that, when throttled, they take alternate frames.

        In "async" mode inference never blocks the frame: each detector works
        in its own thread on the newest frame it is free to take (a busy one is
        skipped), and this frame's effects use whatever finished last, moved
        forward to now by the trackers' velocity prediction (landmarks are
        dated with the capture time, so a late result is extrapolated correctly).
        The other modes wait for the detectors before returning.
        """
        stamp = time.perf_counter()
        jobs, predictions = [], []                  # inference calls to run; trackers to extrapolate
        if s.wants_face:
            if due(self._index, self.detect_every, 0):
                jobs.append(("face tracking", self.tracker.update, frame, stamp))
            if hasattr(self.tracker, "predict"):
                predictions.append(("face tracking", "face prediction", self.tracker.predict))
        if s.wants_background and due(self._index, self.segment_every, 1):
            jobs.append(("person mask", self.segmenter.update, frame))
        if s.wants_hands:
            if due(self._index, self.detect_every, 1):
                jobs.append(("hand tracking", self.hand_tracker.update, frame, stamp))
            predictions.append(("hand tracking", "hand prediction", self.hand_tracker.predict))

        if self.detect == "async":
            pool = self._pool_for_jobs()
            for job in jobs:
                running = self._inflight.get(job[0])
                if running is None or running.done():          # a busy detector skips this frame
                    self._inflight[job[0]] = pool.submit(self._guard, *job)
            # Never show an unprocessed first frame: wait once for the first person mask.
            if s.wants_background and self.segmenter.mask is None and "person mask" in self._inflight:
                self._inflight["person mask"].result()
            fresh = set()                                       # async results are always a little late
        elif self.detect == "parallel" and len(jobs) > 1:
            pool = self._pool_for_jobs()
            for future in [pool.submit(self._guard, *job) for job in jobs]:
                future.result()
            fresh = {job[0] for job in jobs}
        else:
            for job in jobs:
                self._guard(*job)
            fresh = {job[0] for job in jobs}

        for tracked, name, predict in predictions:
            if tracked not in fresh:                            # skipped or late: extrapolate to now
                self._guard(name, predict)
        if s.wants_background:
            self._guard("cutout", self.cutout.refresh, frame)

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
        if s.wants_hands and self.hand_tracker is None:
            self.hand_tracker = self._hand_factory()
        if s.wants_background and self.cutout is None:
            self.cutout = Cutout(self.segmenter)
        if self.cutout is not None:
            self.cutout.hand_tracker = self.hand_tracker if s.wants_hands else None

        self._detect(s, frame)
        self._index += 1

        if s.temporal_denoise:
            fx = self._effect("tdn", TemporalDenoise)
            fx.strength = s.temporal_denoise
            frame = self._run("temporal denoise", fx, frame)
        elif "tdn" in self._fx:
            self._fx["tdn"].reset()        # forget history so re-enabling starts clean
        if s.white_balance:
            fx = self._effect("wb", WhiteBalance)
            fx.strength = s.white_balance
            frame = self._run("white balance", fx, frame)
        if s.face_exposure:
            fx = self._effect("exposure", lambda: FaceExposure(self.tracker))
            fx.strength = s.face_exposure
            frame = self._run("face exposure", fx, frame)
        if s.denoise:
            frame = self._run("denoise", self._effect("denoise", Denoise), frame)
        if s.skin_smooth:
            fx = self._effect("skin", lambda: SkinSmooth(self.tracker))
            fx.strength = s.skin_smooth
            frame = self._run("skin smoothing", lambda f: self._face_effect(fx, f), frame)
        if s.contrast != 1.0 or s.brightness:
            fx = self._effect("bc", BrightnessContrast)
            fx.contrast, fx.brightness = s.contrast, s.brightness
            frame = self._run("brightness/contrast", fx, frame)
        if s.eyes:
            fx = self._effect("eyes", lambda: EyeBrighten(self.tracker))
            fx.amount = s.eyes
            frame = self._run("eye brightening", lambda f: self._face_effect(fx, f), frame)
        if s.teeth:
            fx = self._effect("teeth", lambda: TeethWhiten(self.tracker))
            fx.amount = s.teeth
            frame = self._run("teeth whitening", lambda f: self._face_effect(fx, f), frame)
        if s.wants_background:
            fx = self._background_effect(s)
            if fx is not None:
                frame = self._run("background", fx, frame)
        if s.sharpen:
            fx = self._effect("sharpen", Sharpen)
            fx.amount = s.sharpen
            frame = self._run("sharpen", fx, frame)
        return frame

    def close(self):
        if self._pool is not None:                  # let running inference finish before closing models
            self._pool.shutdown(wait=True)
            self._pool = None
        self._inflight.clear()
        for detector in (self.tracker, self.segmenter, self.hand_tracker):
            if detector is not None:
                detector.close()
        self.tracker = self.segmenter = self.hand_tracker = self.cutout = None
        self._fx.clear()
