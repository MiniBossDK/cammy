import numpy as np
import pytest

from cammy.processor import Processor
from cammy.settings import Settings


class FakeDetector:
    created = 0

    def __init__(self):
        FakeDetector.created += 1
        self.updates = 0
        self.resets = 0
        self.face = None
        self.mask = None
        self.closed = False

    def update(self, frame, stamp=None):
        self.updates += 1
        self.last_frame = frame

    def reset(self):
        self.resets += 1

    def close(self):
        self.closed = True


def frame():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)


class FakeHands(FakeDetector):
    def __init__(self):
        super().__init__()
        self.hands, self.presence, self.predicts = [], 0.0, 0

    def predict(self):
        self.predicts += 1


def make(settings=None, **kw):
    FakeDetector.created = 0
    kw.setdefault("detect", "sequential")        # deterministic: no background threads
    return Processor(settings, tracker_factory=FakeDetector, segmenter_factory=FakeDetector,
                     hand_factory=FakeHands, **kw)


def test_all_off_returns_frame_untouched_and_starts_no_detectors():
    p = make()
    f = frame()
    assert p.process(f) is f
    assert FakeDetector.created == 0


def test_enabling_effect_changes_frame_live():
    p = make()
    f = frame()
    p.settings = Settings(sharpen=1.0)
    assert not (p.process(f) == f).all()
    p.settings = Settings()
    assert p.process(f) is f


def test_parameter_update_reuses_effect_object():
    p = make(Settings(sharpen=0.2))
    p.process(frame())
    fx = p._fx["sharpen"]
    p.settings = Settings(sharpen=0.9)
    p.process(frame())
    assert p._fx["sharpen"] is fx and fx.amount == 0.9


def test_detectors_start_only_when_needed():
    p = make(Settings(white_balance=0.5))
    p.process(frame())
    assert FakeDetector.created == 0
    p.settings = Settings(skin_smooth=0.5)
    p.process(frame())
    assert p.tracker is not None and p.segmenter is None
    p.settings = Settings(skin_smooth=0.5, background="blur")
    p.process(frame())
    assert p.segmenter is not None and p.hand_tracker is not None and FakeDetector.created == 3


def test_detectors_follow_their_own_schedules():
    p = make(Settings(skin_smooth=0.5, background="blur"), detect_every=2, segment_every=2)
    for _ in range(6):
        p.process(frame())
    assert p.tracker.updates == 3 and p.segmenter.updates == 3


def test_colour_background_replaces_when_mask_present():
    p = make(Settings(background="color", bg_color="#ff0000"))
    p.process(frame())          # creates segmenter
    p.segmenter.mask = np.zeros((48, 64), np.float32)   # everything is background
    out = p.process(frame())
    assert (out == (0, 0, 255)).all()                   # BGR red


def test_bad_background_image_reports_error_and_passes_frame_through():
    p = make(Settings(background="image", bg_image="/no/such/file.png"))
    f = frame()
    p.process(f)
    p.segmenter.mask = np.ones((48, 64), np.float32)
    assert p.process(f) is f
    assert "cannot read" in p.error


def test_changing_colour_rebuilds_replacement():
    p = make(Settings(background="color", bg_color="#ff0000"))
    p.process(frame())
    p.segmenter.mask = np.zeros((48, 64), np.float32)
    p.process(frame())
    p.settings = Settings(background="color", bg_color="#0000ff")
    assert (p.process(frame()) == (255, 0, 0)).all()


def test_close_closes_detectors():
    p = make(Settings(skin_smooth=0.5, background="blur"))
    p.process(frame())
    t, s = p.tracker, p.segmenter
    p.close()
    assert t.closed and s.closed and p.tracker is None


def test_temporal_denoise_runs_first_and_resets_when_disabled():
    p = make(Settings(temporal_denoise=1.0))
    f = frame()
    p.process(f)
    fx = p._fx["tdn"]
    assert fx._acc is not None
    p.settings = Settings()
    assert p.process(f) is f
    assert fx._acc is None                      # history dropped while off
    p.settings = Settings(temporal_denoise=1.0)
    assert p.process(f) is f                    # first frame after re-enable passes through


def test_temporal_denoise_sees_the_raw_frame_before_brightening():
    p = make(Settings(temporal_denoise=1.0, sharpen=1.0))
    for _ in range(3):
        p.process(frame())
    assert list(p._fx)[:2] == ["tdn", "sharpen"]


class PresenceDetector(FakeDetector):
    def __init__(self):
        super().__init__()
        self.face = object()
        self.presence = 1.0


def faded(presence, face=True):
    tracker = PresenceDetector()
    tracker.presence = presence
    if not face:
        tracker.face = None
    p = Processor(Settings(), tracker_factory=lambda: tracker, segmenter_factory=FakeDetector)
    p.tracker = tracker
    return p


def test_face_effect_scales_with_presence():
    f = np.full((4, 4, 3), 100, np.uint8)
    bright = lambda x: np.full_like(x, 200)
    assert (faded(1.0)._face_effect(bright, f) == 200).all()
    assert (faded(0.5)._face_effect(bright, f) == 150).all()
    assert faded(0.0)._face_effect(bright, f) is f
    assert faded(1.0, face=False)._face_effect(bright, f) is f


def test_face_effect_without_presence_attribute_is_all_or_nothing():
    f = np.full((4, 4, 3), 100, np.uint8)
    p = make()
    p.tracker = FakeDetector()
    p.tracker.face = object()
    assert (p._face_effect(lambda x: np.full_like(x, 200), f) == 200).all()


def test_person_mask_updates_every_frame_while_landmarks_alternate():
    p = make(Settings(skin_smooth=0.5, background="blur"), detect_every=2)
    for _ in range(6):
        p.process(frame())
    assert p.segmenter.updates == 6 and p.tracker.updates == 3


def test_skipped_frames_ask_the_tracker_to_predict():
    class Predicting(FakeDetector):
        predicts = 0

        def predict(self):
            self.predicts += 1

    FakeDetector.created = 0
    p = Processor(Settings(skin_smooth=0.5), detect_every=2, detect="sequential",
                  tracker_factory=Predicting, segmenter_factory=FakeDetector)
    for _ in range(6):
        p.process(frame())
    assert p.tracker.updates == 3 and p.tracker.predicts == 3


class Boom:
    """An effect that always fails."""
    strength = 0

    def __call__(self, frame):
        raise ValueError("kaboom")


def test_a_failing_effect_is_skipped_and_reported_not_fatal():
    p = make(Settings(sharpen=1.0))
    f = frame()
    p.process(f)                                   # creates the effect
    p._fx["sharpen"] = Boom()
    out = p.process(f)
    assert out is f                                # frame passes through unchanged
    assert "sharpen failed" in p.warning and "kaboom" in p.warning


def test_warning_clears_when_the_effect_recovers():
    p = make(Settings(sharpen=1.0))
    f = frame()
    p.process(f)
    good = p._fx["sharpen"]
    p._fx["sharpen"] = Boom()
    p.process(f)
    assert p.warning
    p._fx["sharpen"] = good
    p.process(f)
    assert p.warning is None


def test_other_effects_still_run_when_one_fails():
    p = make(Settings(sharpen=1.0, contrast=1.5))
    f = frame()
    p.process(f)
    p._fx["sharpen"] = Boom()
    out = p.process(f)
    assert not (out == f).all()                    # contrast still applied


def test_a_failing_detector_does_not_stop_the_frame():
    class BadTracker(FakeDetector):
        def update(self, frame, stamp=None):
            raise RuntimeError("model exploded")

    p = Processor(Settings(white_balance=0.5, skin_smooth=0.5), detect="sequential", tracker_factory=BadTracker, segmenter_factory=FakeDetector)
    f = frame()
    out = p.process(f)
    assert out.shape == f.shape
    assert "face tracking failed" in p.warning


def test_each_distinct_failure_is_logged_once(caplog):
    import logging
    p = make(Settings(sharpen=1.0))
    f = frame()
    p.process(f)
    p._fx["sharpen"] = Boom()
    with caplog.at_level(logging.WARNING, logger="cammy"):
        for _ in range(5):
            p.process(f)
    assert len([r for r in caplog.records if "sharpen failed" in r.getMessage()]) == 1


def test_hand_tracker_only_starts_for_a_background_effect_with_hands_on():
    p = make(Settings(skin_smooth=0.5))
    p.process(frame())
    assert p.hand_tracker is None                               # no background effect: no hand work
    p = make(Settings(background="blur", keep_hands=False))
    p.process(frame())
    assert p.hand_tracker is None and p.segmenter is not None   # hands switched off
    p = make(Settings(background="blur"))
    p.process(frame())
    assert p.hand_tracker is not None


def test_hands_alternate_with_face_landmarks_and_predict_in_between():
    p = make(Settings(skin_smooth=0.5, background="blur"), detect_every=2)
    for _ in range(6):
        p.process(frame())
    assert p.tracker.updates == 3                               # face on even frames
    assert p.hand_tracker.updates == 3 and p.hand_tracker.predicts == 3   # hands on odd frames, predicted otherwise


def test_effects_read_the_combined_cutout_mask():
    import cammy.hands as hands_module
    p = make(Settings(background="color", bg_color="#ff0000"))
    p.process(frame())
    p.segmenter.mask = np.zeros((48, 64), np.float32)           # nobody found by the person model
    p.hand_tracker.hands = []
    assert (p.process(frame()) == (0, 0, 255)).all()            # all background
    hand = np.zeros((21, 2), np.float32)                        # a hand the person model missed
    hand[:, 0], hand[:, 1] = np.linspace(20, 44, 21), np.linspace(36, 12, 21)
    p.hand_tracker.hands, p.hand_tracker.presence = [hand], 1.0
    out = p.process(frame())
    assert not (out == (0, 0, 255)).all()                        # hand area kept from the original frame


def test_hand_failures_are_reported_not_fatal():
    class BadHands(FakeHands):
        def update(self, f, stamp=None):
            raise RuntimeError("hand model exploded")

        def predict(self):
            raise RuntimeError("hand model exploded")

    p = Processor(Settings(background="blur"), detect="sequential", tracker_factory=FakeDetector, segmenter_factory=FakeDetector,
                  hand_factory=BadHands)
    f = frame()
    p.process(f)
    assert "hand" in (p.warning or "")


class Slow(FakeDetector):
    delay = 0.08

    def update(self, frame, stamp=None):
        import time
        time.sleep(self.delay)
        super().update(frame, stamp)


class SlowHands(Slow, FakeHands):
    pass


def timed(mode):
    import time
    p = Processor(Settings(skin_smooth=0.5, background="blur"), detect_every=1, segment_every=1,
                  tracker_factory=Slow, segmenter_factory=Slow, hand_factory=SlowHands, detect=mode)
    f = frame()
    p.process(f)                                  # creates detectors and the thread pool
    start = time.perf_counter()
    p.process(f)
    elapsed = time.perf_counter() - start
    p.close()
    return elapsed


def test_detectors_run_concurrently_when_parallel():
    sequential, parallel = timed("sequential"), timed("parallel")
    assert sequential > 0.22                       # three 80 ms detectors one after another
    assert parallel < sequential * 0.6             # overlapped: about the slowest one


def test_parallel_failures_are_still_guarded_and_reported():
    class Bad(Slow):
        def update(self, frame, stamp=None):
            raise RuntimeError("detector exploded")

    p = Processor(Settings(skin_smooth=0.5, background="blur"), detect_every=1, tracker_factory=Bad,
                  segmenter_factory=Slow, hand_factory=SlowHands, detect="parallel")
    out = p.process(frame())
    assert out.shape == frame().shape
    assert "face tracking failed" in p.warning
    p.close()


def test_close_shuts_the_pool_down():
    p = Processor(Settings(skin_smooth=0.5, background="blur"), detect_every=1, tracker_factory=FakeDetector,
                  segmenter_factory=FakeDetector, hand_factory=FakeHands, detect="parallel")
    p.process(frame())
    assert p._pool is not None
    p.close()
    assert p._pool is None


def test_every_detector_receives_the_actual_frame_in_every_mode():
    for mode in ("sequential", "parallel", "async"):
        p = Processor(Settings(skin_smooth=0.5, background="blur"), detect_every=1, tracker_factory=FakeDetector,
                      segmenter_factory=FakeDetector, hand_factory=FakeHands, detect=mode)
        f = frame()
        p.process(f)
        for fut in list(p._inflight.values()):          # async: let the background jobs finish
            fut.result()
        for detector in (p.tracker, p.segmenter, p.hand_tracker):
            assert isinstance(detector.last_frame, np.ndarray) and detector.last_frame is f, mode
        assert p.warning is None, mode
        p.close()


# ---- async detection -------------------------------------------------------------

class Watched(Slow):
    """A detector that notices if it is closed while still running inference."""
    running = False
    closed_while_running = False

    def update(self, frame, stamp=None):
        self.running = True
        try:
            super().update(frame, stamp)
        finally:
            self.running = False

    def close(self):
        self.closed_while_running = self.running
        super().close()


class MaskedSlow(Slow):
    def update(self, frame, stamp=None):
        super().update(frame, stamp)
        self.mask = np.zeros(frame.shape[:2], np.float32)       # nobody found: everything is background


def async_processor(settings, **detectors):
    FakeDetector.created = 0
    return Processor(settings, detect="async", detect_every=1, segment_every=1,
                     tracker_factory=detectors.get("tracker", Slow), segmenter_factory=detectors.get("segmenter", Slow),
                     hand_factory=detectors.get("hands", SlowHands))


def test_async_frames_do_not_wait_for_inference():
    import time
    p = async_processor(Settings(skin_smooth=0.5))                 # one 80 ms detector (face)
    f = frame()
    p.process(f)
    start = time.perf_counter()
    for _ in range(5):
        p.process(f)
    per_frame = (time.perf_counter() - start) / 5
    assert per_frame < 0.03                                         # nowhere near the 80 ms inference
    p.close()


def test_a_busy_detector_is_not_started_again():
    p = async_processor(Settings(skin_smooth=0.5))
    f = frame()
    for _ in range(10):
        p.process(f)                                                # 10 fast frames, one slow job
    assert p.tracker.updates <= 1
    p.close()
    assert p.tracker is None


def test_finished_results_are_used_on_later_frames():
    import time
    p = async_processor(Settings(skin_smooth=0.5))
    f = frame()
    p.process(f)
    deadline = time.perf_counter() + 2
    while p.tracker.updates < 1 and time.perf_counter() < deadline:
        time.sleep(0.01)
    p.process(f)                                                    # a new job can start now
    deadline = time.perf_counter() + 2
    while p.tracker.updates < 2 and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert p.tracker.updates >= 2
    p.close()


def test_the_first_frame_never_shows_the_unprocessed_background():
    p = async_processor(Settings(background="color", bg_color="#ff0000", keep_hands=False), segmenter=MaskedSlow)
    out = p.process(frame())                                        # waits once for the first person mask
    assert (out == (0, 0, 255)).all()
    p.close()


def test_close_waits_for_running_inference_before_closing_models():
    p = async_processor(Settings(skin_smooth=0.5), tracker=Watched)
    tracker = None
    p.process(frame())
    tracker = p.tracker
    assert tracker.running                                          # job still in flight
    p.close()
    assert tracker.closed and not tracker.closed_while_running


def test_async_failures_are_reported():
    class Bad(Slow):
        delay = 0.0

        def update(self, frame, stamp=None):
            raise RuntimeError("async detector exploded")

    p = async_processor(Settings(skin_smooth=0.5), tracker=Bad)
    p.process(frame())
    for fut in list(p._inflight.values()):
        fut.result()
    assert "face tracking failed" in p.warning
    p.close()


def test_unknown_detect_mode_is_rejected():
    with pytest.raises(ValueError):
        Processor(Settings(), detect="sometimes")


# ---- mirror ----------------------------------------------------------------------

def test_mirror_flips_the_picture_left_right():
    p = make(Settings(mirror=True))
    f = frame()
    assert (p.process(f) == f[:, ::-1]).all()
    p.settings = Settings(mirror=False)
    assert p.process(f) is f


def test_mirror_never_writes_to_the_cameras_frame():
    p = make(Settings(mirror=True, sharpen=1.0, white_balance=1.0))
    f = frame()
    original = f.copy()
    p.process(f)
    assert (f == original).all()


def test_detectors_see_the_mirrored_picture():
    p = make(Settings(mirror=True, skin_smooth=0.5, background="blur"))
    f = frame()
    p.process(f)
    for detector in (p.tracker, p.segmenter, p.hand_tracker):
        assert (detector.last_frame == f[:, ::-1]).all()


def test_toggling_mirror_clears_orientation_dependent_history():
    p = make(Settings(skin_smooth=0.5, background="blur", temporal_denoise=1.0))
    f = frame()
    p.process(f)
    p.process(f)
    detectors = (p.tracker, p.segmenter, p.hand_tracker)
    assert [d.resets for d in detectors] == [0, 0, 0] and p._fx["tdn"]._acc is not None
    p.settings = p.settings.replace(mirror=True)
    p.process(f)
    assert [d.resets for d in detectors] == [1, 1, 1]
    p.process(f)
    assert [d.resets for d in detectors] == [1, 1, 1]            # only on the change, not every frame


def test_mirror_does_not_flip_a_chosen_background_image(tmp_path):
    import cv2
    image = np.zeros((48, 64, 3), np.uint8)
    image[:, :32] = (255, 0, 0)                                  # left half blue
    path = str(tmp_path / "bg.png")
    cv2.imwrite(path, image)
    p = make(Settings(mirror=True, background="image", bg_image=path, keep_hands=False))
    p.process(frame())
    p.segmenter.mask = np.zeros((48, 64), np.float32)            # all background
    out = p.process(frame())
    assert out[10, 5, 0] > 200 and out[10, 60, 0] < 50           # blue is still on the left
