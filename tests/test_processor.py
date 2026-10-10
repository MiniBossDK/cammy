import numpy as np

from cammy.processor import Processor
from cammy.settings import Settings


class FakeDetector:
    created = 0

    def __init__(self):
        FakeDetector.created += 1
        self.updates = 0
        self.face = None
        self.mask = None
        self.closed = False

    def update(self, frame):
        self.updates += 1

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
    p = make(Settings(skin_smooth=0.5, background="blur"))
    for _ in range(6):
        p.process(frame())
    assert p.segmenter.updates == 6 and p.tracker.updates == 3


def test_skipped_frames_ask_the_tracker_to_predict():
    class Predicting(FakeDetector):
        predicts = 0

        def predict(self):
            self.predicts += 1

    FakeDetector.created = 0
    p = Processor(Settings(skin_smooth=0.5), detect_every=2,
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
        def update(self, frame):
            raise RuntimeError("model exploded")

    p = Processor(Settings(white_balance=0.5, skin_smooth=0.5), tracker_factory=BadTracker, segmenter_factory=FakeDetector)
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
        def update(self, f):
            raise RuntimeError("hand model exploded")

        def predict(self):
            raise RuntimeError("hand model exploded")

    p = Processor(Settings(background="blur"), tracker_factory=FakeDetector, segmenter_factory=FakeDetector,
                  hand_factory=BadHands)
    f = frame()
    p.process(f)
    assert "hand" in (p.warning or "")
