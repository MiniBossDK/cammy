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


def make(settings=None, **kw):
    FakeDetector.created = 0
    return Processor(settings, tracker_factory=FakeDetector, segmenter_factory=FakeDetector, **kw)


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
    assert p.segmenter is not None and FakeDetector.created == 2


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
