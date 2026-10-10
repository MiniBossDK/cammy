import time

import numpy as np
import pytest

from cammy.effects.ai_look import AILook
from conftest import FakeTracker, synthetic_face

SHAPE = (240, 320, 3)


class FakeUpscaler:
    """Doubles the size and paints everything one flat colour, so the blend is easy to see."""

    def __init__(self, value=250, delay=0.0):
        self.value, self.delay, self.calls = value, delay, 0

    def upscale(self, bgr):
        self.calls += 1
        time.sleep(self.delay)
        return np.full((bgr.shape[0] * 2, bgr.shape[1] * 2, 3), self.value, np.uint8)


def wait(effect, timeout=2.0):
    deadline = time.perf_counter() + timeout
    while effect._future is not None and not effect._future.done() and time.perf_counter() < deadline:
        time.sleep(0.005)


def make(tracker, **kw):
    fake = kw.pop("fake", None) or FakeUpscaler()
    return AILook(tracker, upscaler_factory=lambda: fake, **kw), fake


def frame(value=100):
    return np.full(SHAPE, value, np.uint8)


def test_blends_the_ai_patch_into_the_skin_only(face, tracker):
    fx, fake = make(tracker, strength=1.0)
    f = frame()
    fx(f)                                        # starts the background job; nothing to blend yet
    wait(fx)
    out = fx(f)
    skin, _ = face.skin_mask()
    x0, y0, _, _ = face.skin_mask()[1]
    centre = skin[skin.shape[0] // 2 + 25, skin.shape[1] // 2]
    assert centre > 0.9                          # a cheek pixel, safely inside the skin mask
    cy, cx = y0 + skin.shape[0] // 2 + 25, x0 + skin.shape[1] // 2
    assert out[cy, cx].mean() > 200              # moved to the AI colour (250)
    assert (out[5, 5] == f[5, 5]).all()          # far from the face: untouched


def test_eyes_stay_live(face, tracker):
    from cammy.face import LEFT_EYE, RIGHT_EYE
    from test_eyes_teeth import with_region
    with_region(face, RIGHT_EYE, (140, 110), half=14)        # realistically sized eye polygons inside the face
    with_region(face, LEFT_EYE, (180, 110), half=14)
    fx, _ = make(tracker, strength=1.0)
    f = frame()
    f[100:121, 130:151] = (30, 40, 50)                       # a distinctive "eye" in the live picture
    fx(f)
    wait(fx)
    out = fx(f)
    assert (out[110, 140] == f[110, 140]).all()              # the eye is the live pixel, not the AI colour
    assert out[130, 160].mean() > 200                        # while skin elsewhere is AI-blended


def test_strength_scales_the_blend(face, tracker):
    results = []
    for strength in (0.3, 1.0):
        fx, _ = make(tracker, strength=strength)
        f = frame()
        fx(f)
        wait(fx)
        results.append(fx(f)[120, 160].mean())
        fx.close()
    assert 100 < results[0] < results[1] <= 250


def test_zero_strength_and_no_face_are_identity(face, tracker):
    f = frame()
    fx, fake = make(tracker, strength=0)
    assert fx(f) is f and fake.calls == 0
    fx, fake = make(FakeTracker(None), strength=1.0)
    assert fx(f) is f and fake.calls == 0


def test_the_frame_never_waits_for_the_model(face, tracker):
    fx, fake = make(tracker, strength=1.0, fake=FakeUpscaler(delay=0.3))
    f = frame()
    start = time.perf_counter()
    for _ in range(5):
        fx(f)
    assert time.perf_counter() - start < 0.15            # five frames while the 300 ms model runs
    assert fake.calls <= 1                               # a busy model is not started again
    fx.close()


def test_a_stale_patch_is_dropped(face, tracker):
    fx, _ = make(tracker, strength=1.0, max_age=0.05)
    f = frame()
    fx(f)
    wait(fx)
    time.sleep(0.12)
    fx._future = fx._future                               # job done; next call would start a new one
    out = fx(f)
    assert (out == f).all()                               # too old to show; the live frame is returned as is


def test_the_patch_follows_a_moving_face(face, tracker):
    fx, _ = make(tracker, strength=1.0)
    f = frame()
    fx(f)
    wait(fx)
    moved = synthetic_face(SHAPE, center=(190, 120), radius=60)          # face moved 30 px right
    tracker.face = moved
    out = fx(f)
    skin, box = moved.skin_mask()
    cy, cx = box[1] + skin.shape[0] // 2 + 25, box[0] + skin.shape[1] // 2
    assert out[cy, cx].mean() > 200                       # the AI patch is on the face's new position


def test_errors_are_reported_not_swallowed(face, tracker):
    class Broken:
        def upscale(self, bgr):
            raise RuntimeError("no GPU")

    fx = AILook(tracker, upscaler_factory=Broken, strength=1.0)
    f = frame()
    fx(f)
    wait(fx)
    with pytest.raises(RuntimeError, match="no GPU"):
        fx(f)


def test_a_missing_library_gives_an_actionable_message(face, tracker):
    def missing():
        raise ImportError("The AI look needs the 'ncnn' package. Install it WITHOUT its dependencies:  pip install --no-deps ncnn")

    fx = AILook(tracker, upscaler_factory=missing, strength=1.0)
    f = frame()
    fx(f)
    wait(fx)
    with pytest.raises(RuntimeError, match="--no-deps ncnn"):
        fx(f)


def test_the_input_frame_is_never_modified(face, tracker):
    fx, _ = make(tracker, strength=1.0)
    f = frame()
    keep = f.copy()
    for _ in range(3):
        fx(f)
        wait(fx)
    assert (f == keep).all()


@pytest.mark.parametrize("cx,cy,radius", [(5, 5, 60), (315, 235, 60), (160, 120, 400), (-200, 100, 60), (160, 700, 60)])
def test_faces_at_the_edge_or_outside_the_frame_are_safe(cx, cy, radius):
    face = synthetic_face(SHAPE, center=(cx, cy), radius=radius)
    fx, _ = make(FakeTracker(face), strength=1.0)
    f = frame()
    for _ in range(3):
        out = fx(f)
        wait(fx)
    assert out.shape == f.shape


def test_close_waits_for_the_running_job_and_is_repeatable(face, tracker):
    fx, fake = make(tracker, strength=1.0, fake=FakeUpscaler(delay=0.1))
    fx(frame())
    fx.close()
    assert fx._result is not None                         # the job finished before the thread stopped
    fx.close()
