from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import cammy.hands as hands_module
from cammy.hands import (FINGERS, HandTracker, add_hands, draw_hand, hand_box, hand_mask, palm_width)
from cammy.presence import Presence

SHAPE = (240, 320, 3)


def make_hand(cx=160, cy=150, size=60.0):
    """A spread hand pointing up: wrist at the bottom, five fingers fanned out."""
    p = np.zeros((21, 2), np.float32)
    p[0] = (cx, cy + size * 1.2)
    bases = {1: (-0.9, -0.2), 5: (-0.55, -0.9), 9: (-0.1, -1.0), 13: (0.35, -0.95), 17: (0.75, -0.8)}
    for first, (bx, by) in bases.items():
        for k in range(4):
            f = 1.0 + 0.45 * k
            p[first + k] = (cx + bx * size * f * (1.0 if first != 1 else 1.1), cy + by * size * f)
    return p


def skin_frame(hand, colour=(120, 150, 200)):
    frame = np.full(SHAPE, (200, 200, 200), np.uint8)
    mask = np.zeros(SHAPE[:2], np.uint8)
    draw_hand(mask, hand, np.zeros(2, np.float32), 1.0)
    frame[mask > 0] = colour
    return frame, mask


def test_palm_width_and_box_contain_the_hand():
    hand = make_hand()
    x0, y0, x1, y1 = hand_box(hand, SHAPE)
    assert x0 <= hand[:, 0].min() and x1 >= hand[:, 0].max()
    assert y0 <= hand[:, 1].min() and y1 >= hand[:, 1].max()
    assert palm_width(hand) > 10


def test_mask_covers_every_finger_and_the_palm():
    hand = make_hand()
    frame, _ = skin_frame(hand)
    mask, (x0, y0, x1, y1) = hand_mask(frame, hand)
    assert mask.shape == (y1 - y0, x1 - x0)
    for chain in FINGERS:
        tip = np.round(hand[chain[-1]] - [x0, y0]).astype(int)
        assert mask[tip[1], tip[0]] > 0.5, f"fingertip {chain[-1]} missing"
    palm = np.round(hand[[0, 5, 9, 13, 17]].mean(axis=0) - [x0, y0]).astype(int)
    assert mask[palm[1], palm[0]] > 0.9


def test_strokes_keep_fingers_even_when_colour_does_not_match():
    hand = make_hand()
    frame = np.full(SHAPE, 200, np.uint8)                     # no skin colour anywhere (e.g. glove)
    mask, (x0, y0, _, _) = hand_mask(frame, hand)
    tip = np.round(hand[8] - [x0, y0]).astype(int)
    assert mask[tip[1], tip[0]] > 0.5


def test_mask_is_empty_far_from_the_hand():
    hand = make_hand(cx=100, cy=120)
    frame, _ = skin_frame(hand)
    mask, (x0, y0, x1, y1) = hand_mask(frame, hand)
    assert (x1 - x0) < SHAPE[1] and mask[0, 0] < 0.05


def test_skin_colour_extends_the_mask_to_the_real_outline():
    hand = make_hand()
    frame, strokes = skin_frame(hand)
    extra = np.zeros(SHAPE[:2], np.uint8)
    extra[135:160, 100:118] = 255                             # skin outside the thin finger strokes
    frame[extra > 0] = (120, 150, 200)
    mask, (x0, y0, _, _) = hand_mask(frame, hand)
    assert mask[147 - y0, 108 - x0] > 0.5


@pytest.mark.parametrize("cx,cy,size", [(-200, 100, 60), (500, 100, 60), (160, -300, 60), (160, 600, 60), (160, 120, 400), (5, 5, 80)])
def test_hands_partly_or_wholly_outside_the_frame_are_safe(cx, cy, size):
    hand = make_hand(cx, cy, size)
    frame = np.full(SHAPE, 150, np.uint8)
    mask, (x0, y0, x1, y1) = hand_mask(frame, hand)
    assert 0 <= x0 <= x1 <= SHAPE[1] and 0 <= y0 <= y1 <= SHAPE[0]
    assert mask.shape == (y1 - y0, x1 - x0)
    person = np.zeros(SHAPE[:2], np.float32)
    assert add_hands(person, frame, [hand]).shape == person.shape


def test_nan_landmarks_give_an_empty_mask():
    hand = make_hand()
    hand[3] = np.nan
    mask, box = hand_mask(np.full(SHAPE, 100, np.uint8), hand)
    assert box == (0, 0, 0, 0) and mask.size == 0


def test_add_hands_unions_and_never_lowers_the_person_mask():
    hand = make_hand()
    frame, _ = skin_frame(hand)
    person = np.zeros(SHAPE[:2], np.float32)
    person[:, :60] = 1.0
    out = add_hands(person, frame, [hand])
    assert (out >= person).all()
    tip = np.round(hand[12]).astype(int)
    assert out[tip[1], tip[0]] > 0.5 and person[tip[1], tip[0]] == 0
    assert out is not person                                  # a new array, original untouched


def test_add_hands_without_hands_returns_the_mask_unchanged():
    person = np.full(SHAPE[:2], 0.3, np.float32)
    assert add_hands(person, np.zeros(SHAPE, np.uint8), []) is person
    assert add_hands(person, np.zeros(SHAPE, np.uint8), [make_hand()], strength=0) is person


# ---- HandTracker ------------------------------------------------------------------

class Landmarker:
    def __init__(self):
        self.script = []

    def detect_for_video(self, image, ts):
        hands = self.script.pop(0) if self.script else []
        return SimpleNamespace(hand_landmarks=[[SimpleNamespace(x=x / 64, y=y / 48) for x, y in h] for h in hands])


@pytest.fixture
def tracker(monkeypatch):
    clock = SimpleNamespace(t=50.0)
    monkeypatch.setattr(hands_module.time, "perf_counter", lambda: clock.t)
    t = HandTracker.__new__(HandTracker)
    t._landmarker, t.detect_width, t._t0 = Landmarker(), 640, 50.0
    t.hands, t.presence, t._presence = [], 0.0, Presence(hold=0.15, fade_in=0.05, fade_out=0.25)
    t._detections, t._last_hands = [], []
    t.clock = clock
    return t


def hand_at(x, y=24.0, size=5.0):
    return make_hand(x, y, size)


def step(tracker, dt, hands):
    tracker.clock.t += dt
    tracker._landmarker.script = [hands] if hands else []
    return tracker.update(np.zeros((48, 64, 3), np.uint8))


def test_tracker_reports_detected_hands_in_pixels(tracker):
    got = step(tracker, 0.05, [hand_at(30)])
    assert len(got) == 1 and got[0].shape == (21, 2)
    assert got[0][0, 0] == pytest.approx(30, abs=0.01)


def test_hands_are_held_briefly_then_fade_and_clear(tracker):
    for _ in range(4):
        step(tracker, 0.05, [hand_at(30)])
    assert tracker.presence == 1.0
    step(tracker, 0.05, [])
    assert len(tracker.hands) == 1 and tracker.presence == 1.0      # held through a short dropout
    for _ in range(12):
        step(tracker, 0.05, [])
    assert tracker.hands == [] and tracker.presence == 0.0


def test_predict_moves_each_hand_along_its_own_velocity(tracker):
    step(tracker, 0.1, [hand_at(20), hand_at(44)])
    step(tracker, 0.1, [hand_at(22), hand_at(40)])                  # left +20 px/s, right -40 px/s
    tracker.clock.t += 0.05
    moved = tracker.predict()
    right = max(moved, key=lambda h: h[0, 0]); left = min(moved, key=lambda h: h[0, 0])
    assert left[0, 0] == pytest.approx(22 + 20 * 0.05, abs=0.05)
    assert right[0, 0] == pytest.approx(40 - 40 * 0.05, abs=0.05)


def test_predict_needs_two_recent_detections(tracker):
    step(tracker, 0.1, [hand_at(20)])
    before = tracker.hands
    assert tracker.predict() is before


def test_predict_ignores_stale_or_unmatched_hands(tracker):
    step(tracker, 0.1, [hand_at(10)])
    step(tracker, 0.1, [hand_at(12)])
    tracker.clock.t += 0.6
    assert tracker.predict() is tracker.hands                      # too old to extrapolate
    step(tracker, 0.1, [hand_at(12)])
    step(tracker, 0.1, [hand_at(300 / 5)])                         # a different hand jumped in far away
    before = [h.copy() for h in tracker.hands]
    tracker.clock.t += 0.05
    after = tracker.predict()
    assert np.allclose(after[0], before[0])                        # no match -> no invented velocity


def test_predicted_shift_is_capped(tracker):
    step(tracker, 0.1, [hand_at(5)])
    step(tracker, 0.1, [hand_at(15)])
    tracker.clock.t += 0.1
    far = tracker.predict()[0][0, 0] - 15
    assert far <= tracker.MAX_SHIFT + 1e-3


def test_hand_tracker_reset_forgets_hands_and_history(tracker):
    step(tracker, 0.1, [hand_at(20)])
    step(tracker, 0.1, [hand_at(22)])
    tracker.reset()
    assert tracker.hands == [] and tracker.presence == 0.0 and tracker._detections == []
    assert tracker.predict() == []
