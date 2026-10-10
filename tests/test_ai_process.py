import os
import runpy
import signal
import time

import numpy as np
import pytest

from cammy.ai import NEEDS_NCNN, UpscalerProcess


class Doubler:
    """Stands in for the model: doubles the picture and remembers how often it was asked."""
    device = 0

    def __init__(self):
        self.calls = 0

    def upscale(self, bgr):
        self.calls += 1
        return np.repeat(np.repeat(bgr, 2, axis=0), 2, axis=1)


def start(make=lambda gpu, directory: Doubler(), **kw):
    kw.setdefault("context", "fork")          # fork keeps the test's own function usable in the child
    return UpscalerProcess(make=make, **kw)


def patch(value=7, size=(6, 8)):
    return np.full((*size, 3), value, np.uint8)


def test_the_picture_goes_through_the_other_process_and_comes_back_doubled():
    up = start()
    try:
        out = up.upscale(patch())
        assert out.shape == (12, 16, 3) and out.dtype == np.uint8 and (out == 7).all()
        assert up.device == 0
    finally:
        up.close()


def test_it_handles_many_requests_and_odd_inputs():
    up = start()
    try:
        for value in range(1, 6):
            assert (up.upscale(patch(value)) == value).all()
        sliced = np.full((20, 30, 3), 9, np.uint8)[::2, ::3]                 # not contiguous
        assert up.upscale(sliced).shape == (sliced.shape[0] * 2, sliced.shape[1] * 2, 3)
    finally:
        up.close()


def test_an_error_inside_the_model_is_raised_and_the_process_keeps_serving():
    class Flaky(Doubler):
        def upscale(self, bgr):
            if bgr[0, 0, 0] == 99:
                raise ValueError("bad patch")
            return super().upscale(bgr)

    up = start(make=lambda gpu, directory: Flaky())
    try:
        with pytest.raises(RuntimeError, match="ValueError: bad patch"):
            up.upscale(patch(99))
        assert (up.upscale(patch(5)) == 5).all()                              # still alive afterwards
    finally:
        up.close()


def test_a_model_that_cannot_start_is_reported_with_its_reason():
    def broken(gpu, directory):
        raise OSError("no such model file")

    with pytest.raises(RuntimeError, match="could not start.*no such model file"):
        start(make=broken)


def test_a_missing_ncnn_becomes_the_install_hint():
    def missing(gpu, directory):
        raise ImportError("No module named 'ncnn'")

    with pytest.raises(ImportError) as info:
        start(make=missing)
    assert str(info.value) == NEEDS_NCNN


def test_a_model_that_never_starts_times_out_and_is_cleaned_up():
    def slow(gpu, directory):
        time.sleep(30)

    began = time.perf_counter()
    with pytest.raises(RuntimeError, match="did not start"):
        start(make=slow, start_timeout=0.3)
    assert time.perf_counter() - began < 10                                   # the child was terminated, not waited for


def test_a_model_that_stops_answering_times_out():
    class Hangs(Doubler):
        def upscale(self, bgr):
            time.sleep(30)

    up = start(make=lambda gpu, directory: Hangs(), reply_timeout=0.3)
    with pytest.raises(RuntimeError, match="did not answer"):
        up.upscale(patch())
    assert not up._process.is_alive()


def test_a_killed_process_is_reported_not_hung():
    up = start()
    os.kill(up._process.pid, signal.SIGKILL)
    up._process.join(timeout=5)
    with pytest.raises(RuntimeError, match="stopped unexpectedly"):
        up.upscale(patch())
    up.close()


def test_close_stops_the_process_and_can_be_repeated():
    up = start()
    pid_process = up._process
    up.close()
    assert not pid_process.is_alive()
    up.close()


# ---- the entry point must not start the app again in a child ----------------------

def test_a_child_process_importing_the_entry_module_does_not_start_the_app(monkeypatch):
    calls = []
    monkeypatch.setattr("cammy.main.main", lambda: calls.append("started"))
    runpy.run_module("cammy.__main__", run_name="__mp_main__")                # what a spawned child does
    assert calls == []
    runpy.run_module("cammy.__main__", run_name="__main__")                   # what `python -m cammy` does
    assert calls == ["started"]
