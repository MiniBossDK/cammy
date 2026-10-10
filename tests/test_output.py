import numpy as np
import pytest

from cammy.output import VirtualCamera, VirtualCameraError


class FakeCam:
    instances = []

    def __init__(self, width, height, fps, fmt=None, device=None):
        self.args = (width, height, fps, device)
        self.fmt = fmt
        self.frames = []
        self.closed = False
        FakeCam.instances.append(self)

    def send(self, frame):
        self.frames.append(frame.copy())

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def reset():
    FakeCam.instances.clear()


def frame(w=64, h=48, value=10):
    return np.full((h, w, 3), value, np.uint8)


def test_opens_lazily_sized_to_first_frame():
    cam = VirtualCamera(fps=30, device="/dev/video10", camera_factory=FakeCam)
    assert not FakeCam.instances
    cam.send(frame())
    assert FakeCam.instances[0].args == (64, 48, 30, "/dev/video10")


def test_sends_bgr_frames_unchanged():
    import pyvirtualcam

    cam = VirtualCamera(camera_factory=FakeCam)
    f = frame(value=77)
    cam.send(f)
    fake = FakeCam.instances[0]
    assert fake.fmt == pyvirtualcam.PixelFormat.BGR
    assert (fake.frames[0] == f).all()


def test_resizes_mismatched_frames():
    cam = VirtualCamera(camera_factory=FakeCam)
    cam.send(frame(64, 48))
    cam.send(frame(128, 96))
    assert FakeCam.instances[0].frames[1].shape == (48, 64, 3)
    assert len(FakeCam.instances) == 1


def test_close_closes_device_and_is_safe_twice():
    with VirtualCamera(camera_factory=FakeCam) as cam:
        cam.send(frame())
    assert FakeCam.instances[0].closed
    cam.close()


def test_close_without_frames_is_safe():
    VirtualCamera(camera_factory=FakeCam).close()
    assert not FakeCam.instances


def test_open_failure_gives_setup_hint():
    def broken(*a, **k):
        raise RuntimeError("no device found")

    with pytest.raises(VirtualCameraError) as err:
        VirtualCamera(camera_factory=broken).send(frame())
    assert "v4l2loopback" in str(err.value) and "no device found" in str(err.value)
