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


# ---- fixed output size and the device-format check --------------------------------------

import struct  # noqa: E402

import cammy.output as output  # noqa: E402
from cammy.output import IN_USE_HINT, device_format  # noqa: E402


class DeviceCam(FakeCam):
    """A fake camera that reports which device it opened, like pyvirtualcam does."""

    def __init__(self, width, height, fps, fmt=None, device=None):
        super().__init__(width, height, fps, fmt, device)
        self.device = "/dev/video10"


def test_the_output_size_is_fixed_whatever_size_the_frames_are():
    cam = VirtualCamera(size=(64, 48), camera_factory=FakeCam)
    cam.send(frame(128, 96))                       # a bigger capture: scaled down, device opens at 64x48
    cam.send(frame(32, 24))                        # a smaller one: scaled up
    fake = FakeCam.instances[0]
    assert fake.args[:2] == (64, 48) and len(FakeCam.instances) == 1
    assert all(f.shape == (48, 64, 3) for f in fake.frames)


def test_a_device_in_use_at_another_size_is_refused_not_written_to():
    seen = []
    cam = VirtualCamera(size=(1920, 1080), camera_factory=DeviceCam, format_reader=lambda path: seen.append(path) or (1280, 720))
    with pytest.raises(VirtualCameraError) as error:
        cam.send(frame(1920, 1080))
    assert seen == ["/dev/video10"]
    text = str(error.value)
    assert "1280x720" in text and "1920x1080" in text and "another program" in text
    fake = FakeCam.instances[0]
    assert fake.closed and not cam.is_open
    assert len(fake.frames) == 1 and fake.frames[0].shape == (1080, 1920, 3) and not fake.frames[0].any()   # only a black probe was written


def test_a_device_at_the_same_size_is_fine():
    cam = VirtualCamera(size=(1280, 720), camera_factory=DeviceCam, format_reader=lambda path: (1280, 720))
    cam.send(frame(1280, 720, value=10))
    frames = FakeCam.instances[0].frames
    assert cam.is_open and len(frames) == 2                      # the black probe, then the real frame
    assert not frames[0].any() and frames[1].mean() == 10


def test_an_unreadable_format_does_not_block_streaming(monkeypatch):
    sleeps = []
    monkeypatch.setattr(output.time, "sleep", sleeps.append)
    cam = VirtualCamera(size=(64, 48), camera_factory=DeviceCam, format_reader=lambda path: None)
    cam.send(frame(64, 48))
    assert cam.is_open and len(sleeps) == 10                      # it retried briefly, then carried on


def test_the_check_is_skipped_when_the_camera_does_not_name_a_device():
    called = []
    cam = VirtualCamera(size=(64, 48), camera_factory=FakeCam, format_reader=lambda p: called.append(p))
    cam.send(frame(64, 48))
    assert called == [] and cam.is_open


def test_the_hint_explains_what_to_do():
    assert "Cammy keeps sending" in IN_USE_HINT and "leave the call's camera off" in IN_USE_HINT and "pick the size again" in IN_USE_HINT


def test_device_format_reads_the_width_and_height(monkeypatch):
    def fake_ioctl(fd, request, buf):
        assert request == output.VIDIOC_G_FMT and struct.unpack_from("I", buf, 0)[0] == 1     # asks for the capture side
        struct.pack_into("II", buf, 8, 1280, 720)
    monkeypatch.setattr(output.fcntl, "ioctl", fake_ioctl)
    assert device_format("/dev/null") == (1280, 720)


def test_device_format_is_none_when_there_is_nothing_to_read(monkeypatch):
    assert device_format("/dev/definitely-not-a-camera") is None
    assert device_format(None) is None
    assert device_format("/dev/null") is None                                      # not a v4l2 device: the ioctl fails
    monkeypatch.setattr(output.fcntl, "ioctl", lambda fd, request, buf: None)       # idle loopback: width and height 0
    assert device_format("/dev/null") is None


def test_the_format_is_read_after_the_probe_frame_and_retried_until_it_appears(monkeypatch):
    answers = iter([None, None, (64, 48)])
    cam = VirtualCamera(size=(64, 48), camera_factory=DeviceCam, format_reader=lambda path: next(answers))
    monkeypatch.setattr(output.time, "sleep", lambda s: None)
    cam.send(frame(64, 48))
    assert cam.is_open and FakeCam.instances[0].frames[0].shape == (48, 64, 3)


def test_the_error_says_which_size_the_device_is_held_at():
    cam = VirtualCamera(size=(1920, 1080), camera_factory=DeviceCam, format_reader=lambda path: (1280, 720))
    with pytest.raises(VirtualCameraError) as error:
        cam.send(frame(1920, 1080))
    assert error.value.actual == (1280, 720)
    assert VirtualCameraError("plain").actual is None
