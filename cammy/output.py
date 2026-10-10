"""Virtual camera output, so other apps (Zoom, Meet, OBS) see the processed feed."""
import fcntl
import os
import struct
import time

import cv2
import numpy as np

SETUP_HINT = (
    "Could not open a virtual camera: {error}\n"
    "On Linux this needs the v4l2loopback kernel module. Load it with:\n"
    "    cammy-setup-camera\n"
    "then run again (use --virtual-device /dev/videoN to pick a specific device)."
)

IN_USE_HINT = (
    "The virtual camera {device} is already in use at {actual} by another program (probably your meeting "
    "app), so it cannot be switched to {wanted}: sending that size would scramble the picture. Cammy keeps "
    "sending {actual}. To change the size, stop the camera in that app (leave the call's camera off, or close "
    "the tab) and pick the size again."
)

VIDIOC_G_FMT = 0xC0D05604              # _IOWR('V', 4, struct v4l2_format): 208 bytes on 64-bit Linux
V4L2_BUF_TYPE_VIDEO_CAPTURE = 1


class VirtualCameraError(RuntimeError):
    """The virtual camera could not start. `actual` is the (width, height) another program holds it at, if that is why."""

    def __init__(self, message, actual=None):
        super().__init__(message)
        self.actual = actual


def device_format(path):
    """(width, height) the loopback device currently shows to readers, or None if it cannot be read.

    A device that no program is writing to yet has no capture format, and neither does something that
    is not a camera: both give None.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except (OSError, TypeError):
        return None
    try:
        buf = bytearray(208)
        struct.pack_into("I", buf, 0, V4L2_BUF_TYPE_VIDEO_CAPTURE)
        fcntl.ioctl(fd, VIDIOC_G_FMT, buf)
        width, height = struct.unpack_from("II", buf, 8)       # struct v4l2_pix_format starts at offset 8
        return (width, height) if width and height else None
    except OSError:
        return None
    finally:
        os.close(fd)


class VirtualCamera:
    """Sends BGR frames to a virtual camera device at one fixed size.

    `size` is the output size (default: the first frame's). Frames of any other size are scaled to it
    with area averaging, so the capture resolution can change freely without the device noticing.
    That matters: v4l2loopback cannot change its format while a program (a meeting app) is reading it,
    and the library does not notice; writing frames of a new size then gives a scrambled picture.
    So after opening, the device's real format is checked, and a mismatch is an error.

    `camera_factory` defaults to pyvirtualcam.Camera and `format_reader` to `device_format`; both
    exist so tests can run without a loopback device.
    """

    def __init__(self, fps=30, device=None, size=None, camera_factory=None, format_reader=device_format):
        self.fps = fps
        self.device = device
        self.size = tuple(size) if size else None
        self._factory = camera_factory
        self._format_reader = format_reader
        self._cam = None

    def _open(self, width, height):
        import pyvirtualcam

        factory = self._factory or pyvirtualcam.Camera
        try:
            cam = factory(width, height, self.fps, fmt=pyvirtualcam.PixelFormat.BGR, device=self.device)
        except (RuntimeError, OSError) as error:
            raise VirtualCameraError(SETUP_HINT.format(error=error)) from error
        path = getattr(cam, "device", None)
        if path and self._format_reader:
            self._verify_size(cam, path, width, height)
        self._cam, self.size = cam, (width, height)

    def _verify_size(self, cam, path, width, height):
        """Make sure the device really takes frames of this size, or raise.

        The device has no readable format until the first frame is written, and a program that already
        reads it (a meeting app) pins the size. So write one black frame first: if the size is wrong the
        truncated frame is still just black, and nothing visible can be scrambled.
        """
        cam.send(np.zeros((height, width, 3), np.uint8))
        actual = None
        for _ in range(10):
            actual = self._format_reader(path)
            if actual is not None:
                break
            time.sleep(0.01)
        if actual is not None and actual != (width, height):
            cam.close()
            raise VirtualCameraError(IN_USE_HINT.format(
                device=path, actual="x".join(map(str, actual)), wanted=f"{width}x{height}"), actual=actual)

    @property
    def device_name(self):
        return getattr(self._cam, "device", None)

    @property
    def is_open(self):
        return self._cam is not None

    def send(self, frame):
        if self._cam is None:
            self._open(*(self.size or (frame.shape[1], frame.shape[0])))
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        self._cam.send(frame)

    def close(self):
        if self._cam is not None:
            self._cam.close()
            self._cam = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
