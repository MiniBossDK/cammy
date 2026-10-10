"""Virtual camera output, so other apps (Zoom, Meet, OBS) see the processed feed."""
import cv2

SETUP_HINT = (
    "Could not open a virtual camera: {error}\n"
    "On Linux this needs the v4l2loopback kernel module. Load it with:\n"
    "    cammy-setup-camera\n"
    "then run again (use --virtual-device /dev/videoN to pick a specific device)."
)


class VirtualCameraError(RuntimeError):
    pass


class VirtualCamera:
    """Sends BGR frames to a virtual camera device.

    The device opens on the first frame, sized to that frame; later frames of a
    different size are resized to match, since consumers expect a fixed size.
    `camera_factory` defaults to pyvirtualcam.Camera and exists so tests can
    run without a loopback device.
    """

    def __init__(self, fps=30, device=None, camera_factory=None):
        self.fps = fps
        self.device = device
        self._factory = camera_factory
        self._cam = None
        self._size = None

    def _open(self, width, height):
        import pyvirtualcam

        factory = self._factory or pyvirtualcam.Camera
        try:
            self._cam = factory(width, height, self.fps,
                                fmt=pyvirtualcam.PixelFormat.BGR, device=self.device)
        except (RuntimeError, OSError) as error:
            raise VirtualCameraError(SETUP_HINT.format(error=error)) from error
        self._size = (width, height)

    @property
    def device_name(self):
        return getattr(self._cam, "device", None)

    def send(self, frame):
        if self._cam is None:
            self._open(frame.shape[1], frame.shape[0])
        if (frame.shape[1], frame.shape[0]) != self._size:
            frame = cv2.resize(frame, self._size, interpolation=cv2.INTER_AREA)
        self._cam.send(frame)

    def close(self):
        if self._cam is not None:
            self._cam.close()
            self._cam = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
