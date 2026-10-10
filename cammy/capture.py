"""Webcam capture."""
import cv2


class Capture:
    def __init__(self, index=0, width=1280, height=720, fps=30, mjpg=True):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open webcam at index {index}")
        # Ask for MJPG first: many USB webcams only reach 30 fps at 720p in MJPG
        # and fall back to ~10 fps in raw YUYV, which is what they pick by default.
        if mjpg:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    @property
    def size(self):
        """The (width, height) the camera is actually delivering, which may differ from what was asked."""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        """Return next BGR frame, or None if the read failed."""
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
