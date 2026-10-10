import cv2

from cammy import capture


class FakeVideoCapture:
    def __init__(self, index, backend):
        self.sets = []

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.sets.append((prop, value))

    def release(self):
        pass


def make(monkeypatch, **kw):
    monkeypatch.setattr(cv2, "VideoCapture", FakeVideoCapture)
    return capture.Capture(**kw).cap.sets


def test_requests_mjpg_before_size(monkeypatch):
    sets = make(monkeypatch)
    props = [p for p, _ in sets]
    assert props[0] == cv2.CAP_PROP_FOURCC
    assert sets[0][1] == cv2.VideoWriter_fourcc(*"MJPG")
    assert props.index(cv2.CAP_PROP_FRAME_WIDTH) > 0


def test_mjpg_can_be_disabled(monkeypatch):
    props = [p for p, _ in make(monkeypatch, mjpg=False)]
    assert cv2.CAP_PROP_FOURCC not in props
