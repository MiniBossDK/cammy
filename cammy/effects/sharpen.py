"""Unsharp-mask sharpening."""
import cv2


class Sharpen:
    """out = frame + amount * (frame - blurred). amount 0 is identity."""

    def __init__(self, amount=0.5, radius=1.5):
        self.amount = amount
        self.radius = radius

    def __call__(self, frame):
        blurred = cv2.GaussianBlur(frame, (0, 0), self.radius)
        return cv2.addWeighted(frame, 1 + self.amount, blurred, -self.amount, 0)
