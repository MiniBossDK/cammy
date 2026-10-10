"""Noise reduction."""
import cv2


class Denoise:
    """Edge-preserving bilateral filter. Cheap enough for real time."""

    def __init__(self, diameter=5, sigma=40):
        self.diameter = diameter
        self.sigma = sigma

    def __call__(self, frame):
        return cv2.bilateralFilter(frame, self.diameter, self.sigma, self.sigma)
