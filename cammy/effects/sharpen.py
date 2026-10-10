"""Edge-aware sharpening for crisp text and detail without amplified noise."""
import cv2
import numpy as np

REFERENCE_WIDTH = 1280      # `radius` is defined at this frame width and scales with the picture


class Sharpen:
    """Sharpen brightness only, only where there are real edges, without halos.

    A plain unsharp mask on every colour channel also boosts sensor noise and
    skin texture, colours the edges, and overshoots into bright halos. This one:

      * works on brightness only and adds the same change to all three channels,
        so colours do not shift (chroma is untouched);
      * ignores detail smaller than the noise floor and only boosts what is above
        it (text strokes, edges): flat areas and skin stay as they were. The
        floor follows the measured sensor noise (about 3.5 sigma, smoothed over
        time, never below `coring`), so a dim, noisy room is not "sharpened"
        into grain;
      * limits the result to the range of the pixel's 3x3 neighbourhood plus
        `overshoot` levels, which allows crisp, slightly deeper strokes but no
        bright rims or ringing around them.

    `amount` 0 is identity; the slider scale is 0..1.5 with 1.0 a strong, clean
    setting (internally `gain` = 2 x amount). The blur radius scales with the
    frame width, so a 1080p picture is sharpened as much as a 720p one.
    """

    GAIN_PER_AMOUNT = 2.0
    NOISE_SIGMAS = 3.5          # noise floor = this many estimated sigmas of the fine detail
    MAX_CORING = 14

    def __init__(self, amount=0.5, radius=1.5, coring=5, overshoot=14, smoothing=0.9,
                 reference_width=REFERENCE_WIDTH):
        self.amount = amount
        self.radius = radius
        self.reference_width = reference_width
        self.coring = int(coring)           # lowest noise floor, in brightness levels
        self.overshoot = int(overshoot)
        self.smoothing = smoothing
        self._sigma = None                  # smoothed noise estimate

    def radius_for(self, width):
        """Blur radius in pixels for a frame `width` wide."""
        return max(self.radius * width / self.reference_width, 0.5)

    # For Gaussian noise of std s, the q-th percentile of |x| is s * sqrt(2) * erfinv(q); at the 20th
    # percentile that is 0.2533 s. A low percentile reads the flattest part of the picture, so a frame
    # that is mostly text edges (a page filling the view) is not mistaken for noise.
    NOISE_PERCENTILE = 20
    NOISE_PERCENTILE_TO_SIGMA = 1 / 0.2533

    def _noise_floor(self, detail):
        """Noise floor in levels, from the flat areas of a subsample of the detail."""
        sample = np.abs(detail[::7, ::7]).astype(np.float32)
        sigma = float(np.percentile(sample, self.NOISE_PERCENTILE)) * self.NOISE_PERCENTILE_TO_SIGMA
        self._sigma = sigma if self._sigma is None else self.smoothing * self._sigma + (1 - self.smoothing) * sigma
        return int(np.clip(round(self._sigma * self.NOISE_SIGMAS), self.coring, self.MAX_CORING))

    def __call__(self, frame):
        if self.amount <= 0:
            return frame
        gain = self.amount * self.GAIN_PER_AMOUNT
        radius = self.radius_for(frame.shape[1])

        luma = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(luma, (0, 0), radius)
        detail = cv2.subtract(luma, blurred, dtype=cv2.CV_16S)            # signed: edge detail
        coring = self._noise_floor(detail)
        # Soft threshold: drop |detail| <= coring (noise), keep the rest minus coring.
        kept = cv2.subtract(detail, cv2.min(cv2.max(detail, -coring), coring))
        sharpened = cv2.addWeighted(luma, 1.0, kept, gain, 0, dtype=cv2.CV_16S)

        kernel = np.ones((3, 3), np.uint8)
        low = cv2.subtract(cv2.erode(luma, kernel), self.overshoot, dtype=cv2.CV_16S)
        high = cv2.add(cv2.dilate(luma, kernel), self.overshoot, dtype=cv2.CV_16S)
        result = cv2.min(cv2.max(sharpened, low), high)                   # now within the 0..255 neighbourhood range
        result = np.clip(result, 0, 255)

        change = cv2.subtract(result, luma.astype(np.int16), dtype=cv2.CV_16S)
        return cv2.add(frame, cv2.merge([change, change, change]), dtype=cv2.CV_8U)
