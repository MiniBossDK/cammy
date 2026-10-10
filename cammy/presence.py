"""Smooth 'is a subject there' signal, so effects fade instead of snapping."""


class Presence:
    """A value in [0, 1] that follows a noisy yes/no detection over time.

    Detection drops out whenever the subject is covered, turns away or leaves
    the frame; switching effects off at that instant makes the picture jump.
    Instead the value holds at its level for `hold` seconds after the last
    detection, then fades to 0 over `fade_out`, and climbs back over `fade_in`.
    """

    def __init__(self, hold=0.3, fade_in=0.1, fade_out=0.4):
        self.hold = hold
        self.fade_in = fade_in
        self.fade_out = fade_out
        self.value = 0.0
        self._last_seen = None
        self._last_time = None

    def update(self, detected, now):
        dt = 0.0 if self._last_time is None else max(now - self._last_time, 0.0)
        self._last_time = now
        if detected:
            self._last_seen = now
            self.value = 1.0 if self.fade_in <= 0 else min(1.0, self.value + dt / self.fade_in)
        elif self._last_seen is not None and now - self._last_seen > self.hold:
            # Only time past the hold counts toward the fade, however sparse the updates.
            fading = min(dt, now - self._last_seen - self.hold)
            self.value = 0.0 if self.fade_out <= 0 else max(0.0, self.value - fading / self.fade_out)
        return self.value
