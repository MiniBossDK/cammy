"""Frame scheduling for expensive per-frame detectors."""


def due(frame_index, every=1, offset=0):
    """True when a detector running every `every` frames, shifted by `offset`, should run.

    Giving two detectors different offsets spreads their cost over alternate
    frames instead of stacking it on the same one.
    """
    if every <= 1:
        return True
    return frame_index % every == offset % every
