"""The mask background effects use: the person mask plus the hands."""
from .hands import add_hands


class Cutout:
    """Combines the person segmenter's mask with hand masks into one `mask`.

    Background effects read `cutout.mask`. Call refresh(frame) once per frame
    after the segmenter and (optional) hand tracker have been updated.
    """

    def __init__(self, segmenter, hand_tracker=None):
        self.segmenter = segmenter
        self.hand_tracker = hand_tracker
        self.mask = None

    def refresh(self, frame):
        person = self.segmenter.mask
        if person is None:
            self.mask = None
        elif self.hand_tracker is not None and self.hand_tracker.hands:
            self.mask = add_hands(person, frame, self.hand_tracker.hands, self.hand_tracker.presence)
        else:
            self.mask = person
        return self.mask
