"""Effect pipeline: applies effects to a frame in order."""


class Pipeline:
    def __init__(self, effects=None):
        self.effects = list(effects or [])

    def add(self, effect):
        self.effects.append(effect)

    def __call__(self, frame):
        for effect in self.effects:
            frame = effect(frame)
        return frame
