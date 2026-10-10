"""Built-in and user presets, plus remembering the last-used settings."""
import json
import os
from pathlib import Path

from .settings import Settings

# Chosen from measurements on a real webcam (see the README, "Recommended setup"):
#  - temporal denoise 0.7: removes 35-50% of the flicker in still areas with ~1% ghosting;
#  - no spatial denoise: temporal does the work, and a bilateral filter softens text;
#  - skin smoothing, eyes and teeth kept light (heavy settings look plastic and shimmer);
#  - contrast 1.0 / brightness 0: pushing either crushes shadows, which only adds noise;
#  - sharpen 0.8 on the edge-aware sharpener: crisper text with no extra grain;
#  - no background effect: a held-up page counts as background and would be blurred.
RECOMMENDED = Settings(temporal_denoise=0.7, white_balance=0.5, face_exposure=0.8, skin_smooth=0.3,
                       eyes=0.1, teeth=0.2, sharpen=0.8)

BUILTIN = {
    "Off": Settings(),
    "Recommended": RECOMMENDED,
    "Natural": Settings(temporal_denoise=0.5, white_balance=0.6, face_exposure=0.6, skin_smooth=0.25,
                        eyes=0.15, sharpen=0.4),
    "Polished": Settings(temporal_denoise=0.7, white_balance=0.8, face_exposure=1.0, denoise=True,
                         skin_smooth=0.5, eyes=0.3, teeth=0.4, contrast=1.05,
                         sharpen=0.7),
    # For calls: the recommended look plus a light background blur (and hands kept sharp).
    "Meeting": RECOMMENDED.replace(background="blur", bg_blur=0.03),
}


def starting_settings(store):
    """What to start with: the last-used settings, else the recommended look."""
    return store.load_last() or RECOMMENDED


def default_dir():
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "cammy"


class PresetStore:
    """User presets live in <dir>/presets.json, last-used settings in last.json.

    Built-in presets cannot be overwritten or deleted. Unreadable files are
    treated as empty, so a corrupt file never stops the app from starting.
    """

    def __init__(self, directory=None):
        self.directory = Path(directory) if directory else default_dir()

    @property
    def _presets_path(self):
        return self.directory / "presets.json"

    @property
    def _last_path(self):
        return self.directory / "last.json"

    @staticmethod
    def _read(path):
        try:
            data = json.loads(path.read_text())
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write(self, path, data):
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)

    def user_presets(self):
        return {name: Settings.from_dict(d) for name, d in self._read(self._presets_path).items()
                if isinstance(d, dict)}

    def all(self):
        return {**BUILTIN, **self.user_presets()}

    def names(self):
        return list(BUILTIN) + sorted(self.user_presets())

    def get(self, name):
        return self.all().get(name)

    def save(self, name, settings):
        name = name.strip()
        if not name:
            raise ValueError("preset name is empty")
        if name in BUILTIN:
            raise ValueError(f"'{name}' is a built-in preset; pick another name")
        data = self._read(self._presets_path)
        data[name] = settings.without_camera().to_dict()           # camera choices are not part of a look
        self._write(self._presets_path, data)

    def delete(self, name):
        if name in BUILTIN:
            raise ValueError(f"'{name}' is a built-in preset and cannot be deleted")
        data = self._read(self._presets_path)
        if data.pop(name, None) is not None:
            self._write(self._presets_path, data)

    def load_last(self):
        data = self._read(self._last_path)
        return Settings.from_dict(data) if data else None

    def save_last(self, settings):
        self._write(self._last_path, settings.to_dict())

    def name_of(self, settings):
        """Name of a preset equal to `settings` (ignoring camera choices), else None."""
        looks = settings.without_camera()
        for name, preset in self.all().items():
            if preset.without_camera() == looks:
                return name
        return None
