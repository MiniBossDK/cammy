"""All user-adjustable options in one immutable object.

The CLI, presets and the GUI all produce a Settings; the Processor consumes one.
Replacing the whole object (never mutating it) makes it safe to hand from the
GUI thread to the processing thread.
"""
import dataclasses
import re
from dataclasses import dataclass

BACKGROUND_MODES = ("off", "blur", "image", "color")

# Capture sizes the app offers; the camera must support them (MJPG, 30 fps).
RESOLUTIONS = {"720p": (1280, 720), "1080p": (1920, 1080)}

# How you like your camera, not part of a look: presets neither store nor change these.
CAMERA_FIELDS = ("mirror", "resolution")

# (min, max) for every numeric field; the GUI builds its sliders from this.
RANGES = {
    "ai_look": (0.0, 1.0),
    "temporal_denoise": (0.0, 1.0),
    "white_balance": (0.0, 1.0),
    "face_exposure": (0.0, 1.0),
    "brightness": (-50.0, 50.0),
    "contrast": (0.7, 1.5),
    "skin_smooth": (0.0, 1.0),
    "eyes": (0.0, 1.0),
    "teeth": (0.0, 1.0),
    "bg_blur": (0.0, 0.08),
    "sharpen": (0.0, 1.5),
}

_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


@dataclass(frozen=True)
class Settings:
    mirror: bool = False             # flip the camera picture left-right (a camera preference, not part of a preset)
    resolution: str = "720p"         # capture size, one of RESOLUTIONS (a camera preference)
    temporal_denoise: float = 0.0
    white_balance: float = 0.0
    face_exposure: float = 0.0
    denoise: bool = False
    brightness: float = 0.0
    contrast: float = 1.0
    skin_smooth: float = 0.0
    ai_look: float = 0.0             # AI enhancement of the face (needs ncnn; off by default)
    eyes: float = 0.0
    teeth: float = 0.0
    sharpen: float = 0.0
    background: str = "off"          # one of BACKGROUND_MODES
    bg_blur: float = 0.03            # blur sigma as a fraction of frame width
    bg_image: str = ""               # path, used when background == "image"
    bg_color: str = "#00ff00"        # "#rrggbb", used when background == "color"
    keep_hands: bool = True          # find hands and fingers so the background effect keeps them

    def replace(self, **changes):
        return dataclasses.replace(self, **changes).normalized()

    def normalized(self):
        """Clamp numbers into RANGES and fall back to defaults for bad values."""
        values = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            try:
                if f.name in RANGES:
                    lo, hi = RANGES[f.name]
                    value = min(max(float(value), lo), hi)
                elif isinstance(f.default, bool):
                    value = bool(value)
                else:
                    value = str(value)
            except (TypeError, ValueError):
                value = f.default
            values[f.name] = value
        if values["background"] not in BACKGROUND_MODES:
            values["background"] = "off"
        if values["resolution"] not in RESOLUTIONS:
            values["resolution"] = "720p"
        if not _COLOR.match(values["bg_color"]):
            values["bg_color"] = Settings.bg_color
        return Settings(**values)

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data):
        """Build from a dict, ignoring unknown keys and repairing bad values."""
        known = {f.name for f in dataclasses.fields(cls)}
        picked = {k: v for k, v in (data or {}).items() if k in known}
        return cls(**picked).normalized()

    @property
    def size(self):
        """(width, height) to capture at."""
        return RESOLUTIONS[self.resolution]

    def without_camera(self):
        """These settings with the camera preferences (mirror, resolution) at their defaults."""
        return self.replace(**{name: getattr(Settings(), name) for name in CAMERA_FIELDS})

    def keeping_camera_of(self, other):
        """These settings, but with `other`'s camera preferences."""
        return self.replace(**{name: getattr(other, name) for name in CAMERA_FIELDS})

    @property
    def wants_face(self):
        return bool(self.face_exposure or self.skin_smooth or self.ai_look or self.eyes or self.teeth)

    @property
    def wants_background(self):
        return self.background != "off"

    @property
    def wants_hands(self):
        return self.wants_background and self.keep_hands

    @classmethod
    def from_args(cls, args):
        """Build from the CLI namespace (flags are documented in main.py)."""
        if args.bg_blur:
            mode, blur, image, color = "blur", args.bg_blur, "", cls.bg_color
        elif args.bg_image:
            mode, blur, image, color = "image", cls.bg_blur, str(args.bg_image), cls.bg_color
        elif args.bg_color:
            b, g, r = args.bg_color
            mode, blur, image, color = "color", cls.bg_blur, "", f"#{r:02x}{g:02x}{b:02x}"
        else:
            mode, blur, image, color = "off", cls.bg_blur, "", cls.bg_color
        return cls(
            mirror=args.mirror, resolution=args.resolution, temporal_denoise=args.temporal_denoise,
            white_balance=args.white_balance, face_exposure=args.face_exposure,
            denoise=args.denoise, brightness=args.brightness, contrast=args.contrast,
            skin_smooth=args.skin_smooth, ai_look=args.ai_look, eyes=args.eyes, teeth=args.teeth,
            sharpen=args.sharpen, background=mode, bg_blur=blur, bg_image=image,
            bg_color=color, keep_hands=args.keep_hands,
        ).normalized()
