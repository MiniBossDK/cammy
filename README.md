# Cammy

Make your webcam look good, in real time.

## Install

Linux only (it uses V4L2 for the camera and v4l2loopback for the virtual camera), Python 3.12.

    pipx install git+https://github.com/MiniBossDK/cammy        # or a built wheel: pipx install cammy-*.whl

or, in a virtual environment: `pip install .` (add `[dev]` for the tests).

**Optional AI look:** add `ncnn` *without its dependencies*:
`pipx runpip cammy install --no-deps ncnn` (or `pip install --no-deps ncnn` in a venv). Do not use a plain
`pip install ncnn`: it declares `opencv-python`, which installs a second copy of OpenCV that clashes
with the `opencv-contrib-python` MediaPipe requires (they share one `cv2` folder, so removing or upgrading
either one can break the other). ncnn needs nothing else for what Cammy uses.

| Command | What it does |
|---|---|
| `cammy-gui` | the control window: live preview, sliders, presets, virtual camera switch |
| `cammy` | the command-line version with a plain preview window (`cammy --help` lists every option) |
| `cammy-setup-camera` | loads the `v4l2loopback` module so other apps can see Cammy (asks for sudo) |
| `cammy-install-desktop` | adds Cammy to your application menu (`--uninstall` removes it) |

Where things go: settings and presets in `~/.config/cammy/`, downloaded models (about 15 MB, and a
one-time 47 MB archive for the AI look) in `~/.local/share/cammy/models/` (override with
`CAMMY_MODELS_DIR`). Nothing is written next to the installed package.

Tested with Python 3.12, numpy 2.5, OpenCV 5.0, MediaPipe 1.1, PySide6 6.12, pyvirtualcam 0.15 and
ncnn 1.0.20260526 (all installed fresh from PyPI into an isolated pipx environment). Older versions are untested, and the minimums in `pyproject.toml` are those versions.

For development: `python3 -m venv .venv && .venv/bin/pip install -e ".[ai,dev]"`, then
`.venv/bin/python -m pytest` (set `QT_QPA_PLATFORM=offscreen` on a machine without a display).
To build the wheel and source package: `pip install build && python -m build` (output in `dist/`).
The version lives in `cammy/__init__.py`.

## Use

    cammy-gui
    cammy --temporal-denoise 0.7 --white-balance 1 --face-exposure 1 --sharpen 0.8   # command line

The face, person and hand models download on first use.

Capture size: `--resolution 1080p` (or the **Camera resolution** box) uses the
camera's full 1920x1080 mode, which shows much more detail such as text, at the
cost of more processing (about 25 ms per frame with typical settings, more with
heavy ones; the status line shows the real figure). The virtual camera follows the
size; a meeting app that already has it open may need restarting to see the new
size.

Mirror the picture left-right with `--mirror` or the **Mirror image** checkbox. It
flips what the camera sees before anything else, so the preview and the virtual
camera are both mirrored, while a background image you picked is not. Mirror is a
camera preference, so presets neither store nor change it.

Background options (pick one): `--bg-blur 0.03`, `--bg-image photo.jpg`,
`--bg-color "#00ff00"`.

Hands and fingers are tracked so the background effect keeps them sharp (the
person model alone loses thin fingers and smears moving hands). Turn it off
with `--no-hands` or the "Keep hands and fingers sharp" checkbox.

Face, hand and person detection run in background threads: a frame never waits
for them, it uses the newest finished result moved forward to the present by
the tracked velocity. `--detect-every N` throttles face and hand detection on a
slow machine.

## Recommended setup

Start from the **Recommended** preset (it is what new users get). For calls, **Meeting** is the
same look plus a light background blur. The values come from measurements on a real webcam:

| Setting | Value | Why |
|---|---|---|
| Temporal denoise | 0.7 | removes 35-50% of the flicker in still areas, about 1% ghosting |
| Denoise (spatial) | off | temporal does the work; a bilateral filter softens text |
| Face exposure | 0.8 | follows your face and keeps its lift when the face is covered |
| White balance | 0.5 | gray-world at half strength avoids colour shifts from a large coloured shirt or wall |
| Skin smoothing / eyes / teeth | 0.3 / 0.1 / 0.2 | light: heavy settings look plastic and shimmer |
| Brightness / contrast | 0 / 1.0 | pushing either crushes shadows, which only adds noise |
| Sharpen | 0.8 | edge-aware: crisper text with no extra grain |
| Background | off | see "showing a document" below |

On the same recording, Recommended raised your face from 45 to 85 (of 255) and cut fine-texture
flicker by more than half relative to the raw camera, while keeping skin texture; the old heavy
settings flicker less only because they flatten the face.

Things a preset cannot hold (they are about your camera and room):

- **Light.** Light on your face from the front and no bright window behind you is the biggest
  single improvement: raw face brightness went from 6-12 to 52-62 just by changing the lighting.
- **Resolution.** Use 720p normally (16 ms per frame with Recommended). Use 1080p when you show
  text or documents (about 31 ms: right at the limit, so avoid heavy presets and the background
  blur there; Meeting is too heavy at 1080p).
- **The camera's own controls.** Leave `sharpness` at its default 50 (80 made it worse at 1080p),
  leave exposure on auto (manual exposure drops the frame rate), and ignore the 50/60 Hz setting
  unless you see banding. The lens is fixed-focus, so a page held too close stays soft.
- **Mirror** is a personal choice: it flips the virtual camera too.

**Showing a document.** The person model treats a held-up page as background, so the background
blur smears it (in testing only 13-34% of the text area stayed sharp). Use Recommended (no background
effect) or turn the background off while you show it.

**The infrared camera.** The webcam has an IR sensor (`/dev/video2`, 640x360 greyscale). Its emitter
strobes on alternate frames, so there are only about 7 usable lit frames per second, and fast hands
smear in them. It is a good depth cue (nearby things are much brighter) and can be mapped onto the
colour image with a simple shift and scale (1.4 px error). In a test it kept a held page about twice
as sharp under background blur (page detail kept 0.35 -> 0.65 with the person model alone vs. with IR),
at the cost of letting some of the far wall through as "near". It is not used by the app today.

## AI look (optional, off by default)

`--ai-look 0.7` or the **AI look** slider (preset **AI look**) enhances your face with Real-ESRGAN
running on the GPU: cleaner, smoother skin and crisper edges (it removes the speckle on glasses
frames and brows). Eyes, brows and lips stay live, so blinks and speech are never delayed.

- **It redraws fine detail.** On a photographed page it turned "appreciate" into "approciate", so do
  not use it while showing text or documents; it is off in every other preset.
- **Needs** `ncnn` (installed with `--no-deps`, see Install) and a Vulkan GPU (CPU works but is slow). The model (1 MB) is
  downloaded once from the official Real-ESRGAN release, checked against a pinned SHA-256.
- **Cost:** about 4 ms per frame on the main thread; the model itself uses the GPU in its own
  process (it has to: ncnn's Python binding holds the interpreter lock for a whole inference, which
  stalled the rest of the app by ~22 ms when run on a thread).
- Scripts that use `Processor` with the AI look must have the usual `if __name__ == "__main__":`
  guard, because the model's process re-imports the main module. `cammy` and `cammy-gui` already do.

## Control window

    cammy-gui                              # same as: cammy --gui

Live preview with sliders for every effect, a background picker (blur, image or
colour), and a virtual camera switch. Presets: Off, Natural, Polished and
Meeting are built in; **Save...** stores your own in `~/.config/cammy/`. Your
last settings come back next time. Moving a slider after picking a preset
shows "Custom".

## Virtual camera (Linux)

Send the processed feed to Zoom, Meet, OBS and other apps:

    cammy-setup-camera                     # once per boot; asks for sudo
    cammy --virtual-camera --virtual-device /dev/video10 \
        --white-balance 1 --face-exposure 1 --skin-smooth 0.5 --bg-blur 0.03

Then pick the **Cammy** camera in the other app. Add `--no-preview` to run
without the window (stop with Ctrl+C). The FPS counter is drawn on the preview
only, never on the virtual camera feed. Only one app should open your real
webcam at a time, so let Cammy have it and use the Cammy camera everywhere else.

## Tests

    .venv/bin/python -m pytest -q
