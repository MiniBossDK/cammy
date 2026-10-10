# Cammy

Make your webcam look good, in real time.

## Run

    python3 -m venv --system-site-packages .venv
    .venv/bin/pip install -r requirements.txt
    .venv/bin/python -m cammy --temporal-denoise 0.7 --white-balance 1 --face-exposure 1 --denoise \
        --skin-smooth 0.5 --eyes 0.3 --teeth 0.5 --sharpen 0.4 --bg-blur 0.03

Face and background effects download their MediaPipe models (about 4 MB) into
`models/` on first use.

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

## Control window

    .venv/bin/python -m cammy --gui        # same as: python -m cammy.gui

Live preview with sliders for every effect, a background picker (blur, image or
colour), and a virtual camera switch. Presets: Off, Natural, Polished and
Meeting are built in; **Save...** stores your own in `~/.config/cammy/`. Your
last settings come back next time. Moving a slider after picking a preset
shows "Custom".

## Virtual camera (Linux)

Send the processed feed to Zoom, Meet, OBS and other apps:

    scripts/setup_virtual_camera.sh        # once per boot; asks for sudo
    .venv/bin/python -m cammy --virtual-camera --virtual-device /dev/video10 \
        --white-balance 1 --face-exposure 1 --skin-smooth 0.5 --bg-blur 0.03

Then pick the **Cammy** camera in the other app. Add `--no-preview` to run
without the window (stop with Ctrl+C). The FPS counter is drawn on the preview
only, never on the virtual camera feed. Only one app should open your real
webcam at a time, so let Cammy have it and use the Cammy camera everywhere else.

## Tests

    .venv/bin/python -m pytest -q
