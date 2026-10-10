# Cammy

Make your webcam look good, in real time.

## Run

    python3 -m venv --system-site-packages .venv
    .venv/bin/pip install -r requirements.txt
    .venv/bin/python -m cammy --temporal-denoise 0.7 --white-balance 1 --face-exposure 1 --denoise \
        --skin-smooth 0.5 --eyes 0.3 --teeth 0.5 --sharpen 0.4 --bg-blur 0.03

Face and background effects download their MediaPipe models (about 4 MB) into
`models/` on first use.

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
