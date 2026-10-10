# Cammy

Make your webcam look good, in real time.

## Run

    python3 -m venv --system-site-packages .venv
    .venv/bin/pip install -r requirements.txt
    .venv/bin/python -m cammy --white-balance 1 --face-exposure 1 --denoise \
        --skin-smooth 0.5 --eyes 0.3 --teeth 0.5 --sharpen 0.4 --bg-blur 0.03

Face and background effects download their MediaPipe models (about 4 MB) into
`models/` on first use.

Background options (pick one): `--bg-blur 0.03`, `--bg-image photo.jpg`,
`--bg-color "#00ff00"`. Face and person detection run on alternate frames by
default to stay inside the frame budget; use `--detect-every 1` for every frame.

Press `q` or `Esc` to quit. Run `python3 -m cammy --help` for all options.

## Status

- [x] M1: live preview with FPS counter
- [x] M2: brightness/contrast, white balance, denoise, sharpen
- [x] M3: face-aware effects (exposure, skin smoothing, eye brighten, teeth whiten)
- [x] M4: background blur/replace
- [x] M5: virtual camera output (Linux, v4l2loopback)
- [ ] M6: UI with sliders and presets
- [ ] M7: performance and packaging

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
