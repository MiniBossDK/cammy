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
- [ ] M5: virtual camera output
- [ ] M6: UI with sliders and presets
- [ ] M7: performance and packaging

## Tests

    .venv/bin/python -m pytest -q
