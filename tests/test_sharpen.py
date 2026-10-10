import numpy as np
import pytest

from cammy.effects.sharpen import Sharpen


def text_like(h=120, w=160, seed=0, noise=0.0):
    """Dark vertical strokes on light paper, softened like a slightly out-of-focus lens."""
    import cv2
    img = np.full((h, w), 200, np.float32)
    for x in range(10, w - 10, 20):
        img[20:h - 20, x:x + 6] = 60
    img = cv2.GaussianBlur(img, (0, 0), 1.6)
    if noise:
        img += np.random.default_rng(seed).normal(0, noise, img.shape)
    gray = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.merge([gray, gray, gray])


def edge_slope(img):
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)).max())


def test_amount_zero_is_identity():
    f = text_like()
    assert Sharpen(0)(f) is f


def test_edges_get_steeper():
    f = text_like()
    out = Sharpen(1.0, reference_width=160)(f)
    assert edge_slope(out) > edge_slope(f) * 1.1


def test_flat_areas_and_the_picture_shape_are_untouched():
    f = np.full((60, 80, 3), (90, 130, 170), np.uint8)
    out = Sharpen(1.5)(f)
    assert out.shape == f.shape and out.dtype == np.uint8
    assert (out == f).all()


def test_noise_is_not_amplified():
    f = text_like(noise=2.0)
    fx = Sharpen(1.0, reference_width=160)
    for _ in range(5):                           # let the noise estimate settle
        out = fx(f)
    paper = (slice(5, 15), slice(0, 160))        # blank paper above the strokes
    grain = lambda im: float(np.abs(im[paper][..., 0].astype(float) - im[paper][..., 0].astype(float).mean()).mean())
    assert grain(out) <= grain(f) * 1.1


def dense_strokes(spacing):
    import cv2
    f = np.full((120, 640, 3), 210, np.uint8)
    for x in range(8, 632, spacing):
        f[10:110, x:x + 3] = 50
    return cv2.GaussianBlur(f, (0, 0), 1.2)


def floor_of(f, fx):
    import cv2
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    detail = cv2.subtract(gray, cv2.GaussianBlur(gray, (0, 0), 1.5), dtype=cv2.CV_16S)
    return fx._noise_floor(detail)


def test_a_page_of_text_is_not_mistaken_for_noise():
    """Text lines leave plenty of flat paper, so the floor must stay at its minimum there."""
    fx = Sharpen(1.0, reference_width=640)
    for spacing in (16, 22):
        assert floor_of(dense_strokes(spacing), fx) == fx.coring, spacing


def test_a_pathologically_dense_pattern_only_makes_it_more_cautious():
    """Almost no flat area: the floor may rise (less sharpening) but stays bounded."""
    fx = Sharpen(1.0, reference_width=640)
    floor = floor_of(dense_strokes(9), fx)
    assert fx.coring <= floor <= Sharpen.MAX_CORING


def test_noise_floor_rises_with_noise():
    import cv2
    calm, grainy = text_like(noise=0.5), text_like(noise=6.0)
    floors = []
    for f in (calm, grainy):
        fx = Sharpen(1.0, reference_width=160)
        for _ in range(8):
            fx(f)
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        detail = cv2.subtract(gray, cv2.GaussianBlur(gray, (0, 0), 1.5), dtype=cv2.CV_16S)
        floors.append(fx._noise_floor(detail))
    assert floors[0] == Sharpen().coring            # calm: the minimum floor
    assert floors[1] > floors[0]                    # grainy: raised
    assert floors[1] <= Sharpen.MAX_CORING


def test_no_overshoot_beyond_the_neighbourhood():
    import cv2
    f = text_like()
    fx = Sharpen(1.5, reference_width=160)
    out = fx(f)
    g, o = f[..., 0], out[..., 0]
    k = np.ones((3, 3), np.uint8)
    low = cv2.erode(g, k).astype(int) - fx.overshoot
    high = cv2.dilate(g, k).astype(int) + fx.overshoot
    assert ((o >= low) & (o <= high)).all()


def test_colours_are_not_shifted():
    import cv2
    rng = np.random.default_rng(1)
    base = text_like()
    tinted = np.clip(base.astype(np.float32) * np.array([1.0, 0.8, 0.6]), 0, 255).astype(np.uint8)   # coloured paper
    out = Sharpen(1.0, reference_width=160)(tinted)
    chroma = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)[..., 1:].astype(int)
    assert np.abs(chroma(out) - chroma(tinted)).mean() < 0.5


def test_every_channel_gets_the_same_change():
    f = text_like()
    f[..., 2] = np.clip(f[..., 2].astype(int) + 30, 0, 255).astype(np.uint8)    # unequal channels, no clipping
    out = Sharpen(1.0, reference_width=160)(f)
    change = out.astype(int) - f.astype(int)
    interior = change[20:100, 20:140]
    assert (interior[..., 0] == interior[..., 1]).all()


def test_radius_follows_the_frame_width():
    fx = Sharpen(1.0, radius=1.5)
    assert fx.radius_for(1280) == pytest.approx(1.5)
    assert fx.radius_for(1920) == pytest.approx(2.25)            # 1080p blurs by more pixels, so sharpen wider
    assert fx.radius_for(640) == pytest.approx(0.75)
    assert fx.radius_for(10) == 0.5                              # never collapses below half a pixel


def test_a_bigger_picture_is_sharpened_by_a_matching_amount():
    import cv2
    small = text_like(120, 160)                                  # 1.6 px lens blur at 160 px wide...
    big = cv2.resize(small, (320, 240), interpolation=cv2.INTER_CUBIC)       # ...is ~3.2 px at 320 px wide
    gain = lambda img, ref: edge_slope(Sharpen(1.0, reference_width=ref)(img)) / edge_slope(img)
    assert gain(small, 160) > 1.1 and gain(big, 160) > 1.1


@pytest.mark.parametrize("shape", [(1, 1, 3), (2, 5, 3), (3, 3, 3), (7, 9, 3)])
def test_tiny_frames_do_not_crash(shape):
    f = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    assert Sharpen(1.0)(f).shape == f.shape


def test_input_is_never_modified():
    f = text_like(noise=1.0)
    keep = f.copy()
    Sharpen(1.5)(f)
    assert (f == keep).all()
