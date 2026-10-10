import argparse

from cammy.settings import RANGES, Settings


def test_defaults_are_all_off():
    s = Settings()
    assert not s.wants_face and not s.wants_background


def test_clamps_out_of_range():
    s = Settings(white_balance=5, contrast=0, brightness=-999).normalized()
    assert s.white_balance == RANGES["white_balance"][1]
    assert s.contrast == RANGES["contrast"][0]
    assert s.brightness == RANGES["brightness"][0]


def test_repairs_bad_values():
    s = Settings.from_dict({"skin_smooth": "oops", "background": "lava",
                            "bg_color": "red", "denoise": 1, "unknown": 3})
    assert s.skin_smooth == 0.0 and s.background == "off"
    assert s.bg_color == "#00ff00" and s.denoise is True


def test_dict_round_trip():
    s = Settings(white_balance=0.5, denoise=True, background="blur", bg_blur=0.04)
    assert Settings.from_dict(s.to_dict()) == s


def test_replace_validates():
    assert Settings().replace(eyes=3).eyes == 1.0


def test_wants_flags():
    assert Settings(teeth=0.3).wants_face
    assert Settings(background="blur").wants_background


def ns(**kw):
    d = dict(temporal_denoise=0.0, white_balance=0.0, face_exposure=0.0, denoise=False, skin_smooth=0.0, contrast=1.0,
             brightness=0.0, eyes=0.0, teeth=0.0, sharpen=0.0, bg_blur=0.0, bg_image=None, bg_color=None)
    d.update(kw)
    return argparse.Namespace(**d)


def test_from_args_background_modes():
    assert Settings.from_args(ns(bg_blur=0.04)).background == "blur"
    assert Settings.from_args(ns(bg_image="a.png")).bg_image == "a.png"
    green_bgr = (0, 255, 0)
    assert Settings.from_args(ns(bg_color=(255, 0, 0))).bg_color == "#0000ff"  # BGR blue -> #0000ff
    assert Settings.from_args(ns(bg_color=green_bgr)).background == "color"
    assert Settings.from_args(ns()).background == "off"
