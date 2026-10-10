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
    d = dict(resolution="720p", mirror=False, keep_hands=True, temporal_denoise=0.0, white_balance=0.0, face_exposure=0.0, denoise=False, skin_smooth=0.0, contrast=1.0,
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


def test_keep_hands_defaults_on_and_only_matters_with_a_background():
    assert Settings().keep_hands is True
    assert not Settings(keep_hands=True).wants_hands                       # no background effect
    assert Settings(background="blur").wants_hands
    assert not Settings(background="blur", keep_hands=False).wants_hands


def test_keep_hands_survives_a_round_trip_and_bad_values():
    assert Settings.from_dict(Settings(keep_hands=False).to_dict()).keep_hands is False
    assert Settings.from_dict({"keep_hands": 0}).keep_hands is False
    assert Settings.from_dict({}).keep_hands is True


def test_from_args_reads_keep_hands():
    assert Settings.from_args(ns(keep_hands=False)).keep_hands is False


def test_mirror_defaults_off_and_round_trips():
    assert Settings().mirror is False
    assert Settings.from_dict(Settings(mirror=True).to_dict()).mirror is True
    assert Settings.from_dict({"mirror": "yes"}).mirror is True
    assert Settings.from_dict({}).mirror is False


def test_from_args_reads_mirror():
    assert Settings.from_args(ns(mirror=True)).mirror is True


def test_resolution_defaults_to_720p_and_maps_to_a_size():
    assert Settings().resolution == "720p" and Settings().size == (1280, 720)
    assert Settings(resolution="1080p").size == (1920, 1080)


def test_unknown_resolution_falls_back_to_720p():
    assert Settings.from_dict({"resolution": "4k"}).resolution == "720p"
    assert Settings.from_dict({"resolution": 1080}).resolution == "720p"
    assert Settings.from_dict(Settings(resolution="1080p").to_dict()).resolution == "1080p"


def test_from_args_reads_resolution():
    assert Settings.from_args(ns(resolution="1080p")).size == (1920, 1080)


def test_camera_preferences_can_be_stripped_and_carried_over():
    mine = Settings(mirror=True, resolution="1080p", sharpen=0.3)
    plain = mine.without_camera()
    assert plain.mirror is False and plain.resolution == "720p" and plain.sharpen == 0.3
    carried = Settings(eyes=0.5).keeping_camera_of(mine)
    assert carried.mirror and carried.resolution == "1080p" and carried.eyes == 0.5
