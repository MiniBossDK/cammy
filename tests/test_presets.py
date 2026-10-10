import pytest

from cammy.presets import BUILTIN, PresetStore
from cammy.settings import Settings


@pytest.fixture
def store(tmp_path):
    return PresetStore(tmp_path / "cfg")


def test_builtins_listed_first(store):
    assert store.names()[:len(BUILTIN)] == list(BUILTIN)


def test_save_get_delete_user_preset(store):
    s = Settings(skin_smooth=0.7, denoise=True)
    store.save("Mine", s)
    assert store.get("Mine") == s and "Mine" in store.names()
    store.delete("Mine")
    assert store.get("Mine") is None


def test_builtins_are_protected(store):
    with pytest.raises(ValueError):
        store.save("Natural", Settings())
    with pytest.raises(ValueError):
        store.delete("Natural")
    with pytest.raises(ValueError):
        store.save("  ", Settings())


def test_last_settings_round_trip(store):
    assert store.load_last() is None
    s = Settings(white_balance=0.4)
    store.save_last(s)
    assert store.load_last() == s


def test_corrupt_files_are_ignored(store):
    store.directory.mkdir(parents=True)
    (store.directory / "presets.json").write_text("{not json")
    (store.directory / "last.json").write_text("[1, 2]")
    assert store.user_presets() == {} and store.load_last() is None


def test_name_of_matches_builtin(store):
    assert store.name_of(BUILTIN["Natural"]) == "Natural"
    assert store.name_of(Settings(eyes=0.123)) is None


def test_a_saved_preset_does_not_remember_mirror(store):
    store.save("Mine", Settings(skin_smooth=0.5, mirror=True))
    assert store.get("Mine").mirror is False


def test_mirror_does_not_stop_a_preset_from_matching(store):
    assert store.name_of(BUILTIN["Natural"].replace(mirror=True)) == "Natural"
    assert store.name_of(Settings(eyes=0.123, mirror=True)) is None


def test_a_saved_preset_does_not_remember_resolution(store):
    store.save("Mine", Settings(skin_smooth=0.5, resolution="1080p", mirror=True))
    saved = store.get("Mine")
    assert saved.resolution == "720p" and saved.mirror is False and saved.skin_smooth == 0.5


def test_resolution_does_not_stop_a_preset_from_matching(store):
    assert store.name_of(BUILTIN["Natural"].replace(resolution="1080p", mirror=True)) == "Natural"


# ---- the recommended preset ---------------------------------------------------------

from cammy.presets import RECOMMENDED, starting_settings  # noqa: E402


def test_recommended_is_listed_right_after_off(store):
    assert store.names()[:2] == ["Off", "Recommended"]
    assert store.get("Recommended") == RECOMMENDED


def test_recommended_avoids_the_settings_that_hurt_picture_quality():
    r = RECOMMENDED
    assert r.contrast == 1.0 and r.brightness == 0.0          # pushing either crushes shadows
    assert r.denoise is False                                 # a bilateral filter softens text
    assert r.background == "off"                              # a held-up page would be blurred
    assert 0.5 <= r.temporal_denoise <= 0.8                   # strong enough to help, weak enough not to ghost
    assert r.skin_smooth <= 0.4 and r.eyes <= 0.2 and r.teeth <= 0.3   # light touch
    assert 0.5 <= r.sharpen <= 1.0
    assert r.mirror is False and r.resolution == "720p"       # camera preferences stay at their defaults


def test_meeting_is_the_recommended_look_plus_a_light_blur():
    meeting = BUILTIN["Meeting"]
    assert meeting.without_camera().replace(background="off") == RECOMMENDED
    assert meeting.background == "blur" and meeting.bg_blur <= 0.04 and meeting.keep_hands


def test_new_users_start_on_recommended_and_returning_users_keep_their_settings(store):
    assert starting_settings(store) == RECOMMENDED
    mine = Settings(eyes=0.4, mirror=True)
    store.save_last(mine)
    assert starting_settings(store) == mine
