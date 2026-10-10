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
