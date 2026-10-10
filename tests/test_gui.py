import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from cammy.gui import MainWindow  # noqa: E402
from cammy.presets import BUILTIN, PresetStore  # noqa: E402
from cammy.settings import Settings  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def window(app, tmp_path):
    w = MainWindow(PresetStore(tmp_path), Settings())
    w.emitted = []
    w.settings_changed.connect(w.emitted.append)
    return w


def test_slider_emits_new_settings(window):
    window._sliders["skin_smooth"].setValue(500)
    assert window.emitted[-1].skin_smooth == pytest.approx(0.5)
    assert window.settings == window.emitted[-1]


def test_slider_range_maps_to_field_range(window):
    window._sliders["contrast"].setValue(0)
    assert window.emitted[-1].contrast == pytest.approx(0.7)
    window._sliders["contrast"].setValue(1000)
    assert window.emitted[-1].contrast == pytest.approx(1.5)


def test_value_label_follows_slider(window):
    window._sliders["white_balance"].setValue(750)
    assert window._value_labels["white_balance"].text() == "75%"


def test_picking_preset_updates_controls_and_emits(window):
    index = window.preset_combo.findText("Polished")
    window.preset_combo.setCurrentIndex(index)
    window.preset_combo.activated.emit(index)
    assert window.emitted[-1] == BUILTIN["Polished"]
    assert window.denoise.isChecked()
    assert window._from_slider("skin_smooth") == pytest.approx(BUILTIN["Polished"].skin_smooth, abs=1e-3)


def test_moving_slider_after_preset_shows_custom(window):
    index = window.preset_combo.findText("Natural")
    window.preset_combo.setCurrentIndex(index)
    window.preset_combo.activated.emit(index)
    window._sliders["eyes"].setValue(900)
    assert window.preset_combo.currentText() == "Custom"


def test_save_and_delete_user_preset(window):
    window._sliders["teeth"].setValue(400)
    assert window.save_preset("Mine") is None
    assert window.preset_combo.currentText() == "Mine"
    assert window.delete_button.isEnabled()
    assert window.delete_preset("Mine") is None
    assert window.preset_combo.findText("Mine") < 0


def test_builtin_presets_cannot_be_overwritten_or_deleted(window):
    assert "built-in" in window.save_preset("Natural")
    assert "built-in" in window.delete_preset("Natural")


def test_background_mode_enables_matching_controls(window):
    window.bg_combo.setCurrentIndex(window.bg_combo.findData("blur"))
    window.bg_combo.activated.emit(window.bg_combo.currentIndex())
    assert window.emitted[-1].background == "blur"
    assert window._sliders["bg_blur"].isEnabled() and not window.color_button.isEnabled()
    window.bg_combo.setCurrentIndex(window.bg_combo.findData("color"))
    window.bg_combo.activated.emit(window.bg_combo.currentIndex())
    assert window.color_button.isEnabled() and not window._sliders["bg_blur"].isEnabled()


def test_denoise_checkbox_emits(window):
    window.denoise.setChecked(True)
    assert window.emitted[-1].denoise is True


def test_set_settings_does_not_emit_by_default(window):
    window.set_settings(Settings(sharpen=0.5))
    assert window.emitted == []
    assert window._from_slider("sharpen") == pytest.approx(0.5, abs=1e-3)


def test_virtual_camera_toggle_signal(window):
    seen = []
    window.virtual_camera_toggled.connect(lambda on, dev: seen.append((on, dev)))
    window.vcam_device.setText("/dev/video11")
    window.vcam_check.setChecked(True)
    assert seen == [(True, "/dev/video11")]


def test_virtual_camera_failure_unchecks_without_signal(window):
    seen = []
    window.virtual_camera_toggled.connect(lambda on, dev: seen.append(on))
    window.vcam_check.setChecked(True)
    window.virtual_camera_failed("no device")
    assert not window.vcam_check.isChecked()
    assert seen == [True] and window.message.text() == "no device"


def test_hands_checkbox_emits_and_follows_the_background_mode(window):
    assert window.hands_check.isChecked() and not window.hands_check.isEnabled()      # background off
    window.bg_combo.setCurrentIndex(window.bg_combo.findData("blur"))
    window.bg_combo.activated.emit(window.bg_combo.currentIndex())
    assert window.hands_check.isEnabled()
    window.hands_check.setChecked(False)
    assert window.emitted[-1].keep_hands is False


def test_mirror_checkbox_emits(window):
    window.mirror_check.setChecked(True)
    assert window.emitted[-1].mirror is True
    window.mirror_check.setChecked(False)
    assert window.emitted[-1].mirror is False


def test_picking_a_preset_keeps_the_mirror_choice(window):
    window.mirror_check.setChecked(True)
    index = window.preset_combo.findText("Polished")
    window.preset_combo.setCurrentIndex(index)
    window.preset_combo.activated.emit(index)
    assert window.emitted[-1].mirror is True
    assert window.emitted[-1].replace(mirror=False) == BUILTIN["Polished"]
    assert window.mirror_check.isChecked()


def test_mirroring_does_not_turn_a_preset_into_custom(window):
    index = window.preset_combo.findText("Natural")
    window.preset_combo.setCurrentIndex(index)
    window.preset_combo.activated.emit(index)
    window.mirror_check.setChecked(True)
    assert window.preset_combo.currentText() == "Natural"


def test_resolution_combo_emits(window):
    window.resolution_combo.setCurrentIndex(window.resolution_combo.findData("1080p"))
    window.resolution_combo.activated.emit(window.resolution_combo.currentIndex())
    assert window.emitted[-1].resolution == "1080p" and window.emitted[-1].size == (1920, 1080)


def test_picking_a_preset_keeps_the_resolution_and_does_not_make_it_custom(window):
    window.resolution_combo.setCurrentIndex(window.resolution_combo.findData("1080p"))
    window.resolution_combo.activated.emit(window.resolution_combo.currentIndex())
    index = window.preset_combo.findText("Natural")
    window.preset_combo.setCurrentIndex(index)
    window.preset_combo.activated.emit(index)
    assert window.emitted[-1].resolution == "1080p"
    assert window.preset_combo.currentText() == "Natural"


def test_a_refused_resolution_snaps_the_selector_back_without_emitting(window):
    window.resolution_combo.setCurrentIndex(window.resolution_combo.findData("1080p"))
    window.resolution_combo.activated.emit(window.resolution_combo.currentIndex())
    before = len(window.emitted)
    window.resolution_was_reverted("720p")
    assert window.resolution_combo.currentData() == "720p" and window.settings.resolution == "720p"
    assert len(window.emitted) == before


class FakeCapture:
    opened = []

    def __init__(self, index=0, width=0, height=0, fps=30, **kw):
        self.size_asked = (width, height)
        self.released = False
        FakeCapture.opened.append(self)

    def release(self):
        self.released = True


def test_worker_reopens_the_camera_only_when_the_size_changes(monkeypatch):
    import cammy.gui as gui
    from cammy.processor import Processor
    monkeypatch.setattr(gui, "Capture", FakeCapture)
    FakeCapture.opened.clear()
    worker = gui.Worker(Processor(Settings()), camera=3, fps=30)
    cap, size = worker._capture_for((1280, 720))
    assert size == (1280, 720) and cap.size_asked == (1280, 720)
    same, size = worker._capture_for((1280, 720), cap, size)
    assert same is cap and not cap.released and len(FakeCapture.opened) == 1
    bigger, size = worker._capture_for((1920, 1080), cap, size)
    assert bigger is not cap and cap.released and bigger.size_asked == (1920, 1080) and size == (1920, 1080)


def test_worker_reports_a_camera_that_cannot_reopen(monkeypatch):
    import cammy.gui as gui
    from cammy.processor import Processor

    class Refuses(FakeCapture):
        def __init__(self, *a, **k):
            raise RuntimeError("cannot open webcam at index 0")

    monkeypatch.setattr(gui, "Capture", Refuses)
    worker = gui.Worker(Processor(Settings()))
    with pytest.raises(RuntimeError):
        worker._capture_for((1920, 1080))


def test_the_ai_look_slider_warns_about_text_and_the_requirement(window):
    tip = window._sliders["ai_look"].toolTip()
    assert "ncnn" in tip and "text" in tip


def test_the_ai_look_slider_emits_its_value(window):
    window._sliders["ai_look"].setValue(700)
    assert window.emitted[-1].ai_look == pytest.approx(0.7)


def test_the_preview_is_scaled_smoothly_not_with_nearest_neighbour():
    import inspect
    from cammy.gui import MainWindow
    source = inspect.getsource(MainWindow.show_frame)
    assert "SmoothTransformation" in source and "FastTransformation" not in source.replace("not Fast", "")
