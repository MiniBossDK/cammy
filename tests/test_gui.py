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
