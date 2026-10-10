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


def test_ticking_the_virtual_camera_emits_settings_with_the_device(window):
    window.vcam_device.setText("/dev/video11")
    window.vcam_check.setChecked(True)
    last = window.emitted[-1]
    assert last.virtual_camera is True and last.virtual_device == "/dev/video11"


def test_editing_the_device_or_size_emits(window):
    window.vcam_device.setText("/dev/video12")
    window.vcam_device.editingFinished.emit()
    assert window.emitted[-1].virtual_device == "/dev/video12"
    window.vcam_size_combo.setCurrentIndex(window.vcam_size_combo.findData("1080p"))
    window.vcam_size_combo.activated.emit(window.vcam_size_combo.currentIndex())
    assert window.emitted[-1].output_size == "1080p" and window.emitted[-1].output_dims == (1920, 1080)


def test_a_blank_device_falls_back_to_the_default(window):
    window.vcam_device.setText("   ")
    window.vcam_check.setChecked(True)
    assert window.emitted[-1].virtual_device == "/dev/video10"


def test_a_virtual_camera_failure_unticks_remembers_it_is_off_and_does_not_emit(window):
    window.vcam_check.setChecked(True)
    before = len(window.emitted)
    window.virtual_camera_failed("no device")
    assert not window.vcam_check.isChecked() and window.settings.virtual_camera is False
    assert window.message.text() == "no device" and len(window.emitted) == before


def test_the_status_line_shows_what_the_worker_reports(window):
    assert window.vcam_status.text() == "Off"
    window.show_vcam_state("Streaming to /dev/video10 at 1280x720")
    assert "Streaming" in window.vcam_status.text()


def test_a_remembered_virtual_camera_is_restored_when_the_window_opens(app, tmp_path):
    mine = Settings(virtual_camera=True, virtual_device="/dev/video11", output_size="1080p")
    w = MainWindow(PresetStore(tmp_path), mine)
    assert w.vcam_check.isChecked() and w.vcam_device.text() == "/dev/video11" and w.vcam_size_combo.currentData() == "1080p"


def test_picking_a_preset_does_not_switch_the_virtual_camera_off(window):
    window.vcam_check.setChecked(True)
    index = window.preset_combo.findText("Natural")
    window.preset_combo.setCurrentIndex(index)
    window.preset_combo.activated.emit(index)
    assert window.emitted[-1].virtual_camera is True
    assert window.preset_combo.currentText() == "Natural"


# ---- the worker decides when to open, reopen and close the virtual camera ----------------

class SpyVirtualCamera:
    instances = []

    def __init__(self, fps=30, device=None, size=None, **kw):
        self.device, self.size, self.closed = device, tuple(size), False
        SpyVirtualCamera.instances.append(self)

    def close(self):
        self.closed = True


def make_worker(monkeypatch, settings):
    import cammy.gui as gui
    from cammy.processor import Processor
    monkeypatch.setattr(gui, "VirtualCamera", SpyVirtualCamera)
    SpyVirtualCamera.instances.clear()
    return gui.Worker(Processor(settings), camera=0)


def test_the_worker_opens_the_virtual_camera_at_the_fixed_output_size(monkeypatch):
    worker = make_worker(monkeypatch, Settings(virtual_camera=True, resolution="1080p", output_size="720p"))
    vcam = worker._sync_vcam(None)
    assert vcam.size == (1280, 720) and vcam.device == "/dev/video10"       # the 1080p capture does not change it


def test_switching_the_capture_resolution_does_not_touch_the_virtual_camera(monkeypatch):
    worker = make_worker(monkeypatch, Settings(virtual_camera=True))
    vcam = worker._sync_vcam(None)
    worker.processor.settings = worker.processor.settings.replace(resolution="1080p")
    assert worker._sync_vcam(vcam) is vcam and not vcam.closed


def test_changing_the_output_size_or_device_reopens_it(monkeypatch):
    worker = make_worker(monkeypatch, Settings(virtual_camera=True))
    vcam = worker._sync_vcam(None)
    worker.processor.settings = worker.processor.settings.replace(output_size="1080p")
    again = worker._sync_vcam(vcam)
    assert vcam.closed and again is not vcam and again.size == (1920, 1080)
    worker.processor.settings = worker.processor.settings.replace(virtual_device="/dev/video11")
    third = worker._sync_vcam(again)
    assert again.closed and third.device == "/dev/video11"


def test_turning_it_off_closes_it_and_nothing_opens_while_off(monkeypatch):
    worker = make_worker(monkeypatch, Settings(virtual_camera=False))
    assert worker._sync_vcam(None) is None and SpyVirtualCamera.instances == []
    worker.processor.settings = worker.processor.settings.replace(virtual_camera=True)
    vcam = worker._sync_vcam(None)
    worker.processor.settings = worker.processor.settings.replace(virtual_camera=False)
    assert worker._sync_vcam(vcam) is None and vcam.closed


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


def test_a_size_held_by_another_app_keeps_streaming_at_that_size(monkeypatch):
    from cammy.output import VirtualCameraError
    worker = make_worker(monkeypatch, Settings(virtual_camera=True, output_size="1080p"))
    seen = []
    worker.output_size_reverted.connect(lambda name, text: seen.append((name, text)))
    assert worker._recover_from_vcam_error(VirtualCameraError("in use at 1280x720", actual=(1280, 720))) is True
    assert worker.processor.settings.output_size == "720p" and worker.processor.settings.virtual_camera is True
    assert seen == [("720p", "in use at 1280x720")]


def test_any_other_virtual_camera_failure_switches_it_off_and_says_why(monkeypatch):
    from cammy.output import VirtualCameraError
    worker = make_worker(monkeypatch, Settings(virtual_camera=True))
    seen = []
    worker.vcam_failed.connect(seen.append)
    assert worker._recover_from_vcam_error(VirtualCameraError("no loopback device")) is False
    assert worker.processor.settings.virtual_camera is False and seen == ["no loopback device"]
    assert worker._recover_from_vcam_error(VirtualCameraError("odd", actual=(800, 600))) is False      # a size we do not offer


def test_the_window_shows_the_size_the_meeting_app_forced(window):
    window.vcam_check.setChecked(True)
    window.vcam_size_combo.setCurrentIndex(window.vcam_size_combo.findData("1080p"))
    before = len(window.emitted)
    window.output_size_was_reverted("720p", "kept at 1280x720")
    assert window.vcam_size_combo.currentData() == "720p" and window.vcam_check.isChecked()
    assert "1280x720" in window.message.text() and len(window.emitted) == before
