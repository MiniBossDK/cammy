"""Qt control window: live preview, sliders, presets and virtual camera toggle.

Run with:  python -m cammy.gui   (or: python -m cammy --gui)
"""
import argparse
import logging
import sys
import time

import cv2
from PySide6 import QtCore, QtGui, QtWidgets

from .capture import Capture
from .output import VirtualCamera, VirtualCameraError
from .presets import BUILTIN, PresetStore, starting_settings
from .processor import Processor
from .settings import RANGES, RESOLUTIONS, Settings

STEPS = 1000  # slider resolution
RETRY_FRAMES = 60  # after a processing failure, show raw frames this long before trying again
log = logging.getLogger("cammy")

# (field, label, formatter) per group, in display order.
SLIDERS = {
    "Light and colour": [
        ("temporal_denoise", "Temporal denoise", lambda v: f"{v:.0%}"),
        ("white_balance", "White balance", lambda v: f"{v:.0%}"),
        ("face_exposure", "Face exposure", lambda v: f"{v:.0%}"),
        ("brightness", "Brightness", lambda v: f"{v:+.0f}"),
        ("contrast", "Contrast", lambda v: f"{v:.2f}x"),
    ],
    "Face": [
        ("skin_smooth", "Skin smoothing", lambda v: f"{v:.0%}"),
        ("ai_look", "AI look", lambda v: f"{v:.0%}"),
        ("eyes", "Eye brightening", lambda v: f"{v:.0%}"),
        ("teeth", "Teeth whitening", lambda v: f"{v:.0%}"),
    ],
    "Detail": [
        ("sharpen", "Sharpen", lambda v: f"{v:.0%}"),
    ],
}
CUSTOM = "Custom"
BACKGROUND_LABELS = {"off": "Off", "blur": "Blur", "image": "Image", "color": "Colour"}


class MainWindow(QtWidgets.QWidget):
    """Controls plus preview. It owns no camera: it only emits what to change.

    Signals:
        settings_changed(Settings)   user moved a control or picked a preset
        virtual_camera_toggled(bool, str)   enable/disable, device path
        show_original_toggled(bool)
    """

    settings_changed = QtCore.Signal(object)
    virtual_camera_toggled = QtCore.Signal(bool, str)
    show_original_toggled = QtCore.Signal(bool)

    def __init__(self, store, settings, device="/dev/video10"):
        super().__init__()
        self.setWindowTitle("Cammy")
        self.store = store
        self.settings = settings
        self._sliders = {}
        self._value_labels = {}
        self._formatters = {}
        self._building = True

        root = QtWidgets.QHBoxLayout(self)
        self.preview = QtWidgets.QLabel("Starting camera...")
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setMinimumSize(480, 270)
        self.preview.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.preview.setStyleSheet("background: #111; color: #aaa;")
        root.addWidget(self.preview, 1)

        panel = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(panel)
        col.addLayout(self._build_presets())
        for title, rows in SLIDERS.items():
            col.addWidget(self._build_group(title, rows))
        col.addWidget(self._build_background())
        col.addWidget(self._build_output(device))
        col.addStretch(1)
        self.status = QtWidgets.QLabel("")
        self.message = QtWidgets.QLabel("")
        self.message.setWordWrap(True)
        self.message.setStyleSheet("color: #c0392b;")
        col.addWidget(self.status)
        col.addWidget(self.message)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(360)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        root.addWidget(scroll)
        self.resize(1200, 700)

        self._building = False
        self.set_settings(settings)

    # ---- building -------------------------------------------------------
    def _build_presets(self):
        row = QtWidgets.QHBoxLayout()
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.activated.connect(self._preset_picked)
        save = QtWidgets.QPushButton("Save...")
        save.clicked.connect(self._save_clicked)
        self.delete_button = QtWidgets.QPushButton("Delete")
        self.delete_button.clicked.connect(self._delete_clicked)
        row.addWidget(QtWidgets.QLabel("Preset"))
        row.addWidget(self.preset_combo, 1)
        row.addWidget(save)
        row.addWidget(self.delete_button)
        self._reload_presets()
        return row

    def _build_group(self, title, rows):
        box = QtWidgets.QGroupBox(title)
        grid = QtWidgets.QGridLayout(box)
        for r, (name, label, fmt) in enumerate(rows):
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(0, STEPS)
            slider.valueChanged.connect(lambda _v, n=name: self._slider_moved(n))
            value = QtWidgets.QLabel()
            value.setMinimumWidth(48)
            value.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            grid.addWidget(QtWidgets.QLabel(label), r, 0)
            grid.addWidget(slider, r, 1)
            grid.addWidget(value, r, 2)
            self._sliders[name], self._value_labels[name], self._formatters[name] = slider, value, fmt
        if title == "Light and colour":
            self.denoise = QtWidgets.QCheckBox("Denoise")
            self.denoise.toggled.connect(lambda _c: self._emit_from_widgets())
            grid.addWidget(self.denoise, len(rows), 0, 1, 3)
        return box

    def _build_background(self):
        box = QtWidgets.QGroupBox("Background")
        grid = QtWidgets.QGridLayout(box)
        self.bg_combo = QtWidgets.QComboBox()
        for key, label in BACKGROUND_LABELS.items():
            self.bg_combo.addItem(label, key)
        self.bg_combo.activated.connect(lambda _i: self._emit_from_widgets())
        grid.addWidget(self.bg_combo, 0, 0, 1, 3)

        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(0, STEPS)
        slider.valueChanged.connect(lambda _v: self._slider_moved("bg_blur"))
        value = QtWidgets.QLabel()
        value.setMinimumWidth(48)
        value.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        grid.addWidget(QtWidgets.QLabel("Blur"), 1, 0)
        grid.addWidget(slider, 1, 1)
        grid.addWidget(value, 1, 2)
        self._sliders["bg_blur"], self._value_labels["bg_blur"] = slider, value
        self._formatters["bg_blur"] = lambda v: f"{v:.1%}"

        self.image_button = QtWidgets.QPushButton("Choose image...")
        self.image_button.clicked.connect(self._choose_image)
        self.color_button = QtWidgets.QPushButton("Choose colour...")
        self.color_button.clicked.connect(self._choose_color)
        grid.addWidget(self.image_button, 2, 0, 1, 3)
        grid.addWidget(self.color_button, 3, 0, 1, 3)
        self.hands_check = QtWidgets.QCheckBox("Keep hands and fingers sharp")
        self.hands_check.toggled.connect(lambda _c: self._emit_from_widgets())
        grid.addWidget(self.hands_check, 4, 0, 1, 3)
        return box

    def _build_output(self, device):
        box = QtWidgets.QGroupBox("Output")
        grid = QtWidgets.QGridLayout(box)
        self.vcam_check = QtWidgets.QCheckBox("Virtual camera")
        self.vcam_device = QtWidgets.QLineEdit(device)
        self.vcam_check.toggled.connect(
            lambda on: self.virtual_camera_toggled.emit(on, self.vcam_device.text().strip()))
        self.resolution_combo = QtWidgets.QComboBox()
        for label in RESOLUTIONS:
            self.resolution_combo.addItem(label, label)
        self.resolution_combo.setToolTip("1080p shows more detail, such as text, but needs more processing")
        self.resolution_combo.activated.connect(lambda _i: self._emit_from_widgets())
        self.mirror_check = QtWidgets.QCheckBox("Mirror image")
        self.mirror_check.toggled.connect(lambda _c: self._emit_from_widgets())
        self.original_check = QtWidgets.QCheckBox("Show original in preview")
        self.original_check.toggled.connect(self.show_original_toggled)
        grid.addWidget(self.vcam_check, 0, 0)
        grid.addWidget(self.vcam_device, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Camera resolution"), 1, 0)
        grid.addWidget(self.resolution_combo, 1, 1)
        grid.addWidget(self.mirror_check, 2, 0, 1, 2)
        grid.addWidget(self.original_check, 3, 0, 1, 2)
        return box

    # ---- settings <-> widgets ------------------------------------------
    def _to_slider(self, name, value):
        lo, hi = RANGES[name]
        return round((value - lo) / (hi - lo) * STEPS)

    def _from_slider(self, name):
        lo, hi = RANGES[name]
        return lo + self._sliders[name].value() / STEPS * (hi - lo)

    def _refresh_labels(self):
        for name, label in self._value_labels.items():
            label.setText(self._formatters[name](self._from_slider(name)))
        self.image_button.setText(self._short(self.settings.bg_image) or "Choose image...")
        self.color_button.setText(f"Colour {self.settings.bg_color}")
        self.color_button.setStyleSheet(f"border-left: 14px solid {self.settings.bg_color};")
        mode = self.settings.background
        self._sliders["bg_blur"].setEnabled(mode == "blur")
        self.image_button.setEnabled(mode == "image")
        self.color_button.setEnabled(mode == "color")
        self.hands_check.setEnabled(mode != "off")

    @staticmethod
    def _short(path):
        return path.rsplit("/", 1)[-1] if path else ""

    def set_settings(self, settings, emit=False):
        """Show `settings` in the controls; emit only if asked."""
        self._building = True
        try:
            self.settings = settings
            for name, slider in self._sliders.items():
                slider.setValue(self._to_slider(name, getattr(settings, name)))
            self.denoise.setChecked(settings.denoise)
            self.hands_check.setChecked(settings.keep_hands)
            self.mirror_check.setChecked(settings.mirror)
            self.resolution_combo.setCurrentIndex(self.resolution_combo.findData(settings.resolution))
            self.bg_combo.setCurrentIndex(self.bg_combo.findData(settings.background))
            self._refresh_labels()
            self._sync_preset_combo()
        finally:
            self._building = False
        if emit:
            self.settings_changed.emit(self.settings)

    def _collect(self):
        # Sliders have only STEPS positions, so reading one back is slightly off the
        # exact value (a preset's contrast 1.0 comes back as 1.0000000000000002).
        # Keep the exact value unless that slider was actually moved.
        values = {}
        for name, slider in self._sliders.items():
            if slider.value() != self._to_slider(name, getattr(self.settings, name)):
                values[name] = self._from_slider(name)
        return self.settings.replace(
            denoise=self.denoise.isChecked(),
            keep_hands=self.hands_check.isChecked(),
            mirror=self.mirror_check.isChecked(),
            resolution=self.resolution_combo.currentData(),
            background=self.bg_combo.currentData(),
            **values,
        )

    def _emit_from_widgets(self):
        if self._building:
            return
        self.settings = self._collect()
        self._refresh_labels()
        self._sync_preset_combo()
        self.settings_changed.emit(self.settings)

    def _slider_moved(self, name):
        if self._building:
            return
        self._value_labels[name].setText(self._formatters[name](self._from_slider(name)))
        self._emit_from_widgets()

    # ---- presets --------------------------------------------------------
    def _reload_presets(self):
        self.preset_combo.clear()
        self.preset_combo.addItems(self.store.names())

    def _sync_preset_combo(self):
        match = self.store.name_of(self.settings)
        custom_at = self.preset_combo.findText(CUSTOM)
        if match is None:
            if custom_at < 0:
                self.preset_combo.insertItem(0, CUSTOM)
            self.preset_combo.setCurrentIndex(0)
        else:
            if custom_at >= 0:
                self.preset_combo.removeItem(custom_at)
            self.preset_combo.setCurrentIndex(self.preset_combo.findText(match))
        self.delete_button.setEnabled(match is not None and match not in BUILTIN)

    def _preset_picked(self, _index):
        preset = self.store.get(self.preset_combo.currentText())
        if preset is not None:
            # Mirror and resolution are how you like your camera, not part of a look: keep them.
            self.set_settings(preset.keeping_camera_of(self.settings), emit=True)

    def save_preset(self, name):
        """Store current settings under `name`; returns an error text or None."""
        try:
            self.store.save(name, self.settings)
        except (ValueError, OSError) as error:
            return str(error)
        self._reload_presets()
        self._sync_preset_combo()
        return None

    def delete_preset(self, name):
        try:
            self.store.delete(name)
        except (ValueError, OSError) as error:
            return str(error)
        self._reload_presets()
        self._sync_preset_combo()
        return None

    def _save_clicked(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Save preset", "Preset name:")
        if ok:
            error = self.save_preset(name)
            if error:
                QtWidgets.QMessageBox.warning(self, "Cannot save preset", error)

    def _delete_clicked(self):
        name = self.preset_combo.currentText()
        if QtWidgets.QMessageBox.question(self, "Delete preset", f"Delete '{name}'?") \
                == QtWidgets.QMessageBox.Yes:
            error = self.delete_preset(name)
            if error:
                QtWidgets.QMessageBox.warning(self, "Cannot delete preset", error)

    # ---- background pickers --------------------------------------------
    def _choose_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Background image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if path:
            self.set_settings(self.settings.replace(bg_image=path, background="image"), emit=True)

    def _choose_color(self):
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.settings.bg_color), self)
        if color.isValid():
            self.set_settings(self.settings.replace(bg_color=color.name(), background="color"),
                              emit=True)

    # ---- worker feedback -----------------------------------------------
    @QtCore.Slot(QtGui.QImage)
    def show_frame(self, image):
        pixmap = QtGui.QPixmap.fromImage(image)
        self.preview.setPixmap(pixmap.scaled(self.preview.size(), QtCore.Qt.KeepAspectRatio,
                                             QtCore.Qt.FastTransformation))

    @QtCore.Slot(float, float)
    def show_stats(self, fps, ms):
        self.status.setText(f"{fps:.1f} fps   processing {ms:.0f} ms")

    @QtCore.Slot(str)
    def show_message(self, text):
        self.message.setText(text)

    @QtCore.Slot(str)
    def resolution_was_reverted(self, label):
        """The camera could not switch size: show (and remember) the one still in use, without re-emitting."""
        self.set_settings(self.settings.replace(resolution=label))

    @QtCore.Slot(str)
    def virtual_camera_failed(self, text):
        self.message.setText(text)
        self.vcam_check.blockSignals(True)
        self.vcam_check.setChecked(False)
        self.vcam_check.blockSignals(False)


class Worker(QtCore.QThread):
    """Capture -> process -> (virtual camera) -> preview, off the UI thread."""

    frame = QtCore.Signal(QtGui.QImage)
    stats = QtCore.Signal(float, float)
    message = QtCore.Signal(str)
    vcam_failed = QtCore.Signal(str)
    resolution_reverted = QtCore.Signal(str)      # the camera refused a size: name of the one still in use

    def __init__(self, processor, camera=0, fps=30, preview_width=960):
        super().__init__()
        self.processor = processor
        self.camera = camera
        self.fps = fps
        self.preview_width = preview_width
        self.show_original = False
        self._vcam_wanted = False
        self._vcam_device = None
        self._stop = False

    def set_virtual_camera(self, on, device):
        self._vcam_device = device or None
        self._vcam_wanted = on

    def set_show_original(self, on):
        self.show_original = on

    def stop(self):
        self._stop = True
        self.wait(5000)

    def _sync_vcam(self, vcam):
        """Open or close the virtual camera to match what the user wants."""
        if self._vcam_wanted and vcam is None:
            return VirtualCamera(self.fps, self._vcam_device)
        if not self._vcam_wanted and vcam is not None:
            vcam.close()
            return None
        return vcam

    def _to_qimage(self, bgr):
        h, w = bgr.shape[:2]
        if w > self.preview_width:
            bgr = cv2.resize(bgr, (self.preview_width, round(h * self.preview_width / w)),
                             interpolation=cv2.INTER_AREA)
            h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()

    def _capture_for(self, size, cap=None, current=None):
        """The capture at `size`: reuse `cap` if it already is, else release it and reopen."""
        if cap is not None and current == size:
            return cap, current
        if cap is not None:
            cap.release()
        return Capture(self.camera, *size, fps=self.fps), size

    def run(self):
        size = self.processor.settings.size
        try:
            cap, size = self._capture_for(size)
        except RuntimeError as error:
            self.message.emit(str(error))
            return
        vcam, last_message, failure, frames, t0, busy, retry_in = None, "", "", 0, time.perf_counter(), 0.0, 0
        try:
            while not self._stop:
                wanted = self.processor.settings.size
                if wanted != size:                       # resolution changed in the window
                    try:
                        cap, size = self._capture_for(wanted, cap, size)
                    except RuntimeError as error:
                        self.message.emit(f"Cannot switch resolution: {error}")
                        kept = next(k for k, v in RESOLUTIONS.items() if v == size)
                        self.processor.settings = self.processor.settings.replace(resolution=kept)
                        self.resolution_reverted.emit(kept)
                        continue
                    if vcam is not None:                 # the virtual camera is fixed-size: reopen at the new one
                        vcam.close()
                        vcam = None
                    self.processor.reset()
                raw = cap.read()
                if raw is None:
                    self.message.emit("Camera stopped delivering frames.")
                    break
                start = time.perf_counter()
                if retry_in > 0:                         # recently failed: show the raw picture for a bit
                    retry_in -= 1
                    out = raw
                else:
                    try:
                        out = self.processor.process(raw)
                        failure = ""
                    except Exception as exc:             # e.g. a model that cannot load: keep the video alive
                        log.exception("processing failed")
                        out, retry_in = raw, RETRY_FRAMES
                        failure = f"Processing failed: {exc}"
                try:
                    vcam = self._sync_vcam(vcam)
                    if vcam is not None:
                        vcam.send(out)
                except VirtualCameraError as error:
                    self._vcam_wanted, vcam = False, None
                    self.vcam_failed.emit(str(error))
                busy += time.perf_counter() - start
                self.frame.emit(self._to_qimage(raw if self.show_original else out))
                shown = failure or self.processor.error or self.processor.warning or ""
                if shown != last_message:
                    last_message = shown
                    self.message.emit(shown)
                frames += 1
                now = time.perf_counter()
                if now - t0 >= 0.5:
                    self.stats.emit(frames / (now - t0), busy / frames * 1000)
                    frames, t0, busy = 0, now, 0.0
        finally:
            if vcam is not None:
                vcam.close()
            cap.release()


def launch(camera=0, device="/dev/video10", store=None, fps=30):
    """Create app, window and worker, wired together. Returns (app, window, worker)."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    store = store or PresetStore()
    settings = starting_settings(store)
    window = MainWindow(store, settings, device)
    worker = Worker(Processor(settings), camera, fps)

    window.settings_changed.connect(lambda s: setattr(worker.processor, "settings", s))
    window.virtual_camera_toggled.connect(worker.set_virtual_camera)
    window.show_original_toggled.connect(worker.set_show_original)
    worker.frame.connect(window.show_frame)
    worker.stats.connect(window.show_stats)
    worker.message.connect(window.show_message)
    worker.vcam_failed.connect(window.virtual_camera_failed)
    worker.resolution_reverted.connect(window.resolution_was_reverted)
    worker.start()
    return app, window, worker


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cammy.gui")
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--virtual-device", default="/dev/video10",
                        help="virtual camera device shown in the window")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv)

    app, window, worker = launch(args.camera, args.virtual_device, fps=args.fps)
    window.show()
    code = app.exec()
    worker.stop()
    worker.processor.close()
    window.store.save_last(window.settings)
    return code


if __name__ == "__main__":
    sys.exit(main())
