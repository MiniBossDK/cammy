import importlib
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import cammy
from cammy import desktop, models, setup_camera

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_the_version_is_semver_and_the_command_reports_it():
    assert re.fullmatch(r"\d+\.\d+\.\d+", cammy.__version__)
    out = subprocess.run([sys.executable, "-m", "cammy", "--version"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and out.stdout.strip() == f"cammy {cammy.__version__}"


def test_every_console_script_points_at_a_real_function():
    scripts = PYPROJECT["project"]["scripts"]
    assert set(scripts) == {"cammy", "cammy-gui", "cammy-setup-camera", "cammy-install-desktop"}
    for name, target in scripts.items():
        module, function = target.split(":")
        assert callable(getattr(importlib.import_module(module), function)), name


def test_the_project_metadata_is_consistent():
    project = PYPROJECT["project"]
    assert project["requires-python"].startswith(">=3.12")                 # the only version it was tested on
    names = {re.split(r"[<>=!~ \[]", d)[0].lower() for d in project["dependencies"]}
    assert {"numpy", "mediapipe", "pyvirtualcam", "pyside6-essentials", "opencv-contrib-python"} <= names
    assert "opencv-python" not in names, "mediapipe needs the contrib build; both installed together conflict"
    assert "ncnn" in {re.split(r"[<>=!~ \[]", d)[0] for d in project["optional-dependencies"]["ai"]}
    assert PYPROJECT["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "cammy.__version__"}


def test_the_files_the_package_ships_exist():
    data = ROOT / "cammy" / "data"
    assert (data / "setup_virtual_camera.sh").is_file() and (data / "cammy.svg").is_file()
    assert PYPROJECT["tool"]["setuptools"]["package-data"]["cammy"] == ["data/*"]
    assert not (ROOT / "requirements.txt").exists(), "dependencies live in pyproject.toml only"


def test_the_setup_script_is_valid_bash():
    assert subprocess.run(["bash", "-n", str(ROOT / "cammy" / "data" / "setup_virtual_camera.sh")]).returncode == 0


def test_the_setup_command_runs_the_bundled_script_with_its_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "call", lambda cmd: calls.append(cmd) or 0)
    assert setup_camera.main(["11"]) == 0
    bash, script, number = calls[0]
    assert Path(bash).name == "bash" and script.endswith("setup_virtual_camera.sh") and number == "11"


# ---- the application menu entry ----------------------------------------------------

def test_the_menu_entry_launches_the_gui_with_an_icon():
    text = desktop.desktop_entry("/opt/venv/bin/cammy-gui")
    lines = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
    assert lines["Exec"] == "/opt/venv/bin/cammy-gui" and lines["Icon"] == "cammy" and lines["Terminal"] == "false"
    assert text.startswith("[Desktop Entry]") and lines["Type"] == "Application"


def test_install_writes_the_entry_and_icon_and_uninstall_removes_them(tmp_path):
    entry, icon = desktop.install("/x/cammy-gui", base=tmp_path)
    assert entry == tmp_path / "applications" / "cammy.desktop" and entry.read_text().count("Exec=/x/cammy-gui") == 1
    assert icon == tmp_path / "icons" / "hicolor" / "scalable" / "apps" / "cammy.svg"
    assert icon.read_text().lstrip().startswith("<?xml") and "<svg" in icon.read_text()
    desktop.install("/y/cammy-gui", base=tmp_path)                           # installing again just updates it
    assert "Exec=/y/cammy-gui" in entry.read_text()
    assert sorted(desktop.uninstall(tmp_path)) == sorted([entry, icon])
    assert not entry.exists() and not icon.exists()
    assert desktop.uninstall(tmp_path) == []                                  # nothing left: not an error


def test_the_menu_command_defaults_to_the_users_data_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert desktop.main([]) == 0
    assert (tmp_path / "applications" / "cammy.desktop").exists()
    assert desktop.main(["--uninstall"]) == 0
    assert not (tmp_path / "applications" / "cammy.desktop").exists()


# ---- where models are kept ----------------------------------------------------------

def test_models_live_in_a_per_user_folder_not_in_the_package(tmp_path, monkeypatch):
    monkeypatch.delenv("CAMMY_MODELS_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert models.models_dir() == tmp_path / "cammy" / "models"
    monkeypatch.delenv("XDG_DATA_HOME")
    assert models.models_dir() == Path.home() / ".local" / "share" / "cammy" / "models"
    assert ROOT not in models.models_dir().parents


def test_the_models_folder_can_be_overridden(tmp_path, monkeypatch):
    monkeypatch.setenv("CAMMY_MODELS_DIR", str(tmp_path / "m"))
    assert models.models_dir() == tmp_path / "m"
