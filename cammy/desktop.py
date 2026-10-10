"""`cammy-install-desktop`: add Cammy to the application menu for the current user (no root needed)."""
import argparse
import os
import shutil
import sys
from importlib import resources
from pathlib import Path

DESKTOP_NAME = "cammy.desktop"
ICON_NAME = "cammy.svg"


def data_home():
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")


def desktop_entry(command):
    """The text of the menu entry that starts the control window with `command`."""
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=Cammy\n"
        "GenericName=Webcam enhancer\n"
        "Comment=Make your webcam look good, with a virtual camera for video calls\n"
        f"Exec={command}\n"
        "Icon=cammy\n"
        "Terminal=false\n"
        "Categories=AudioVideo;Video;\n"
        "Keywords=webcam;camera;virtual camera;video call;\n"
    )


def paths(base=None):
    base = Path(base) if base else data_home()
    return (base / "applications" / DESKTOP_NAME,
            base / "icons" / "hicolor" / "scalable" / "apps" / ICON_NAME)


def install(command, base=None):
    """Write the menu entry and icon under `base` (default: the user's XDG data home)."""
    desktop, icon = paths(base)
    desktop.parent.mkdir(parents=True, exist_ok=True)
    icon.parent.mkdir(parents=True, exist_ok=True)
    desktop.write_text(desktop_entry(command))
    icon.write_bytes(resources.files("cammy").joinpath("data/" + ICON_NAME).read_bytes())
    return desktop, icon


def uninstall(base=None):
    removed = []
    for path in paths(base):
        if path.exists():
            path.unlink()
            removed.append(path)
    return removed


def gui_command():
    """Absolute path of the installed `cammy-gui`, else a way to start it with this Python."""
    found = shutil.which("cammy-gui")
    return found if found else f"{sys.executable} -m cammy.gui"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cammy-install-desktop", description=__doc__)
    parser.add_argument("--uninstall", action="store_true", help="remove the menu entry and icon again")
    args = parser.parse_args(argv)
    if args.uninstall:
        removed = uninstall()
        print("removed:" if removed else "nothing to remove", *map(str, removed))
        return 0
    desktop, icon = install(gui_command())
    print(f"installed {desktop}\n          {icon}\nCammy should now appear in your application menu.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
