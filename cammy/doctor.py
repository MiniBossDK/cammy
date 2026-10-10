"""`cammy-doctor`: find out why a meeting app does not list the Cammy virtual camera.

The checks mirror the chain a camera has to pass: kernel module -> device -> something writing to it ->
(for apps that go through PipeWire) a PipeWire source -> the app. Gathering facts and judging them are
separate, so the judgement can be tested without a camera.
"""
import argparse
import fcntl
import json
import os
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .output import device_format

VIDIOC_QUERYCAP = 0x80685600                     # _IOR('V', 0, struct v4l2_capability): 104 bytes
V4L2_CAP_VIDEO_CAPTURE = 0x00000001
V4L2_CAP_VIDEO_OUTPUT = 0x00000002

BROWSERS = {                                      # process name -> (display name, how it finds cameras)
    "firefox-bin": ("Firefox", "pipewire?"), "firefox": ("Firefox", "pipewire?"),
    "brave": ("Brave", "v4l2"), "chrome": ("Chrome", "v4l2"), "chromium": ("Chromium", "v4l2"),
    "google-chrome": ("Chrome", "v4l2"), "msedge": ("Edge", "v4l2"),
}


@dataclass
class Findings:
    device: str = "/dev/video10"
    module_loaded: bool = False
    device_exists: bool = False
    card_name: str = ""
    exclusive_caps: bool | None = None            # None: unknown
    has_capture_cap: bool | None = None           # the device advertises itself as a camera right now
    format: tuple | None = None                   # (w, h) readers get; None while nothing is writing
    cammy_running: bool = False
    holders: list = field(default_factory=list)   # names of programs that have the device open
    browsers: list = field(default_factory=list)  # names of browsers that are running
    holding_browsers: list = field(default_factory=list)   # browsers that have the device open right now
    pipewire: str = "unknown"                     # "source", "no-source", "unavailable", "unknown"
    other_loopbacks: list = field(default_factory=list)


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def device_caps(path):
    """The device's current capability flags (VIDIOC_QUERYCAP device_caps), or None."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        buf = bytearray(104)
        fcntl.ioctl(fd, VIDIOC_QUERYCAP, buf)
        return struct.unpack_from("I", buf, 88)[0]        # device_caps follows driver, card, bus_info, version, capabilities
    except OSError:
        return None
    finally:
        os.close(fd)


def is_cammy_command(argv):
    """True for a running Cammy app: `cammy`, `cammy-gui`, or `python [script] -m cammy[.gui|.main]` (not the doctor).

    Only the program itself, or the script handed to Python, counts: a terminal merely opened in a folder
    called "cammy" has that name among its arguments but is not Cammy.
    """
    if not argv:
        return False
    program = Path(argv[0]).name
    if program in ("cammy", "cammy-gui"):
        return True
    if not program.startswith("python"):
        return False
    if len(argv) > 1 and not argv[1].startswith("-") and Path(argv[1]).name in ("cammy", "cammy-gui"):
        return True                                           # python /path/to/bin/cammy
    return any(flag == "-m" and module in ("cammy", "cammy.gui", "cammy.main") for flag, module in zip(argv, argv[1:]))


def processes():
    """[(pid, name, [open file targets])] for processes we may inspect (our own, normally)."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        name = _read(entry / "comm") or ""
        targets = []
        try:
            targets = [os.readlink(fd) for fd in (entry / "fd").iterdir()]
        except OSError:
            pass
        raw = _read(entry / "cmdline") or ""
        found.append((int(entry.name), name, targets, raw.split("\0") if raw else []))
    return found


def pipewire_state(device_name):
    """Does PipeWire expose a camera source for this device? 'source' | 'no-source' | 'unavailable'."""
    if not shutil.which("pw-dump"):
        return "unavailable"
    try:
        dump = json.loads(subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=10).stdout or "[]")
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return "unavailable"
    for obj in dump:
        props = (obj.get("info") or {}).get("props") or {}
        if props.get("media.class") == "Video/Source" and device_name and device_name in json.dumps(props):
            return "source"
    return "no-source"


def gather(device="/dev/video10"):
    f = Findings(device=device)
    f.module_loaded = Path("/sys/module/v4l2loopback").exists()
    number = "".join(ch for ch in device if ch.isdigit())
    f.device_exists = Path(device).exists()
    f.card_name = _read(f"/sys/class/video4linux/video{number}/name") or ""
    if f.module_loaded:
        numbers = (_read("/sys/module/v4l2loopback/parameters/video_nr") or "").split(",")
        flags = (_read("/sys/module/v4l2loopback/parameters/exclusive_caps") or "").split(",")
        if number in numbers and numbers.index(number) < len(flags):
            f.exclusive_caps = flags[numbers.index(number)].strip().upper() in ("Y", "1")
        for entry in Path("/sys/class/video4linux").iterdir():
            label = _read(entry / "name") or ""
            if label and "webcam" not in label.lower() and "ir camera" not in label.lower() and entry.name != f"video{number}":
                f.other_loopbacks.append(f"/dev/{entry.name} ({label})")
    if f.device_exists:
        caps = device_caps(device)
        f.has_capture_cap = None if caps is None else bool(caps & V4L2_CAP_VIDEO_CAPTURE)
        f.format = device_format(device)
    me = os.getpid()
    for pid, name, targets, cmdline in processes():
        if pid != me and is_cammy_command(cmdline):
            f.cammy_running = True
        if device in targets and pid != me:
            f.holders.append(name)
            if name in BROWSERS and BROWSERS[name][0] not in f.holding_browsers:
                f.holding_browsers.append(BROWSERS[name][0])
        if name in BROWSERS:
            label = BROWSERS[name][0]
            if label not in f.browsers:
                f.browsers.append(label)
    f.pipewire = pipewire_state(f.card_name) if f.has_capture_cap else "unknown"
    return f


def diagnose(f):
    """Judge the findings: a list of (level, message) with level 'ok', 'problem' or 'note', best advice last."""
    out = []
    if not f.module_loaded:
        return [("problem", "The v4l2loopback kernel module is not loaded, so there is no virtual camera at all."),
                ("note", "Run:  cammy-setup-camera   (it asks for your password; after a reboot it must be run again "
                         "unless you make it permanent, see the commands it prints).")]
    out.append(("ok", "The v4l2loopback module is loaded."))
    if not f.device_exists:
        out.append(("problem", f"{f.device} does not exist. Another number may have been used"
                               + (f": {', '.join(f.other_loopbacks)}" if f.other_loopbacks else ".")))
        out.append(("note", "Run:  cammy-setup-camera   or start Cammy with  --virtual-device /dev/videoN."))
        return out
    out.append(("ok", f"{f.device} exists" + (f" and is named '{f.card_name}'." if f.card_name else ".")))
    if f.exclusive_caps is False:
        out.append(("problem", "The device was created without exclusive_caps=1. Chrome-based browsers and Meet "
                               "usually refuse to list such a device. Recreate it with cammy-setup-camera "
                               "(unload first: sudo modprobe -r v4l2loopback, with no program using it)."))
    elif f.exclusive_caps:
        out.append(("ok", "exclusive_caps is on, so the device shows up as a camera only while a program is sending "
                          "video to it."))
    if not f.has_capture_cap:
        out.append(("problem", "Nothing is sending video to the virtual camera right now, so it is not advertised as a "
                               "camera and meeting apps cannot list it."
                               + ("" if not f.cammy_running else " (A Cammy process is running, but its virtual camera is switched off or failed.)")))
        out.append(("note", "Start Cammy with the Virtual camera box ticked (cammy-gui: it is remembered from then on) or "
                            "`cammy --virtual-camera`, wait until it says 'Streaming to ...', then reload the meeting page "
                            "or reopen the camera list."))
        return out
    size = f"{f.format[0]}x{f.format[1]}" if f.format else "an unknown size"
    out.append(("ok", f"Something is streaming to it ({size}), so it is advertised as a camera."
                      + (f" Open in: {', '.join(sorted(set(f.holders)))}." if f.holders else "")))
    pipewire_browsers = [b for b in f.browsers if b == "Firefox"]
    if f.holding_browsers:
        out.append(("ok", f"{', '.join(f.holding_browsers)} already has the camera open, so it can see it"
                          + (" (PipeWire does not list it, but that browser reads the device directly)." if f.pipewire == "no-source" else ".")))
    elif f.pipewire == "no-source":
        out.append(("problem", "PipeWire has the device but no camera source for it. Apps that get their cameras from "
                               "PipeWire (Firefox on many systems) will not list it; apps that read the device directly "
                               "(Chrome, Brave, Chromium) will."))
        out.append(("note", "Fix: restart WirePlumber while Cammy is streaming:  systemctl --user restart wireplumber  "
                            "(a second of audio/video routing hiccup, so not during a call), then reload the page. "
                            "Or use a Chromium-based browser."
                            + (f" You have {', '.join(f.browsers)} running." if f.browsers else "")))
    elif f.pipewire == "source":
        out.append(("ok", "PipeWire exposes a camera source for it too."))
    elif pipewire_browsers:
        out.append(("note", "Firefox may get its cameras through PipeWire; if it does not list Cammy, install pipewire-utils "
                            "(pw-dump) and run this again for a precise answer."))
    out.append(("note", "Meeting apps read the camera list when the page loads: after the status above says streaming, reload "
                        "the tab (or leave and rejoin) and choose 'Cammy' in the camera menu."))
    return out


MARK = {"ok": "  ok    ", "problem": "  PROBLEM", "note": "  ->    "}


def report(findings):
    lines = [f"Cammy virtual camera check for {findings.device}", ""]
    lines += [f"{MARK[level]} {text}" for level, text in diagnose(findings)]
    problems = sum(1 for level, _ in diagnose(findings) if level == "problem")
    lines += ["", "Everything looks fine." if not problems else f"{problems} problem{'s' if problems != 1 else ''} found."]
    return "\n".join(lines), problems


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cammy-doctor", description=__doc__)
    parser.add_argument("--device", default="/dev/video10", help="the virtual camera device to check")
    args = parser.parse_args(argv)
    text, problems = report(gather(args.device))
    print(text)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
