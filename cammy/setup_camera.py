"""`cammy-setup-camera`: load the v4l2loopback module so Cammy can publish a virtual camera (Linux)."""
import shutil
import subprocess
import sys
from importlib import resources


def script_path():
    """The bundled shell script (it asks for sudo itself, so you can type your password)."""
    return resources.files("cammy").joinpath("data/setup_virtual_camera.sh")


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    bash = shutil.which("bash")
    if bash is None:
        print("bash was not found; run the commands in the script by hand:", script_path(), file=sys.stderr)
        return 1
    with resources.as_file(script_path()) as path:
        return subprocess.call([bash, str(path), *args])


if __name__ == "__main__":
    sys.exit(main())
