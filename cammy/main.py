"""Live preview with FPS counter. Press q or Esc to quit."""
import argparse
import time
from pathlib import Path

import cv2

from .capture import Capture
from .output import VirtualCamera, VirtualCameraError
from .processor import Processor
from .settings import Settings


def parse_color(text):
    """'#rrggbb' or 'rrggbb' -> BGR tuple."""
    text = text.lstrip("#")
    if len(text) != 6:
        raise argparse.ArgumentTypeError("colour must look like #00ff00")
    r, g, b = (int(text[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def run(args):
    processor = Processor(Settings.from_args(args), detect_every=args.detect_every)
    vcam = VirtualCamera(args.fps, args.virtual_device) if args.virtual_camera else None
    fps, frames, t0 = 0.0, 0, time.perf_counter()
    try:
        with Capture(args.camera, fps=args.fps) as cap:
            while True:
                frame = cap.read()
                if frame is None:
                    print("frame read failed")
                    break
                frame = processor.process(frame)
                if processor.error:
                    print(processor.error)
                    break

                if vcam:
                    first = vcam.device_name is None
                    vcam.send(frame)  # before the FPS overlay: callers see a clean feed
                    if first:
                        print(f"virtual camera on {vcam.device_name}")

                frames += 1
                now = time.perf_counter()
                if now - t0 >= 0.5:
                    fps, frames, t0 = frames / (now - t0), 0, now
                if args.no_preview:
                    continue
                cv2.putText(frame, f"{fps:.1f} FPS", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Cammy", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if vcam:
            vcam.close()
        processor.close()


def main():
    parser = argparse.ArgumentParser(prog="cammy")
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--white-balance", type=float, default=0.0,
                        help="gray-world strength, 0 off, 1 full")
    parser.add_argument("--denoise", action="store_true", help="bilateral denoise")
    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--brightness", type=float, default=0.0)
    parser.add_argument("--sharpen", type=float, default=0.0,
                        help="unsharp amount, 0 off, ~0.5 subtle")
    parser.add_argument("--face-exposure", type=float, default=0.0,
                        help="lift a dark face toward a good brightness, 0 off, 1 full")
    parser.add_argument("--skin-smooth", type=float, default=0.0,
                        help="skin smoothing strength, 0 off, ~0.5 subtle")
    parser.add_argument("--eyes", type=float, default=0.0,
                        help="eye brightening, 0 off, ~0.3 subtle")
    parser.add_argument("--teeth", type=float, default=0.0,
                        help="teeth whitening, 0 off, ~0.5 subtle")
    parser.add_argument("--detect-every", type=int, default=2,
                        help="run face/person detection every N frames (1 = every frame)")
    bg = parser.add_mutually_exclusive_group()
    bg.add_argument("--bg-blur", type=float, default=0.0,
                    help="blur the background, as a fraction of frame width "
                         "(0 off, 0.02 soft, 0.05 heavy)")
    bg.add_argument("--bg-image", type=Path, help="replace the background with an image")
    bg.add_argument("--bg-color", type=parse_color, help="replace the background with a colour, e.g. #00ff00")
    parser.add_argument("--fps", type=int, default=30, help="capture and output frame rate")
    parser.add_argument("--virtual-camera", action="store_true",
                        help="also send the processed feed to a virtual camera (Linux: v4l2loopback)")
    parser.add_argument("--virtual-device", help="virtual camera device, e.g. /dev/video10")
    parser.add_argument("--no-preview", action="store_true",
                        help="do not open the preview window; stop with Ctrl+C")
    args = parser.parse_args()
    if args.no_preview and not args.virtual_camera:
        parser.error("--no-preview needs --virtual-camera, otherwise there is no output")
    try:
        run(args)
    except VirtualCameraError as error:
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
