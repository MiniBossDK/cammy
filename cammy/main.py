"""Live preview with FPS counter. Press q or Esc to quit."""
import argparse
import time

import cv2

from .capture import Capture
from .effects.brightness_contrast import BrightnessContrast
from .effects.denoise import Denoise
from .effects.sharpen import Sharpen
from .effects.white_balance import WhiteBalance
from .pipeline import Pipeline


def build_pipeline(args):
    """Order matters: denoise before sharpen so noise is not amplified."""
    pipeline = Pipeline()
    if args.white_balance:
        pipeline.add(WhiteBalance(args.white_balance))
    if args.denoise:
        pipeline.add(Denoise())
    if args.contrast != 1.0 or args.brightness:
        pipeline.add(BrightnessContrast(args.contrast, args.brightness))
    if args.sharpen:
        pipeline.add(Sharpen(args.sharpen))
    return pipeline


def run(args):
    pipeline = build_pipeline(args)
    fps, frames, t0 = 0.0, 0, time.perf_counter()
    with Capture(args.camera) as cap:
        while True:
            frame = cap.read()
            if frame is None:
                print("frame read failed")
                break
            frame = pipeline(frame)

            frames += 1
            now = time.perf_counter()
            if now - t0 >= 0.5:
                fps, frames, t0 = frames / (now - t0), 0, now
            cv2.putText(frame, f"{fps:.1f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Cammy", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    cv2.destroyAllWindows()


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
    run(parser.parse_args())


if __name__ == "__main__":
    main()
