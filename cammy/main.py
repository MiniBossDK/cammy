"""Live preview with FPS counter. Press q or Esc to quit."""
import argparse
import time

import cv2

from .capture import Capture
from .effects.brightness_contrast import BrightnessContrast
from .effects.denoise import Denoise
from .effects.eyes_teeth import EyeBrighten, TeethWhiten
from .effects.face_exposure import FaceExposure
from .effects.sharpen import Sharpen
from .effects.skin_smooth import SkinSmooth
from .effects.white_balance import WhiteBalance
from .face import FaceTracker
from .pipeline import Pipeline


def wants_face(args):
    return bool(args.face_exposure or args.skin_smooth or args.eyes or args.teeth)


def build_pipeline(args, tracker=None):
    """Order matters: denoise before sharpen so noise is not amplified, and
    exposure runs early so later effects see a well-lit face."""
    pipeline = Pipeline()
    if args.white_balance:
        pipeline.add(WhiteBalance(args.white_balance))
    if args.face_exposure:
        pipeline.add(FaceExposure(tracker, strength=args.face_exposure))
    if args.denoise:
        pipeline.add(Denoise())
    if args.skin_smooth:
        pipeline.add(SkinSmooth(tracker, strength=args.skin_smooth))
    if args.contrast != 1.0 or args.brightness:
        pipeline.add(BrightnessContrast(args.contrast, args.brightness))
    if args.eyes:
        pipeline.add(EyeBrighten(tracker, amount=args.eyes))
    if args.teeth:
        pipeline.add(TeethWhiten(tracker, amount=args.teeth))
    if args.sharpen:
        pipeline.add(Sharpen(args.sharpen))
    return pipeline


def run(args):
    tracker = FaceTracker() if wants_face(args) else None
    pipeline = build_pipeline(args, tracker)
    fps, frames, t0 = 0.0, 0, time.perf_counter()
    with Capture(args.camera) as cap:
        while True:
            frame = cap.read()
            if frame is None:
                print("frame read failed")
                break
            if tracker:
                tracker.update(frame)  # landmarks from the raw frame
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
    if tracker:
        tracker.close()


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
    run(parser.parse_args())


if __name__ == "__main__":
    main()
