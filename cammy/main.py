"""Live preview with FPS counter. Press q or Esc to quit."""
import argparse
import time

import cv2

from .capture import Capture
from .pipeline import Pipeline


def run(camera=0):
    pipeline = Pipeline()
    fps, frames, t0 = 0.0, 0, time.perf_counter()
    with Capture(camera) as cap:
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
    run(parser.parse_args().camera)


if __name__ == "__main__":
    main()
