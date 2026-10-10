"""AI upscaling model: Real-ESRGAN 'animevideov3' x2, run through ncnn (Vulkan GPU, else CPU)."""
import multiprocessing

import cv2
import numpy as np

from .models import ensure_from_zip

ARCHIVE_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
               "realesrgan-ncnn-vulkan-20220424-ubuntu.zip")
MODEL = "realesr-animevideov3-x2"
# Pinned: the app only accepts these exact files from the (third-party) archive.
MEMBERS = {
    f"models/{MODEL}.param": "b88ff4f00ebf019a7fdac17fdd45a7fd3665d37509efc5baf2e4da2e24420a04",
    f"models/{MODEL}.bin": "548a36f9c3f4ab8da56cd3b13badf23968bee207b396dad14d04b830e5f2ab2d",
}
NEEDS_NCNN = "The AI look needs the 'ncnn' package. Install it with:  pip install ncnn"


def ensure_ai_model(directory=None):
    """Download (once, hash-checked) and return {file name: Path} for the model."""
    return ensure_from_zip(ARCHIVE_URL, MEMBERS, directory)


def pick_gpu(ncnn):
    """Index of the first real GPU (integrated or discrete), or None when there is only a CPU renderer."""
    CPU_RENDERER = 3
    for index in range(ncnn.get_gpu_count()):
        if ncnn.get_gpu_info(index).type() != CPU_RENDERER:
            return index
    return None


class Upscaler:
    """Doubles the size of a BGR image with Real-ESRGAN. Not thread-safe: use from one thread at a time."""

    def __init__(self, gpu=True, directory=None):
        try:
            import ncnn
        except ImportError as error:
            raise ImportError(NEEDS_NCNN) from error
        paths = ensure_ai_model(directory)
        # The GPU does the heavy work, so keep ncnn's CPU thread pool to one thread. Left at its default
        # (every core) it stalls other threads for a whole inference at a time: measured, OpenCV calls on
        # another thread occasionally waited ~24 ms, and a 16 ms frame became 39 ms. One thread removes the
        # stalls (95th percentile 24 ms -> 2.9 ms) at the same inference speed.
        ncnn.set_omp_num_threads(1)
        net = ncnn.Net()
        net.opt.num_threads = 1
        device = pick_gpu(ncnn) if gpu else None
        net.opt.use_vulkan_compute = device is not None
        if device is not None:
            net.set_vulkan_device(device)
            net.opt.use_fp16_packed = net.opt.use_fp16_storage = net.opt.use_fp16_arithmetic = True
        net.load_param(str(paths[f"{MODEL}.param"]))
        net.load_model(str(paths[f"{MODEL}.bin"]))
        self._ncnn, self._net, self.device = ncnn, net, device

    def upscale(self, bgr):
        h, w = bgr.shape[:2]
        mat = self._ncnn.Mat.from_pixels(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self._ncnn.Mat.PixelType.PIXEL_RGB, w, h)
        mat.substract_mean_normalize([0, 0, 0], [1 / 255.0] * 3)
        extractor = self._net.create_extractor()
        extractor.input("data", mat)
        _, out = extractor.extract("output")
        rgb = np.clip(np.array(out).transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _default_make(gpu, directory):
    return Upscaler(gpu=gpu, directory=directory)


def _serve(conn, make, gpu, directory):
    """Child process: build the model, then answer upscale requests until told to stop."""
    try:
        upscaler = make(gpu, directory)
        conn.send(("ready", getattr(upscaler, "device", None)))
    except BaseException as exc:                      # includes ImportError for a missing ncnn
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
        return
    while True:
        try:
            request = conn.recv()
        except EOFError:                              # the app is gone
            return
        if request is None:
            return
        shape, data = request
        try:
            out = upscaler.upscale(np.frombuffer(data, np.uint8).reshape(shape))
            conn.send(("ok", (out.shape, out.tobytes())))
        except Exception as exc:
            conn.send(("error", f"{type(exc).__name__}: {exc}"))


class UpscalerProcess:
    """The same interface as Upscaler, but the model lives in its own process.

    ncnn's Python binding holds Python's global interpreter lock for a whole inference (about 22 ms
    on the integrated GPU). On a thread that stalls every other thread on its way back from numpy
    or OpenCV: measured, a 16 ms frame became 39 ms and any array operation could wait a full
    inference. In a separate process (own lock) the worst stall is 3 ms. Frames cross a pipe as raw
    bytes, which for a face patch is about 0.1 MB in and 0.4 MB out.

    Starting can take a while (first use downloads the model), so construct it off the main thread.
    Use from one thread at a time.
    """

    def __init__(self, gpu=True, directory=None, make=None, context="spawn", start_timeout=300.0,
                 reply_timeout=10.0):
        ctx = multiprocessing.get_context(context)
        self._conn, child = ctx.Pipe()
        self._reply_timeout = reply_timeout
        self._process = ctx.Process(target=_serve, args=(child, make or _default_make, gpu, directory), daemon=True)
        self._process.start()
        child.close()
        kind, payload = self._receive(start_timeout, "start")
        if kind == "error":
            self.close()
            if "ImportError" in payload:
                raise ImportError(NEEDS_NCNN)
            raise RuntimeError(f"AI model could not start: {payload}")
        self.device = payload

    def _receive(self, timeout, what):
        if not self._conn.poll(timeout):
            self.close(graceful=False)                # it is not listening: do not wait for it to say goodbye
            raise RuntimeError(f"AI process did not {what} within {timeout:g} s")
        try:
            return self._conn.recv()
        except (EOFError, OSError):
            self.close(graceful=False)
            raise RuntimeError("AI process stopped unexpectedly") from None

    def upscale(self, bgr):
        array = np.ascontiguousarray(bgr)
        try:
            self._conn.send((array.shape, array.tobytes()))
        except (OSError, ValueError):                 # BrokenPipeError and friends: the process is gone
            self.close(graceful=False)
            raise RuntimeError("AI process stopped unexpectedly") from None
        kind, payload = self._receive(self._reply_timeout, "answer")
        if kind == "error":
            raise RuntimeError(payload)
        shape, data = payload
        return np.frombuffer(data, np.uint8).reshape(shape).copy()

    def close(self, graceful=True):
        """Stop the process: ask it to finish, or (graceful=False) terminate it at once. Safe to repeat."""
        if graceful:
            try:
                self._conn.send(None)
            except (OSError, ValueError):
                pass
            self._process.join(timeout=3)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=3)
        try:
            self._conn.close()
        except OSError:
            pass
