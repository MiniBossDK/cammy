"""AI upscaling model: Real-ESRGAN 'animevideov3' x2, run through ncnn (Vulkan GPU, else CPU)."""
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
        net = ncnn.Net()
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
