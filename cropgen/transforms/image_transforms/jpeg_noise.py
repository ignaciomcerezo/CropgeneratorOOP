import cv2
import numpy as np

from cropgen.shared.parameters import Parameter
from cropgen.transforms.transforms import GlobalImageTransform


class ResolutionNoise(GlobalImageTransform):
    def __init__(self, quality: Parameter | float = 75):
        self.quality = Parameter(quality)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        quality = min(100, max(0, int(self.quality())))
        could_encode, encoded_img = cv2.imencode(
            ".jpg", image, (cv2.IMWRITE_JPEG_QUALITY, quality)
        )
        if not could_encode:
            raise ValueError("JPEG compression failed")

        decoded = cv2.imdecode(
            encoded_img, cv2.IMREAD_GRAYSCALE if image.ndim == 2 else cv2.IMREAD_COLOR
        )
        if decoded is None:
            raise ValueError("JPEG decompression failed")
        return decoded
