from abc import ABC
from typing import Callable
from PIL import Image
from shapely import Polygon
from cropgen.processing import ImageBox, Paragraph
import numpy as np


class LineTransform(ABC):
    """
    Transform to be applied to a single ImageBox, and returns the transformed
    image and polygon.
    """

    def __call__(self, box: ImageBox) -> tuple[np.ndarray, Polygon]:
        raise NotImplementedError


class OverlayTransform(ABC):
    """
    Transform to be applied to the overlay (all strokes from a sample) before
    adding the background
    """

    def __call__(self, overlay: np.ndarray) -> np.ndarray:
        raise NotImplementedError
