import cv2
from cropgen.shared.parameters import Parameter
import numpy as np
from cropgen.processing.line import Line
from cropgen.transforms.transforms import LinewiseTransform
from shapely.geometry import Polygon


class Blur(LinewiseTransform):
    def __init__(self, radius: Parameter | float = 2.0):
        self.radius: Parameter = Parameter(radius)

    def __call__(
        self, image: np.ndarray, polygon: Polygon
    ) -> tuple[np.ndarray, Polygon]:
        return (
            cv2.GaussianBlur(image, (0, 0), sigmaX=self.radius()),
            polygon,
        )
