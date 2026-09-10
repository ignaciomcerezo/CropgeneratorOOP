import cv2
import numpy as np
from shapely.geometry import Polygon

from cropgen.shared.parameters import Parameter
from cropgen.transforms.transforms import LineTransform


class Blur(LineTransform):
    def __init__(self, radius: Parameter | float = 2.0):
        self.radius: Parameter = Parameter(radius)
        self.may_cause_intersections = False

    def __call__(
        self, image: np.ndarray, polygon: Polygon
    ) -> tuple[np.ndarray, Polygon]:
        return (
            cv2.GaussianBlur(image, (0, 0), sigmaX=self.radius()),
            polygon,
        )
