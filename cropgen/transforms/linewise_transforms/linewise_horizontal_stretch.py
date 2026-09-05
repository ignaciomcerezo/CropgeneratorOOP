from cropgen.shared.parameters import Parameter
from shapely.affinity import scale
from shapely.geometry import Polygon
import cv2
from cropgen.ocr_units import OCRLine
from cropgen.transforms.transforms import LinewiseTransform
import numpy as np


class LinewiseHorizontalStretch(LinewiseTransform):
    def __init__(self, scale_factor: Parameter | float = 1.2):

        self.scale_factor = Parameter(scale_factor)

    def __call__(
        self, image: np.ndarray, polygon: Polygon
    ) -> tuple[np.ndarray, Polygon]:
        height, width = image.shape[:2]
        new_width = max(1, round(width * abs(self.scale_factor())))
        image = image
        stretched_image = cv2.resize(
            image,
            (new_width, height),
            interpolation=cv2.INTER_LINEAR,
        )

        min_x, min_y, max_x, max_y = polygon.bounds
        center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
        stretched_polygon = scale(
            polygon,
            xfact=self.scale_factor,
            yfact=1.0,
            origin=center,
        )

        return stretched_image, stretched_polygon
