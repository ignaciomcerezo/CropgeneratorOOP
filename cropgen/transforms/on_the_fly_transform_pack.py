from cropgen.transforms.intraparagraph_transforms.avoid_line_intersections import (
    AvoidLineIntersections,
)
from shapely.geometry import Polygon
from cropgen.transforms.transforms import LinewiseTransform, InterparagraphTransform
from cropgen.transforms import IntraparagraphTransform
from numpy.random import rand
import numpy as np


class OCROnTheFlyTransformPack:
    def __init__(self, avoid_intersections: bool = True):
        self._linewise: list[LinewiseTransform] = []
        self._linewise_p: list[float] = []
        self._intraparagraph: list[IntraparagraphTransform] = []
        self._intraparagraph_p: list[float] = []
        self._avoid_intersections = avoid_intersections

    def add_transform(
        self,
        transform: IntraparagraphTransform | LinewiseTransform,
        probability: float = 1,
    ):
        if (probability > 1) or (probability < 0):
            raise ValueError("probability must be between 0 and 1")
        if isinstance(transform, LinewiseTransform):
            self._linewise.append(transform)
            self._linewise_p.append(probability)
        elif isinstance(transform, IntraparagraphTransform):
            self._intraparagraph.append(transform)
            self._intraparagraph_p.append(probability)
        elif isinstance(transform, InterparagraphTransform):
            raise ValueError(
                "Can only add LinewiseTransform and IntraparagraphTransform instances "
                "but got InterparagraphTransform, which is intended only for LayoutGenerator."
            )
        else:
            raise ValueError(
                "Can only add LinewiseTransform and IntraparagraphTransform instances, but got "
                f"unsupported type {type(transform)}."
            )

    def __call__(
        self, images: list[np.ndarray], polygons: list[Polygon]
    ) -> tuple[list[np.ndarray], list[Polygon]]:

        for i, (image, polygon) in enumerate(zip(images, polygons)):
            for linewise_transform, p in zip(self._linewise, self._linewise_p):
                if (p == 1) or ((p <= 1) and (rand() < p)):
                    images[i], polygons[i] = linewise_transform(image, polygon)

        for intraparagraph_transform, p in zip(
            self._intraparagraph, self._intraparagraph_p
        ):
            if (p == 1) or ((p <= 1) and (rand() < p)):
                images, polygons = intraparagraph_transform((images, polygons))

        if self._avoid_intersections:
            ali = AvoidLineIntersections(0.5)
            images, polygons = ali((images, polygons))

        return images, polygons
