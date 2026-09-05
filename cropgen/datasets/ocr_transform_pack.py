from typing import Callable
from cropgen.transforms.intraparagraph_transforms.avoid_line_intersections import (
    AvoidLineIntersections,
)
from shapely.geometry import Polygon
from cropgen.transforms import (
    LinewiseTransform,
    InterparagraphTransform,
    IntraparagraphTransform,
)
from numpy.random import rand
import numpy as np


class OCRTransformPack:
    def __init__(self, avoid_intersections: bool = True):
        self._linewise: list[LinewiseTransform] = []
        self._linewise_prob: list[float] = []
        self._intra: list[IntraparagraphTransform] = []
        self._intra_prob: list[float] = []
        self._inter: list[InterparagraphTransform] = []
        self._inter_prob: list[float] = []
        self._avoid_intersections = avoid_intersections
        self._line_intersection_avoider = AvoidLineIntersections(0.5)

    @property
    def is_identity(self):
        return (
            sum(self._intra_prob) + sum(self._inter_prob) + sum(self._linewise_prob)
        ) == 0

    def add_transform(
        self,
        transform: (
            IntraparagraphTransform | LinewiseTransform | InterparagraphTransform
        ),
        probability: float = 1,
    ):
        """Only accepts IntraparagraphTransforms and LinewiseTransforms"""
        if (probability > 1) or (probability < 0):
            raise ValueError("probability must be between 0 and 1")
        if isinstance(transform, LinewiseTransform):
            self._linewise.append(transform)
            self._linewise_prob.append(probability)
        elif isinstance(transform, IntraparagraphTransform):
            self._intra.append(transform)
            self._intra_prob.append(probability)
        elif isinstance(transform, InterparagraphTransform):
            self._inter.append(transform)
            self._inter_prob.append(probability)
        else:
            raise ValueError(
                "Can only add LinewiseTransform and IntraparagraphTransform instances, but got "
                f"unsupported type {type(transform)}."
            )

    def __call__(
        self,
        paragraph_eq_list: list[tuple[list[np.ndarray], list[Polygon]]],
    ) -> list[tuple[list[np.ndarray], list[Polygon]]]:
        """
        Takes as input a list of 2-tuples (list of images, list of polygons) that represent the crop
        and polygons of each paragraph
        """

        prob_ok: Callable[[float], bool] = lambda p: (
            (p == 1) or ((p <= 1) and (rand() < p))
        )

        for i, (images, polygons) in enumerate(paragraph_eq_list):
            # for each paragraph

            for i, (image, polygon) in enumerate(zip(images, polygons)):
                # for each line

                for linewise_transform, p in zip(self._linewise, self._linewise_prob):
                    if prob_ok(p):
                        images[i], polygons[i] = linewise_transform(image, polygon)

            # for each paragraph
            for intraparagraph_transform, p in zip(self._intra, self._intra_prob):
                if prob_ok(p):
                    images, polygons = intraparagraph_transform((images, polygons))

            if self._avoid_intersections:
                images, polygons = self._line_intersection_avoider((images, polygons))

            paragraph_eq_list[i] = (images, polygons)

        for interparagraph in self._inter:
            if prob_ok(p):
                paragraph_eq_list = list(zip(*interparagraph(paragraph_eq_list)))

        return paragraph_eq_list
