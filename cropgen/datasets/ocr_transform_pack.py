from typing import Callable
from cropgen.datasets.helpers.intersection_correction import (
    avoid_line_intersections,
    avoid_paragraph_intersections,
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
    ) -> tuple[list[np.ndarray], list[Polygon]]:
        """
        Takes as input a list of 2-tuples (list of images, list of polygons) that represent the crop
        and polygons of each paragraph
        """

        prob_ok: Callable[[float], bool] = lambda p: (
            (p == 1) or ((p <= 1) and (rand() < p))
        )

        for i in range(len(paragraph_eq_list)):
            images, polygons = paragraph_eq_list[i]

            # Process each line
            for j in range(len(images)):
                cur_image = images[j]
                cur_polygon = polygons[j]
                for linewise_transform, p in zip(self._linewise, self._linewise_prob):
                    if prob_ok(p):
                        cur_image, cur_polygon = linewise_transform(
                            cur_image, cur_polygon
                        )
                images[j] = cur_image
                polygons[j] = cur_polygon

            current_paragraph = (images, polygons)

            # Process paragraph-level transforms
            for intraparagraph_transform, p in zip(self._intra, self._intra_prob):
                if prob_ok(p):
                    current_paragraph = intraparagraph_transform(current_paragraph)

            paragraph_eq_list[i] = current_paragraph

        # Process interparagraph transforms
        for interparagraph, p in zip(self._inter, self._inter_prob):
            if prob_ok(p):
                paragraph_eq_list = list(zip(*interparagraph(paragraph_eq_list)))

        polys_by_par = [pp[1] for pp in paragraph_eq_list]
        polygons = sum(polys_by_par, start=[])

        if self._avoid_intersections:
            polygons = avoid_line_intersections(polygons)

            # by-paragraph
            # polys_by_par = [avoid_line_intersections(p) for p in polys_by_par]
            # if len(polys_by_par) > 1:
            #     polys_by_par = avoid_paragraph_intersections(polys_by_par)

        crops: list[np.ndarray] = sum(
            (paragraph_eq[0] for paragraph_eq in paragraph_eq_list), start=[]
        )

        return crops, polygons
