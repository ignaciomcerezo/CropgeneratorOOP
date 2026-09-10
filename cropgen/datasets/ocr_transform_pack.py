import functools
import operator

import numpy as np
from numpy.random import rand
from shapely.geometry import Polygon

from cropgen.datasets.helpers.intersection_correction import (
    avoid_line_intersections,
    avoid_paragraph_intersections,
)
from cropgen.transforms import (
    InterparagraphTransform,
    IntraparagraphTransform,
    LinewiseTransform,
)


class OCRTransformPack:
    def __init__(self, avoid_intersections: bool = True):
        self._linewise: list[LinewiseTransform] = []
        self._linewise_prob: list[float] = []
        self._intra: list[IntraparagraphTransform] = []
        self._intra_prob: list[float] = []
        self._inter: list[InterparagraphTransform] = []
        self._inter_prob: list[float] = []
        self.avoid_intersections = avoid_intersections

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
                "Can only add LinewiseTransform and IntraparagraphTransform instances, "
                "but got "
                f"unsupported type {type(transform)}."
            )

    def should_call(self, p):
        return (p == 1) or ((p <= 1) and (rand() < p))

    def __call__(
        self,
        paragraph_eq_list: list[tuple[list[np.ndarray], list[Polygon]]],
    ) -> tuple[list[np.ndarray], list[Polygon]]:
        """
        Takes as input a list of 2-tuples (list of images, list of polygons) that represent the crop
        and polygons of each paragraph
        """

        for i in range(len(paragraph_eq_list)):
            images, polygons = paragraph_eq_list[i]

            # Process each line
            for j in range(len(images)):
                cur_image = images[j]
                cur_polygon = polygons[j]
                for linewise_transform, p in zip(
                    self._linewise, self._linewise_prob, strict=True
                ):
                    if self.should_call(p):
                        cur_image, cur_polygon = linewise_transform(
                            cur_image, cur_polygon
                        )
                images[j] = cur_image
                polygons[j] = cur_polygon

            current_paragraph = (images, polygons)

            # Process paragraph-level transforms
            for intraparagraph_transform, p in zip(
                self._intra, self._intra_prob, strict=True
            ):
                if self.should_call(p):
                    current_paragraph = intraparagraph_transform(current_paragraph)

            paragraph_eq_list[i] = current_paragraph

        # Process interparagraph transforms
        for interparagraph, p in zip(self._inter, self._inter_prob, strict=True):
            if self.should_call(p):
                paragraph_eq_list = list(
                    zip(*interparagraph(paragraph_eq_list), strict=True)
                )

        polys_by_par = [
            tuple_images_polygons[1] for tuple_images_polygons in paragraph_eq_list
        ]

        if self.avoid_intersections:

            for (
                i,
                par_polys,
            ) in enumerate(polys_by_par):
                polys_by_par[i] = avoid_line_intersections(par_polys)

            if len(polys_by_par) > 1:
                polys_by_par = avoid_paragraph_intersections(polys_by_par)

        polygons = functools.reduce(operator.iadd, polys_by_par, [])

        crops: list[np.ndarray] = functools.reduce(
            operator.iadd, (paragraph_eq[0] for paragraph_eq in paragraph_eq_list), []
        )

        return crops, polygons
