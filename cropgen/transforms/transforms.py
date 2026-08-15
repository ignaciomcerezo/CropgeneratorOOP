from __future__ import annotations
from shapely.geometry import Polygon
from typing import Collection, Sequence, TYPE_CHECKING
from abc import ABC, abstractmethod
import shapely
from PIL import Image
import numpy as np
from shapely.affinity import translate
from copy import copy

if TYPE_CHECKING:
    from cropgen.processing import AnnotatedPage, Paragraph, Line


class LinewiseTransform(ABC):
    """
    Base class used to modify single linges, for example single line
    distortions and stretching.
    """

    @abstractmethod
    def __call__(
        self, image: Image.Image, polygon: Polygon
    ) -> tuple[Image.Image, shapely.Polygon]:
        raise NotImplementedError

    def bulk_transform(
        self,
        line_equivalent_group: (
            Paragraph
            | Paragraph
            | Sequence[Line]
            | tuple[Sequence[Image.Image], Sequence[Polygon]]
        ),
    ) -> tuple[list[Image.Image], list[Polygon]]:
        new_imgs, new_polygons = [], []

        for img, poly in zip(*self._extract_polygons_and_images(line_equivalent_group)):
            new_img, new_polygon = self(img, poly)
            new_imgs.append(new_img)
            new_polygons.append(new_polygon)

        return new_imgs, new_polygons

    def in_place(self, line: Line) -> None:
        """
        Transforms the polygon and image of a Line instance in-place.
        """
        img, poly = self(line.stroke_crop, line.polygon)
        line.stroke_crop = img
        line.polygon = poly

    @staticmethod
    def _extract_polygons_and_images(
        line_equivalent_group: (
            Paragraph
            | Sequence[Line]
            | tuple[Sequence[Image.Image], Sequence[shapely.Polygon]]
        ),
    ) -> tuple[list[Image.Image], list[shapely.Polygon]]:
        if isinstance(line_equivalent_group, tuple) and isinstance(
            line_equivalent_group[0], list
        ):
            return (
                copy(line_equivalent_group[0]),
                copy(line_equivalent_group[1]),
            )  # ty: ignore[invalid-return-type]
        return (
            [
                line.stroke_crop  # ty: ignore[unresolved-attribute]
                for line in line_equivalent_group
            ],
            [
                line.polygon  # ty: ignore[unresolved-attribute]
                for line in line_equivalent_group
            ],
        )

    @staticmethod
    def _extract_polygon_and_image(line: Line) -> tuple[Image.Image, shapely.Polygon]:
        return line.stroke_crop, line.polygon


class IntraparagraphTransform(ABC):
    """
    Base class used to modify layouts for individual paragraphs.
    For example line shears or paragraph rotations. or line-by-line distortions could be
    implemented like this.
    """

    @abstractmethod
    def __call__(
        self,
        line_equivalent_group: (
            Paragraph
            | Paragraph
            | Sequence[Line]
            | tuple[Sequence[Image.Image], Sequence[Polygon]]
        ),
    ) -> tuple[list[Image.Image], list[shapely.Polygon]]:
        raise NotImplementedError

    @staticmethod
    def from_linewise(transform: LinewiseTransform):
        return IntraparagraphFromLinewiseTransform(transform)

    def in_place(
        self,
        line_group: Paragraph | Sequence[Line],
    ) -> None:
        imgs, polys = self(line_group)
        for line, img, poly in zip(line_group, imgs, polys):
            line.stroke_crop = img
            line.polygon = poly

    @staticmethod
    def _extract_polygons_and_images(
        line_equivalent_group: (
            Paragraph
            | Sequence[Line]
            | tuple[Sequence[Image.Image], Sequence[shapely.Polygon]]
        ),
    ) -> tuple[list[Image.Image], list[shapely.Polygon]]:
        if isinstance(line_equivalent_group, tuple) and isinstance(
            line_equivalent_group[0], list
        ):
            return (
                copy(line_equivalent_group[0]),
                copy(line_equivalent_group[1]),
            )  # ty: ignore[invalid-return-type]
        return (
            [
                line.stroke_crop  # ty: ignore[unresolved-attribute]
                for line in line_equivalent_group
            ],
            [
                line.polygon  # ty: ignore[unresolved-attribute]
                for line in line_equivalent_group
            ],
        )

    @staticmethod
    def _extract_polygon_and_image(line: Line) -> tuple[Image.Image, shapely.Polygon]:
        return line.stroke_crop, line.polygon


class IntraparagraphFromLinewiseTransform(IntraparagraphTransform):
    """
    Linewise transform turned paragraph transform.
    """

    def __init__(self, transform: LinewiseTransform):
        self._transform = transform

    def __call__(
        self,
        line_equivalent_group: (
            Paragraph
            | Paragraph
            | Sequence[Line]
            | tuple[Sequence[Image.Image], Sequence[Polygon]]
        ),
    ) -> tuple[list[Image.Image], list[shapely.Polygon]]:
        return self._transform.bulk_transform(line_equivalent_group)


class InterparagraphTransform(ABC):
    """
    Base class used to modify layouts for complete documents.
    For example this could be used to separate paragraphs between them,
    rotate them globally, etc.
    """

    @abstractmethod
    def __call__(
        self,
        *line_equivalent_groups: Paragraph
        | Sequence[Line]
        | tuple[Sequence[Image.Image], Sequence[Polygon]],
    ) -> tuple[list[list[Image.Image]], list[list[Polygon]]]:
        raise NotImplementedError

    def in_place(self, *line_groups: Paragraph | Sequence[Line]) -> None:
        img_groups, poly_groups = self(*line_groups)

        for line_group, img_group, poly_group in zip(
            line_groups, img_groups, poly_groups
        ):
            for line, img, poly in zip(line_group, img_group, poly_group):
                line.stroke_crop = img
                line.polygon = poly

    @staticmethod
    def _extract_polygon_and_image_groups(
        line_equivalent_groups: tuple[
            Paragraph
            | Sequence[Line]
            | tuple[Sequence[Image.Image], Sequence[Polygon]],
            ...,
        ],
    ) -> tuple[list[list[Image.Image]], list[list[Polygon]]]:

        groups = [
            IntraparagraphTransform._extract_polygons_and_images(element)
            for element in line_equivalent_groups
        ]
        image_groups = []
        polygon_groups = []
        for group in groups:
            image_groups.append(group[0])
            polygon_groups.append(group[1])

        return image_groups, polygon_groups
