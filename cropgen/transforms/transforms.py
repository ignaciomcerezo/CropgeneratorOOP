from cropgen.processing.annotated_page import AnnotatedPage
from cropgen.processing.line import Line
from shapely.geometry import Polygon
from typing import Collection, Sequence
from cropgen.processing import Paragraph, Line
from abc import ABC, abstractmethod
import shapely
from PIL import Image
import numpy as np
from shapely.affinity import translate


class LinewiseTransform(ABC):
    """
    Base class used to modify single linges, for example single line
    distortions and stretching.
    """

    @abstractmethod
    def __call__(self, box: Line) -> tuple[Image.Image, shapely.Polygon]:
        raise NotImplementedError

    def bulk_transform(
        self, line_group: Paragraph | Sequence[Line]
    ) -> tuple[list[Image.Image], list[Polygon]]:
        new_imgs, new_polygons = [], []

        for box in line_group:
            new_img, new_polygon = self(box)
            new_imgs.append(new_img)
            new_polygons.append(new_polygon)

        return new_imgs, new_polygons

    def in_place(self, box: Line) -> None:
        img, poly = self(box)
        box.stroke_crop = img
        box.polygon = poly


class IntraparagraphTransform(ABC):
    """
    Base class used to modify layouts for individual paragraphs.
    For example line shears or paragraph rotations. or line-by-line distortions could be
    implemented like this.
    """

    @abstractmethod
    def __call__(
        self, line_group: Paragraph | Sequence[Line]
    ) -> tuple[list[Image.Image], list[shapely.Polygon]]:
        raise NotImplementedError

    @staticmethod
    def from_linewise(transform: LinewiseTransform):
        return IntraparagraphFromLinewiseTransform(transform)

    def in_place(self, line_group: Paragraph | Sequence[Line]) -> None:
        imgs, polys = self(line_group)
        for line, img, poly in zip(line_group, imgs, polys):
            line.stroke_crop = img
            line.polygon = poly


class IntraparagraphFromLinewiseTransform(IntraparagraphTransform):
    """
    Linewise transform turned paragraph transform.
    """

    def __init__(self, transform: LinewiseTransform):
        self._transform = transform

    def __call__(
        self, line_group: Paragraph | Sequence[Line]
    ) -> tuple[list[Image.Image], list[shapely.Polygon]]:
        return self._transform.bulk_transform(line_group)


class InterparagraphTransform(ABC):
    """
    Base class used to modify layouts for complete documents.
    For example this could be used to separate paragraphs between them,
    rotate them globally, etc.
    """

    @abstractmethod
    def __call__(
        self, *line_groups: Paragraph | Sequence[Line]
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
