from prompt_toolkit.key_binding.key_bindings import NotImplementedOrNone
from cropgen.processing.Paragraph import Paragraph
from shapely.ops import unary_union
from abc import ABC, abstractmethod


class _ParagraphInfo:
    """
    Some helpful data to calculate layouts.
    """

    def __init__(self, paragraph: Paragraph):

        # x0,y0,xf,yf
        self.bounds: list[tuple[float, float, float, float]] = [
            box.polygon.bounds for box in paragraph.image_boxes
        ]
        self.centroid: tuple[float, float] = unary_union(
            box.polygon for box in paragraph.image_boxes
        ).centroid
        self.n_points = len(paragraph)
        self.x_deltas = [
            self.bounds[i + 1][0] - self.bounds[i][0] for i in range(len(paragraph) - 1)
        ]
        self.y_deltas = [
            self.bounds[i + 1][1] - self.bounds[i][1] for i in range(len(paragraph) - 1)
        ]


class IntraparagraphTransform(ABC):
    """
    Base class used to modify layouts for individual paragraphs.
    For example line shears or line-by-line distortions could be
    implemented like this.
    """

    def __call__(self, paragraph: Paragraph) -> None:
        raise NotImplementedError


class InterparagraphTransform(ABC):
    """
    Base class used to modify layouts for complete documents.
    For example this could be used to separate paragraphs between them,
    rotate them globally, etc.
    """

    def __call__(self, *paragraphs: Paragraph) -> None:
        raise NotImplementedError
