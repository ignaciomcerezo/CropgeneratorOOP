from cropgen.transforms.helpers.line_group_info import LineGroupInfo
from typing import Sequence
from cropgen.processing.line import Line
from cropgen.processing.paragraph import Paragraph
from cropgen.transforms.transforms import InterparagraphTransform
from cropgen.transforms.intraparagraph_transforms.avoid_line_intersections import (
    AvoidLineIntersections as _AvoidLineIntersections,
)
from PIL import Image
from shapely import Polygon
from shapely.affinity import translate


class AvoidParagraphIntersections(InterparagraphTransform):

    def __init__(self, delta: float = 0.5, max_iterations: int = 1000):
        self._inner_transform = _AvoidLineIntersections(delta, max_iterations)

    @property
    def delta(self) -> float:
        return self._inner_transform.delta

    @delta.setter
    def delta(self, value: float) -> None:
        self._inner_transform.delta = value

    @property
    def max_iterations(self) -> int:
        return self._inner_transform.max_iterations

    @max_iterations.setter
    def max_iterations(self, value: int) -> None:
        self._inner_transform.max_iterations = value

    def __call__(
        self,
        *line_equivalent_groups: (
            Paragraph | Sequence[Line] | tuple[Sequence[Image.Image], Sequence[Polygon]]
        ),
    ):
        image_groups, polygon_groups = self._extract_polygon_and_image_groups(
            line_equivalent_groups
        )

        union_polygons = [
            LineGroupInfo.polygon_union(group) for group in polygon_groups
        ]
        avg_rotations = [
            LineGroupInfo.from_polygons(group).avg_rotation for group in polygon_groups
        ]

        _, shifts = self._inner_transform.call_polygons(union_polygons, avg_rotations)

        for polygon_group, shift in zip(polygon_groups, shifts):
            for i in range(len(polygon_group)):
                polygon_group[i] = translate(polygon_group[i], shift[0], shift[1])

        return image_groups, polygon_groups
