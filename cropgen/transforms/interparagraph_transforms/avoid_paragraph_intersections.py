from typing import Sequence
from cropgen.transforms.helpers.line_group_info import LineGroupInfo
from cropgen.transforms.helpers.polygon_separation import (
    Vector2D,
    separate_polygons,
)
from cropgen.ocr_units import OCRLine, OCRParagraph
from cropgen.transforms.transforms import (
    InterparagraphTransform,
    line_group_equivalent_type,
)
from shapely import Polygon
from shapely.affinity import translate


class AvoidParagraphIntersections(InterparagraphTransform):
    """
    Resolves overlaps between paragraphs on a page.

    The underlying algorithm is very different from that of AvoidLineIntersections,
    as that one only applies to single paragraphs and assumes the elements
    (in that, lines; in this, polygons) can be ordered in a 1D line (in
    the reading direction). his is obviously not the case for nearly all sufficiently
    complex multi-paragraph page layouts, and therefore requires a different approach.
    Overlaps are resolved with genuine 2D separation (see `polygon_separation.py`),
    which makes no assumption about a shared axis or ordering between paragraphs.
    """

    def __init__(
        self,
        delta: float = 0.5,
        max_iterations: int = 1000,
        damping: float = 0.5,
    ):
        self.delta = delta
        self.max_iterations = max_iterations
        self.damping = damping

    def __call__(
        self,
        line_equivalent_groups: Sequence[line_group_equivalent_type],
    ):
        image_groups, polygon_groups = self._extract_polygon_and_image_groups(
            line_equivalent_groups
        )

        # we use convex hulls: we dont want intersections to be so fine-grained that a far line
        # from a paragraph could be interleaved in the space between the lines of another.
        union_hulls = [
            LineGroupInfo.polygon_union(group).convex_hull for group in polygon_groups
        ]

        _, shifts = self.call_polygons(union_hulls)

        for polygon_group, shift in zip(polygon_groups, shifts):
            for i in range(len(polygon_group)):
                polygon_group[i] = translate(polygon_group[i], shift[0], shift[1])

        return image_groups, polygon_groups

    def call_polygons(
        self,
        polygons: list[Polygon],
    ) -> tuple[list[Polygon], list[Vector2D]]:
        """
        Separates a set of (paragraph/hull) polygons in 2D so that none
        overlap and all pairwise gaps are at least self.delta.

        Returns the separated polygons together with the per-polygon
        cumulative shift, so the same shift can be applied to individual
        paragraph components.
        """
        return separate_polygons(
            polygons,
            delta=self.delta,
            max_iterations=self.max_iterations,
            damping=self.damping,
        )
