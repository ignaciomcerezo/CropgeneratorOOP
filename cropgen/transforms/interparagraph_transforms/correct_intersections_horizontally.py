from shapely.geometry import Polygon
from cropgen.processing.line import Line
from typing import Sequence
from cropgen.transforms.interparagraph_transforms.correct_intersections_vertically import (
    _iterunion,
)
from cropgen.processing import Paragraph
from cropgen.transforms.transforms import (
    InterparagraphTransform,
)
from cropgen.transforms.helpers.line_group_info import LineGroupInfo
from shapely.affinity import translate
from shapely import intersection
from PIL import Image


class CorrectIntersectionsHorizontally(InterparagraphTransform):
    """
    Moves paragraphs away from each other horizontally to satisfy a minimum
    clearance constraint, assuming paragraphs are strictly ordered top-to-bottom.
    """

    def __init__(self, absolute_clearance: float = 5):
        self.clearance = absolute_clearance

    def __call__(
        self, *line_groups: Paragraph | Sequence[Line]
    ) -> tuple[list[list[Image.Image]], list[list[Polygon]]]:
        img_groups = [
            [line.stroke_crop for line in line_group] for line_group in line_groups
        ]
        poly_groups = [
            [line.polygon for line in line_group] for line_group in line_groups
        ]

        if len(line_groups) < 2:
            return img_groups, poly_groups

        infos = [LineGroupInfo(line_group) for line_group in line_groups]
        nu = _detect_preferred_side(*infos)

        max_iterations = 100
        iteration = 0
        movement = True

        while movement and iteration < max_iterations:
            movement = False
            for i in range(1, len(line_groups)):
                prev_union = _iterunion(*poly_groups[i - 1])
                curr_union = _iterunion(*poly_groups[i])

                if not prev_union.intersects(curr_union):
                    continue

                intersect_geom = prev_union.intersection(curr_union)
                min_x, _, max_x, _ = intersect_geom.bounds
                intersection_depth = max_x - min_x
                w = (intersection_depth + self.clearance) / 2.0
                eta = 1 - (2 * (i % 2))
                shift = eta * nu * w

                poly_groups[i] = [
                    translate(poly, xoff=shift) for poly in poly_groups[i]
                ]
                poly_groups[i - 1] = [
                    translate(poly, xoff=-shift) for poly in poly_groups[i - 1]
                ]

                movement = True

            iteration += 1

        return img_groups, poly_groups


def _detect_preferred_side(*infos: LineGroupInfo):
    avg_center = sum([info.center[0] for info in infos]) / len(infos)
    even = sum([info.center[0] for info in infos[::2]]) / len(infos[::2])

    if even > avg_center:
        return 1
    else:
        return -1
