from typing import Sequence
from cropgen.processing import Line, Paragraph
from cropgen.transforms.transforms import (
    InterparagraphTransform,
)
from cropgen.transforms.helpers.line_group_info import LineGroupInfo
from PIL import Image
from shapely.affinity import translate
from shapely import Polygon


def _iterunion(*geometries: Polygon) -> Polygon:
    res = Polygon()
    for geo in geometries:
        res = res.union(geo)
    return res


def _get_polys(line_groups: Paragraph | Sequence[Line]) -> list[Polygon]:
    return [line.polygon for line in line_groups]


class CorrectIntersectionsVertically(InterparagraphTransform):
    def __init__(self, absolute_clearance: float = 5.0):
        self.clearance = absolute_clearance

    def __call__(
        self, *line_groups: Paragraph | Sequence[Line]
    ) -> tuple[list[list[Image.Image]], list[list[Polygon]]]:
        n = len(line_groups)
        img_groups = [[line.stroke_crop for line in group] for group in line_groups]
        poly_groups = [[line.polygon for line in group] for group in line_groups]

        if n < 2:
            return img_groups, poly_groups

        infos = [LineGroupInfo(line_group) for line_group in line_groups]
        y_increases_downwards = infos[-1].center[1] > infos[0].center[1]

        for i in range(1, n):
            _, miny_prev, _, maxy_prev = _iterunion(*poly_groups[i - 1]).bounds
            _, miny_curr, _, maxy_curr = _iterunion(*poly_groups[i]).bounds

            if y_increases_downwards:
                bb_gap = miny_curr - maxy_prev
            else:
                bb_gap = miny_prev - maxy_curr

            if bb_gap >= self.clearance:
                continue

            max_required_shift = self.clearance - bb_gap

            prev_poly = _iterunion(*poly_groups[i - 1])
            curr_poly = _iterunion(*poly_groups[i])

            if prev_poly.distance(curr_poly) >= self.clearance:
                continue

            y_shift = (
                max_required_shift if y_increases_downwards else -max_required_shift
            )
            poly_groups[i] = [translate(poly, yoff=y_shift) for poly in poly_groups[i]]

        return img_groups, poly_groups
