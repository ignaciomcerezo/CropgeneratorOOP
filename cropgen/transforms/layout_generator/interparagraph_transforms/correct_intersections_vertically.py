from typing import Sequence
from cropgen.processing.image_box import ImageBox
from cropgen.processing import Paragraph
from cropgen.transforms.layout_generator import (
    InterparagraphTransform,
    ParagraphInfo,
)
from shapely.affinity import translate
from shapely import Polygon


def _iterunion(*geometries: Polygon) -> Polygon:
    res = Polygon()
    for geo in geometries:
        res = res.union(geo)
    return res


def _get_polys(line_groups: Paragraph | Sequence[ImageBox]) -> list[Polygon]:
    return [box.polygon for box in line_groups]


class CorrectIntersectionsVertically(InterparagraphTransform):
    def __init__(self, absolute_clearance: float = 5.0):
        self.clearance = absolute_clearance

    def __call__(self, *line_groups: Paragraph | Sequence[ImageBox]) -> None:
        n = len(line_groups)
        if n < 2:
            return

        infos = [ParagraphInfo(line_group) for line_group in line_groups]
        y_increases_downwards = infos[-1].center[1] > infos[0].center[1]

        for i in range(1, n):

            _, miny_prev, _, maxy_prev = infos[i - 1].union_bounds
            _, miny_curr, _, maxy_curr = infos[i].union_bounds

            if y_increases_downwards:
                bb_gap = miny_curr - maxy_prev
            else:
                bb_gap = miny_prev - maxy_curr
            if bb_gap >= self.clearance:
                continue

            max_required_shift = self.clearance - bb_gap

            prev_poly = _iterunion(*_get_polys(line_groups[i - 1]))
            curr_poly = _iterunion(*_get_polys(line_groups[i]))

            if prev_poly.distance(curr_poly) >= self.clearance:
                continue

            y_shift = (
                max_required_shift if y_increases_downwards else -max_required_shift
            )

            for box in line_groups[i]:
                box.polygon = translate(box.polygon, yoff=y_shift)

            infos[i] = ParagraphInfo(line_groups[i])
