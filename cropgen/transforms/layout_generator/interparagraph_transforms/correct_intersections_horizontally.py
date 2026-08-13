from cropgen.processing import Paragraph
from cropgen.transforms.layout_generator import (
    InterparagraphTransform,
    ParagraphInfo,
)
from shapely.affinity import translate
from shapely import intersection


class CorrectIntersectionsHorizontally:
    """
    Moves paragraphs away from each other horizontally to satisfy a minimum
    clearance constraint, assuming paragraphs are strictly ordered top-to-bottom.
    """

    def __init__(self, absolute_clearance: float = 5):
        self.clearance = absolute_clearance

    def __call__(self, *paragraphs: Paragraph) -> None:
        n = len(paragraphs)
        if n < 2:
            return

        infos = [ParagraphInfo(paragraph) for paragraph in paragraphs]

        nu = _detect_preferred_side(*infos)

        # running this loop twice always yields a non-intersecting result,
        # but is probably unnecessary
        max_iterations = 100
        iteration = 0
        movement = True

        while movement and iteration < max_iterations:
            movement = False
            for i in range(1, len(paragraphs)):
                prev = infos[i - 1]
                curr = infos[i]

                if prev.union_polygon.intersects(curr.union_polygon):

                    intersect_geom = prev.union_polygon.intersection(curr.union_polygon)
                    min_x, _, max_x, _ = intersect_geom.bounds
                    intersection_depth = max_x - min_x

                    W = (intersection_depth + self.clearance) / 2.0

                    eta = 1 - (2 * (i % 2))

                    for box in paragraphs[i].image_boxes:
                        box.polygon = translate(box.polygon, eta * nu * W)

                    for box in paragraphs[i - 1].image_boxes:
                        box.polygon = translate(box.polygon, -eta * nu * W)

                    infos[i] = ParagraphInfo(paragraphs[i])
                    infos[i - 1] = ParagraphInfo(paragraphs[i - 1])
                    movement = True

            iteration += 1


def _detect_preferred_side(*infos: ParagraphInfo):
    avg_center = sum([info.center[0] for info in infos]) / len(infos)
    even = sum([info.center[0] for info in infos[::2]]) / len(infos[::2])

    if even > avg_center:
        return 1
    else:
        return -1
