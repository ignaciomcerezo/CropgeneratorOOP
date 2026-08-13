from typing import Sequence
from cropgen.processing import Paragraph, Line
from cropgen.transforms.transforms import InterparagraphTransform, ParagraphInfo
import numpy as np
from shapely.affinity import translate


class Refresh(InterparagraphTransform):
    """
    Recalculates the cached geometric information of each Paragraph after
    transforms have modified the polygons of their Imagelinees:

    1. Translates all polygons so the global top-left is (0,0).
    2. Recalculates cached data: centroid, avg_rotation, top, left.
    3. Recalculates each Imageline's corrected_centroid.
    """

    def __call__(self, *line_groups: Paragraph | Sequence[Line]) -> None:
        if not line_groups:
            return

        for line_group in line_groups:
            if not isinstance(line_group, Paragraph):
                raise ValueError(
                    "Refresh updates the geometric information of a paragraph, "
                    "therefore all values must be complete paragraphs, not just collections of "
                    "image linees. Take into account Refresh is not a proper InterparagraphTransform."
                )

        all_linees = [line for line_group in line_groups for line in line_group]
        if not all_linees:
            return

        min_x = min(line.polygon.bounds[0] for line in all_linees)
        min_y = min(line.polygon.bounds[1] for line in all_linees)

        for line_group in line_groups:
            for line in line_group:
                line.polygon = translate(line.polygon, xoff=-min_x, yoff=-min_y)

        for line_group in line_groups:

            info = ParagraphInfo(line_group)

            avg_rotation = info.avg_rotation
            centroid = info.centroid

            theta_rad = -np.radians(-avg_rotation)
            cos_theta = float(np.cos(theta_rad))
            sin_theta = float(np.sin(theta_rad))

            cx_para = float(centroid[0])
            cy_para = float(centroid[1])

            for line in line_group:
                cx, cy = line.centroid()
                dx = cx - cx_para
                dy = cy - cy_para

                corrected_x = dx * cos_theta - dy * sin_theta + cx_para
                corrected_y = dx * sin_theta + dy * cos_theta + cy_para

                line.corrected_centroid = (corrected_x, corrected_y)
