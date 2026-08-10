from cropgen.processing.Paragraph import Paragraph
from cropgen.processing.layout_generator.transforms import InterparagraphTransform
import numpy as np
from shapely.affinity import translate


class Refresh(InterparagraphTransform):
    """
    Recalculates the cached geometric information of each Paragraph after
    transforms have modified the polygons of their ImageBoxes:

    1. Translates all polygons so the global top-left is (0,0).
    2. Recalculates cached data: centroid, avg_rotation, top, left.
    3. Recalculates each ImageBox's corrected_centroid.
    """

    def __call__(self, *paragraphs: Paragraph) -> None:
        if not paragraphs:
            return

        all_boxes = [box for p in paragraphs for box in p.image_boxes]
        if not all_boxes:
            return

        min_x = min(box.polygon.bounds[0] for box in all_boxes)
        min_y = min(box.polygon.bounds[1] for box in all_boxes)

        for p in paragraphs:
            for box in p.image_boxes:
                box.polygon = translate(box.polygon, xoff=-min_x, yoff=-min_y)

        for p in paragraphs:

            p._calculate_total_area_and_centroid()

            theta_rad = -np.radians(-p.avg_rotation)
            cos_theta = float(np.cos(theta_rad))
            sin_theta = float(np.sin(theta_rad))

            cx_para = float(p.centroid[0])
            cy_para = float(p.centroid[1])

            for box in p.image_boxes:
                cx, cy = box.centroid()
                dx = cx - cx_para
                dy = cy - cy_para

                corrected_x = dx * cos_theta - dy * sin_theta + cx_para
                corrected_y = dx * sin_theta + dy * cos_theta + cy_para

                box.corrected_centroid = (corrected_x, corrected_y)
